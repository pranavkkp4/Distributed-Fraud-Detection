// Package ingestion defines ordered, Kafka-compatible financial-event consumption.
package ingestion

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/segmentio/kafka-go"
)

type Event struct {
	ID         string    `json:"id"`
	EntityID   string    `json:"entity_id"`
	Amount     float64   `json:"amount"`
	OccurredAt time.Time `json:"occurred_at"`
	Partition  int       `json:"partition"`
	Offset     int64     `json:"offset"`
}

// Consumer receives records in Kafka partition order and commits only after Materialize succeeds.
type Consumer interface {
	Consume(context.Context, func(context.Context, Event) error) error
	Close() error
}

// MemoryConsumer is a deterministic local/testing implementation of the Kafka consumer contract.
type MemoryConsumer struct {
	mu     sync.Mutex
	events []Event
	closed bool
}

func NewMemory(events []Event) *MemoryConsumer {
	return &MemoryConsumer{events: append([]Event(nil), events...)}
}
func (m *MemoryConsumer) Consume(ctx context.Context, handle func(context.Context, Event) error) error {
	for {
		m.mu.Lock()
		if m.closed {
			m.mu.Unlock()
			return errors.New("consumer closed")
		}
		if len(m.events) == 0 {
			m.mu.Unlock()
			return nil
		}
		// Keep the event queued until the handler succeeds. This mirrors Kafka's
		// commit-after-durable-materialization contract and prevents local tests
		// from accidentally modelling at-most-once delivery.
		e := m.events[0]
		m.mu.Unlock()
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := handle(ctx, e); err != nil {
			return err
		}
		m.mu.Lock()
		if len(m.events) > 0 && m.events[0] == e {
			m.events = m.events[1:]
		}
		m.mu.Unlock()
	}
}
func (m *MemoryConsumer) Close() error { m.mu.Lock(); m.closed = true; m.mu.Unlock(); return nil }

// KafkaConfig documents the adapter boundary. A production adapter can use any
// Kafka library while preserving this at-least-once, partition-ordered interface.
type KafkaConfig struct {
	Brokers      []string
	Topic        string
	GroupID      string
	MaxAttempts  int
	RetryBackoff time.Duration
	PoisonPolicy string // "halt" (default) or "dlq"
	// DLQTopic selects the built-in synchronous Kafka dead-letter writer. It
	// must be nonempty and different from Topic when PoisonPolicy is "dlq",
	// unless a custom PoisonHandler is provided.
	DLQTopic      string
	PoisonHandler func(context.Context, []byte, error) error
}

// KafkaConsumer is an at-least-once adapter. It commits a message only after
// the supplied callback succeeds; kafka-go preserves order within each partition.
type messageReader interface {
	FetchMessage(context.Context) (kafka.Message, error)
	CommitMessages(context.Context, ...kafka.Message) error
	Close() error
}

type messageWriter interface {
	WriteMessages(context.Context, ...kafka.Message) error
	Close() error
}

type KafkaConsumer struct {
	reader        messageReader
	maxAttempts   int
	retryBackoff  time.Duration
	poisonPolicy  string
	poisonHandler func(context.Context, []byte, error) error
	dlqWriter     messageWriter
	sourceTopic   string
}

func NewKafka(config KafkaConfig) (*KafkaConsumer, error) {
	if len(config.Brokers) == 0 || config.Topic == "" || config.GroupID == "" {
		return nil, errors.New("kafka brokers, topic, and group ID are required")
	}
	if config.MaxAttempts <= 0 {
		config.MaxAttempts = 3
	}
	if config.RetryBackoff <= 0 {
		config.RetryBackoff = 50 * time.Millisecond
	}
	if config.PoisonPolicy == "" {
		config.PoisonPolicy = "halt"
	}
	if config.PoisonPolicy != "halt" && config.PoisonPolicy != "dlq" {
		return nil, errors.New("Kafka poison policy must be halt or dlq")
	}
	if config.PoisonPolicy == "dlq" && config.PoisonHandler == nil {
		config.DLQTopic = strings.TrimSpace(config.DLQTopic)
		if config.DLQTopic == "" {
			return nil, errors.New("Kafka dlq policy requires KAFKA_DLQ_TOPIC or a durable poison handler")
		}
		if config.DLQTopic == config.Topic {
			return nil, errors.New("Kafka DLQ topic must differ from source topic")
		}
	}
	consumer := &KafkaConsumer{
		reader:        kafka.NewReader(kafka.ReaderConfig{Brokers: config.Brokers, Topic: config.Topic, GroupID: config.GroupID, MinBytes: 1, MaxBytes: 10e6, CommitInterval: 0}),
		maxAttempts:   config.MaxAttempts,
		retryBackoff:  config.RetryBackoff,
		poisonPolicy:  config.PoisonPolicy,
		poisonHandler: config.PoisonHandler,
		sourceTopic:   config.Topic,
	}
	if config.PoisonPolicy == "dlq" && config.PoisonHandler == nil {
		consumer.dlqWriter = &kafka.Writer{
			Addr:            kafka.TCP(config.Brokers...),
			Topic:           config.DLQTopic,
			RequiredAcks:    kafka.RequireAll,
			Async:           false,
			BatchSize:       1,
			BatchTimeout:    0,
			MaxAttempts:     config.MaxAttempts,
			WriteBackoffMin: config.RetryBackoff,
			WriteBackoffMax: boundedBackoff(config.RetryBackoff, config.MaxAttempts),
			ReadTimeout:     5 * time.Second,
			WriteTimeout:    5 * time.Second,
		}
	}
	return consumer, nil
}
func (k *KafkaConsumer) Consume(ctx context.Context, handle func(context.Context, Event) error) error {
	if handle == nil {
		return errors.New("event handler is required")
	}
	for {
		message, err := k.fetch(ctx)
		if err != nil {
			return err
		}
		if err := k.process(ctx, message, handle); err != nil {
			return err
		}
		if err := k.commit(ctx, message); err != nil {
			return err
		}
	}
}

func (k *KafkaConsumer) fetch(ctx context.Context) (kafka.Message, error) {
	var last error
	for attempt := 1; attempt <= k.maxAttempts; attempt++ {
		message, err := k.reader.FetchMessage(ctx)
		if err == nil {
			return message, nil
		}
		last = err
		if ctx.Err() != nil {
			return kafka.Message{}, ctx.Err()
		}
		if attempt < k.maxAttempts {
			if err := waitBackoff(ctx, k.retryBackoff, attempt); err != nil {
				return kafka.Message{}, err
			}
		}
	}
	return kafka.Message{}, fmt.Errorf("fetch Kafka message after %d attempts: %w", k.maxAttempts, last)
}

func (k *KafkaConsumer) commit(ctx context.Context, message kafka.Message) error {
	var last error
	for attempt := 1; attempt <= k.maxAttempts; attempt++ {
		if err := k.reader.CommitMessages(ctx, message); err == nil {
			return nil
		} else {
			last = err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if attempt < k.maxAttempts {
			if err := waitBackoff(ctx, k.retryBackoff, attempt); err != nil {
				return err
			}
		}
	}
	return fmt.Errorf("commit Kafka message after %d attempts: %w", k.maxAttempts, last)
}

func (k *KafkaConsumer) process(ctx context.Context, message kafka.Message, handle func(context.Context, Event) error) error {
	var last error
	for attempt := 1; attempt <= k.maxAttempts; attempt++ {
		var event Event
		if err := json.Unmarshal(message.Value, &event); err != nil {
			last = err
		} else {
			event.Partition, event.Offset = message.Partition, message.Offset
			last = handle(ctx, event)
		}
		if last == nil {
			return nil
		}
		if attempt < k.maxAttempts {
			if err := waitBackoff(ctx, k.retryBackoff, attempt); err != nil {
				return err
			}
		}
	}
	if k.poisonPolicy == "dlq" {
		if k.poisonHandler != nil {
			if err := k.poisonHandler(ctx, message.Value, last); err != nil {
				return err
			}
			return nil // caller commits only after the durable custom DLQ write succeeds
		}
		if err := k.writeDLQ(ctx, message); err != nil {
			return err
		}
		return nil // caller commits only after RequireAll acknowledges the DLQ write
	}
	return &PoisonError{Partition: message.Partition, Offset: message.Offset, Err: last}
}

// writeDLQ preserves the source key and value exactly. Fixed, bounded headers
// supply only routing-safe source metadata; they intentionally omit the raw
// materialization error, which may carry infrastructure detail.
func (k *KafkaConsumer) writeDLQ(ctx context.Context, source kafka.Message) error {
	if k.dlqWriter == nil {
		return errors.New("Kafka DLQ writer is not configured")
	}
	topic := source.Topic
	if topic == "" {
		topic = k.sourceTopic
	}
	timestamp := ""
	if !source.Time.IsZero() {
		timestamp = source.Time.UTC().Format(time.RFC3339Nano)
	}
	dlq := kafka.Message{
		Key:   append([]byte(nil), source.Key...),
		Value: append([]byte(nil), source.Value...),
		Headers: []kafka.Header{
			{Key: "x-dfd-source-topic", Value: []byte(safeMetadata(topic, 249))},
			{Key: "x-dfd-source-partition", Value: []byte(strconv.Itoa(source.Partition))},
			{Key: "x-dfd-source-offset", Value: []byte(strconv.FormatInt(source.Offset, 10))},
			{Key: "x-dfd-source-timestamp", Value: []byte(timestamp)},
			{Key: "x-dfd-dlq-reason", Value: []byte("retry-exhausted")},
		},
	}
	if err := k.dlqWriter.WriteMessages(ctx, dlq); err != nil {
		return fmt.Errorf("write Kafka DLQ record: %w", err)
	}
	return nil
}

func safeMetadata(value string, max int) string {
	value = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, value)
	if len(value) > max {
		return value[:max]
	}
	return value
}

func boundedBackoff(base time.Duration, attempts int) time.Duration {
	backoff := base
	for range attempts - 1 {
		if backoff >= time.Second/2 {
			return time.Second
		}
		backoff *= 2
	}
	return backoff
}

func waitBackoff(ctx context.Context, base time.Duration, attempt int) error {
	backoff := base
	for range attempt - 1 {
		if backoff >= time.Second/2 {
			backoff = time.Second
			break
		}
		backoff *= 2
	}
	timer := time.NewTimer(backoff)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

type PoisonError struct {
	Partition int
	Offset    int64
	Err       error
}

func (e *PoisonError) Error() string {
	return fmt.Sprintf("poison Kafka record at partition %d offset %d", e.Partition, e.Offset)
}
func (e *PoisonError) Unwrap() error { return e.Err }
func (k *KafkaConsumer) Close() error {
	if k == nil {
		return nil
	}
	var result error
	if k.reader != nil {
		result = k.reader.Close()
	}
	if k.dlqWriter != nil {
		result = errors.Join(result, k.dlqWriter.Close())
	}
	return result
}
