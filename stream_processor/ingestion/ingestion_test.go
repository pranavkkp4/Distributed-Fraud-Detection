package ingestion

import (
	"context"
	"crypto/tls"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/segmentio/kafka-go"
)

func TestMemoryConsumerRetainsFailedEventForRetry(t *testing.T) {
	event := Event{ID: "event-1", EntityID: "account", Amount: 1, OccurredAt: time.Now()}
	consumer := NewMemory([]Event{event})
	attempts := 0
	err := consumer.Consume(context.Background(), func(_ context.Context, got Event) error {
		attempts++
		if got.ID != event.ID {
			t.Fatalf("event = %#v", got)
		}
		return errors.New("durable sink unavailable")
	})
	if err == nil || attempts != 1 {
		t.Fatalf("first consume = %v, attempts=%d", err, attempts)
	}
	err = consumer.Consume(context.Background(), func(_ context.Context, got Event) error {
		attempts++
		if got.ID != event.ID {
			t.Fatalf("redelivery = %#v", got)
		}
		return nil
	})
	if err != nil || attempts != 2 {
		t.Fatalf("retry consume = %v, attempts=%d", err, attempts)
	}
}

func TestKafkaConfigRequiresDurableDLQHandler(t *testing.T) {
	if _, err := NewKafka(KafkaConfig{Brokers: []string{"localhost:9092"}, Topic: "events", GroupID: "group", PoisonPolicy: "dlq"}); err == nil {
		t.Fatal("expected DLQ topic validation error")
	}
	if _, err := NewKafka(KafkaConfig{Brokers: []string{"localhost:9092"}, Topic: "events", GroupID: "group", PoisonPolicy: "dlq", DLQTopic: "events"}); err == nil {
		t.Fatal("expected same-topic DLQ validation error")
	}
}

func TestKafkaTransportDefaultsToTLS13AndValidatesCredentials(t *testing.T) {
	dialer, err := newKafkaDialer(KafkaConfig{})
	if err != nil {
		t.Fatalf("production Kafka transport: %v", err)
	}
	if dialer.TLS == nil || dialer.TLS.MinVersion != tls.VersionTLS13 {
		t.Fatalf("Kafka TLS config = %#v, want TLS 1.3", dialer.TLS)
	}
	if dialer.TLS.RootCAs != nil {
		t.Fatal("empty CA file should retain platform root verification")
	}

	for _, config := range []KafkaConfig{
		{SASLUsername: "fraud"},
		{SASLPassword: "secret"},
		{TLSClientCertFile: "client.pem"},
		{TLSClientKeyFile: "client-key.pem"},
		{DevelopmentInsecure: true, TLSCAFile: "ca.pem"},
		{DevelopmentInsecure: true, SASLUsername: "fraud", SASLPassword: "secret"},
	} {
		if _, err := newKafkaDialer(config); err == nil {
			t.Fatalf("partial Kafka credentials were accepted: %#v", config)
		}
	}
}

func TestKafkaTransportConfiguresSCRAMAndValidatesCA(t *testing.T) {
	dialer, err := newKafkaDialer(KafkaConfig{SASLUsername: "fraud", SASLPassword: "secret", TLSServerName: "broker.internal"})
	if err != nil {
		t.Fatalf("configure SCRAM: %v", err)
	}
	if dialer.SASLMechanism == nil || dialer.SASLMechanism.Name() != "SCRAM-SHA-512" || dialer.TLS.ServerName != "broker.internal" {
		t.Fatalf("Kafka dialer did not preserve TLS/SASL settings: %#v", dialer)
	}

	missingCA := filepath.Join(t.TempDir(), "missing-ca.pem")
	if _, err := newKafkaDialer(KafkaConfig{TLSCAFile: missingCA}); err == nil || !strings.Contains(err.Error(), "read Kafka TLS CA") {
		t.Fatalf("missing Kafka CA error = %v", err)
	}
	invalidCA := filepath.Join(t.TempDir(), "invalid-ca.pem")
	if err := os.WriteFile(invalidCA, []byte("not a certificate"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := newKafkaDialer(KafkaConfig{TLSCAFile: invalidCA}); err == nil || !strings.Contains(err.Error(), "contains no certificates") {
		t.Fatalf("invalid Kafka CA error = %v", err)
	}
}

func TestKafkaDevelopmentInsecureOptInOmitsTLS(t *testing.T) {
	dialer, err := newKafkaDialer(KafkaConfig{DevelopmentInsecure: true})
	if err != nil {
		t.Fatalf("development Kafka transport: %v", err)
	}
	if dialer.TLS != nil {
		t.Fatal("development-insecure Kafka transport unexpectedly enabled TLS")
	}
}

func TestBuiltInDLQWriterIsSynchronousAndRequireAll(t *testing.T) {
	consumer, err := NewKafka(KafkaConfig{
		Brokers:      []string{"localhost:9092"},
		Topic:        "events",
		DLQTopic:     "events-dlq",
		GroupID:      "group",
		PoisonPolicy: "dlq",
		MaxAttempts:  2,
		RetryBackoff: time.Millisecond,
		SASLUsername: "fraud",
		SASLPassword: "secret",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer consumer.Close()
	writer, ok := consumer.dlqWriter.(*kafka.Writer)
	if !ok || writer.Async || writer.RequiredAcks != kafka.RequireAll || writer.MaxAttempts != 2 || writer.BatchSize != 1 {
		t.Fatalf("unsafe DLQ writer: %#v", consumer.dlqWriter)
	}
	transport, ok := writer.Transport.(*kafka.Transport)
	if !ok || transport.TLS == nil || transport.TLS.MinVersion != tls.VersionTLS13 || transport.SASL == nil || transport.SASL.Name() != "SCRAM-SHA-512" {
		t.Fatalf("DLQ writer did not inherit secure Kafka transport: %#v", writer.Transport)
	}
}

type scriptedReader struct {
	message        kafka.Message
	fetchFailures  int
	commitFailures int
	fetches        int
	commits        int
}

func (r *scriptedReader) FetchMessage(context.Context) (kafka.Message, error) {
	r.fetches++
	if r.fetches <= r.fetchFailures {
		return kafka.Message{}, errors.New("temporary fetch failure")
	}
	return r.message, nil
}

func (r *scriptedReader) CommitMessages(context.Context, ...kafka.Message) error {
	r.commits++
	if r.commits <= r.commitFailures {
		return errors.New("temporary commit failure")
	}
	return nil
}

func (*scriptedReader) Close() error { return nil }

func TestKafkaFetchAndCommitUseBoundedRetries(t *testing.T) {
	reader := &scriptedReader{
		message:        kafka.Message{Partition: 2, Offset: 9, Value: []byte(`{"id":"e"}`)},
		fetchFailures:  2,
		commitFailures: 2,
	}
	consumer := &KafkaConsumer{reader: reader, maxAttempts: 3, retryBackoff: time.Microsecond}
	message, err := consumer.fetch(context.Background())
	if err != nil || message.Offset != 9 || reader.fetches != 3 {
		t.Fatalf("fetch = %#v, %v, attempts=%d", message, err, reader.fetches)
	}
	if err := consumer.commit(context.Background(), message); err != nil || reader.commits != 3 {
		t.Fatalf("commit = %v, attempts=%d", err, reader.commits)
	}
}

func TestKafkaProcessingRetriesAndDurableDLQ(t *testing.T) {
	valid := kafka.Message{
		Partition: 1,
		Offset:    4,
		Value:     []byte(`{"id":"e","entity_id":"account","amount":1,"occurred_at":"2026-01-01T00:00:00Z"}`),
	}
	consumer := &KafkaConsumer{maxAttempts: 3, retryBackoff: time.Microsecond, poisonPolicy: "halt"}
	attempts := 0
	if err := consumer.process(context.Background(), valid, func(context.Context, Event) error {
		attempts++
		if attempts < 3 {
			return errors.New("temporary materializer failure")
		}
		return nil
	}); err != nil || attempts != 3 {
		t.Fatalf("processing retry = %v, attempts=%d", err, attempts)
	}

	dlqWrites := 0
	consumer = &KafkaConsumer{
		maxAttempts:  2,
		retryBackoff: time.Microsecond,
		poisonPolicy: "dlq",
		poisonHandler: func(_ context.Context, payload []byte, cause error) error {
			dlqWrites++
			if string(payload) != "not-json" || cause == nil {
				t.Fatalf("DLQ payload=%q cause=%v", payload, cause)
			}
			return nil
		},
	}
	if err := consumer.process(context.Background(), kafka.Message{Value: []byte("not-json")}, func(context.Context, Event) error {
		return nil
	}); err != nil || dlqWrites != 1 {
		t.Fatalf("DLQ processing = %v, writes=%d", err, dlqWrites)
	}
}

type capturedWriter struct {
	messages []kafka.Message
	err      error
	closed   bool
}

func (w *capturedWriter) WriteMessages(_ context.Context, messages ...kafka.Message) error {
	w.messages = append(w.messages, messages...)
	return w.err
}
func (w *capturedWriter) Close() error { w.closed = true; return nil }

func TestBuiltInDLQPreservesSourceAndPrecedesCommit(t *testing.T) {
	message := kafka.Message{
		Topic:     "financial-events",
		Partition: 3,
		Offset:    42,
		Key:       []byte("account-7"),
		Value:     []byte("not-json"),
		Time:      time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC),
	}
	reader := &scriptedReader{message: message}
	writer := &capturedWriter{}
	consumer := &KafkaConsumer{reader: reader, dlqWriter: writer, sourceTopic: "financial-events", maxAttempts: 1, retryBackoff: time.Microsecond, poisonPolicy: "dlq"}
	if err := consumer.process(context.Background(), message, func(context.Context, Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(writer.messages) != 1 {
		t.Fatalf("DLQ messages = %d", len(writer.messages))
	}
	written := writer.messages[0]
	if string(written.Key) != string(message.Key) || string(written.Value) != string(message.Value) {
		t.Fatalf("DLQ did not preserve key/value: %#v", written)
	}
	headers := map[string]string{}
	for _, header := range written.Headers {
		headers[header.Key] = string(header.Value)
	}
	if headers["x-dfd-source-topic"] != "financial-events" || headers["x-dfd-source-partition"] != "3" || headers["x-dfd-source-offset"] != "42" || headers["x-dfd-dlq-reason"] != "retry-exhausted" {
		t.Fatalf("unexpected DLQ headers: %#v", headers)
	}
	if reader.commits != 0 {
		t.Fatal("source must not commit from process; Consume commits after DLQ acknowledgement")
	}
	if err := consumer.Close(); err != nil || !writer.closed {
		t.Fatalf("close = %v, writer closed = %v", err, writer.closed)
	}
}

func TestBuiltInDLQWriteFailureLeavesSourceUncommitted(t *testing.T) {
	message := kafka.Message{Topic: "financial-events", Value: []byte("not-json")}
	writer := &capturedWriter{err: errors.New("DLQ unavailable")}
	consumer := &KafkaConsumer{dlqWriter: writer, sourceTopic: "financial-events", maxAttempts: 1, retryBackoff: time.Microsecond, poisonPolicy: "dlq"}
	if err := consumer.process(context.Background(), message, func(context.Context, Event) error { return nil }); err == nil {
		t.Fatal("expected DLQ write error")
	}
}

type oneMessageReader struct {
	message   kafka.Message
	served    bool
	commits   int
	committed bool
	writer    *capturedWriter
}

func (r *oneMessageReader) FetchMessage(context.Context) (kafka.Message, error) {
	if r.served {
		return kafka.Message{}, context.Canceled
	}
	r.served = true
	return r.message, nil
}
func (r *oneMessageReader) CommitMessages(context.Context, ...kafka.Message) error {
	r.commits++
	if len(r.writer.messages) != 1 {
		return errors.New("source commit happened before durable DLQ write")
	}
	r.committed = true
	return nil
}
func (*oneMessageReader) Close() error { return nil }

func TestBuiltInDLQAcknowledgementPrecedesSourceCommit(t *testing.T) {
	message := kafka.Message{Topic: "financial-events", Partition: 1, Offset: 2, Key: []byte("key"), Value: []byte("not-json")}
	writer := &capturedWriter{}
	reader := &oneMessageReader{message: message, writer: writer}
	consumer := &KafkaConsumer{reader: reader, dlqWriter: writer, sourceTopic: "financial-events", maxAttempts: 1, retryBackoff: time.Microsecond, poisonPolicy: "dlq"}
	err := consumer.Consume(context.Background(), func(context.Context, Event) error { return nil })
	if !errors.Is(err, context.Canceled) || !reader.committed || reader.commits != 1 {
		t.Fatalf("consume = %v, committed=%v commits=%d", err, reader.committed, reader.commits)
	}
}
