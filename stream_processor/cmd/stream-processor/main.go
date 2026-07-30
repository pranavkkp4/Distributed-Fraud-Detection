// Command stream-processor consumes ordered financial events, maintains sliding
// window features, and materializes the gateway's fixed-width Redis vector.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/pranavkkp4/distributed-fraud-detection/stream_processor/aggregations"
	"github.com/pranavkkp4/distributed-fraud-detection/stream_processor/ingestion"
)

type operationalState struct {
	ready        atomic.Bool
	attempts     atomic.Uint64
	events       atomic.Uint64
	materialized atomic.Uint64
	failures     atomic.Uint64
	eventLagBits atomic.Uint64
	probe        func(context.Context) error
}

func (s *operationalState) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if !s.ready.Load() {
			http.Error(w, "not ready", http.StatusServiceUnavailable)
			return
		}
		if s.probe != nil {
			ctx, cancel := context.WithTimeout(r.Context(), 200*time.Millisecond)
			defer cancel()
			if err := s.probe(ctx); err != nil {
				http.Error(w, "materializer unavailable", http.StatusServiceUnavailable)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = w.Write([]byte(
			"fraud_stream_attempts_total " + strconv.FormatUint(s.attempts.Load(), 10) + "\n" +
				"fraud_stream_processed_total " + strconv.FormatUint(s.events.Load(), 10) + "\n" +
				"fraud_stream_errors_total " + strconv.FormatUint(s.failures.Load(), 10) + "\n" +
				"fraud_stream_event_lag_seconds " + strconv.FormatFloat(math.Float64frombits(s.eventLagBits.Load()), 'g', -1, 64) + "\n" +
				"fraud_stream_events_total " + strconv.FormatUint(s.events.Load(), 10) + "\n" +
				"fraud_stream_materialized_total " + strconv.FormatUint(s.materialized.Load(), 10) + "\n" +
				"fraud_stream_failures_total " + strconv.FormatUint(s.failures.Load(), 10) + "\n",
		))
	})
	return mux
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	consumer, source, err := configuredConsumer()
	if err != nil {
		slog.Error("consumer configuration failed", "error", err)
		os.Exit(1)
	}
	defer consumer.Close()

	sink, probe, closeSink, err := configuredMaterializer(source == "kafka")
	if err != nil {
		slog.Error("materializer configuration failed", "error", err)
		os.Exit(1)
	}
	if closeSink != nil {
		defer closeSink()
	}
	window := aggregations.NewWithLimits(
		envDuration("STREAM_WINDOW", 5*time.Minute),
		envPositiveInt("STREAM_MAX_ENTITIES", 100_000),
		envPositiveInt("STREAM_MAX_EVENTS", 1_000_000),
	)
	state := &operationalState{probe: probe}
	startupCtx, startupCancel := context.WithTimeout(context.Background(), envDuration("STREAM_STARTUP_TIMEOUT", 5*time.Second))
	if probe != nil {
		err = probe(startupCtx)
	}
	startupCancel()
	if err != nil {
		slog.Error("materializer dependency probe failed", "error", err)
		os.Exit(1)
	}
	state.ready.Store(true)
	httpAddr := envOr("STREAM_PROCESSOR_HTTP_ADDR", ":9092")
	httpServer := &http.Server{Addr: httpAddr, Handler: state.handler(), ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	httpErr := make(chan error, 1)
	go func() {
		slog.Info("stream processor operational HTTP listening", "addr", httpAddr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			httpErr <- err
		}
	}()

	consumeDone := make(chan error, 1)
	go func() {
		consumeDone <- consumer.Consume(ctx, func(eventCtx context.Context, event ingestion.Event) error {
			state.attempts.Add(1)
			lag := time.Since(event.OccurredAt).Seconds()
			if lag < 0 {
				lag = 0
			}
			state.eventLagBits.Store(math.Float64bits(lag))
			_, changed, err := window.ApplyAndMaterialize(eventCtx, event, sink)
			if err != nil {
				state.ready.Store(false)
				state.failures.Add(1)
				return err
			}
			state.events.Add(1)
			state.ready.Store(true)
			if changed {
				state.materialized.Add(1)
			}
			return nil
		})
	}()
	slog.Info("stream processor ready", "source", source, "redis", os.Getenv("REDIS_ADDR") != "")

	runtimeFailed := false
	select {
	case <-ctx.Done():
	case err := <-consumeDone:
		if err != nil && !errors.Is(err, context.Canceled) {
			state.failures.Add(1)
			slog.Error("consumer failed", "error", err)
			runtimeFailed = true
		}
	case err := <-httpErr:
		slog.Error("operational HTTP failed", "error", err)
		runtimeFailed = true
	}
	state.ready.Store(false)
	cancel()
	_ = consumer.Close()
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	_ = httpServer.Shutdown(shutdownCtx)
	if runtimeFailed {
		if closeSink != nil {
			_ = closeSink()
		}
		os.Exit(1)
	}
}

func configuredConsumer() (ingestion.Consumer, string, error) {
	if value := strings.TrimSpace(os.Getenv("KAFKA_BROKERS")); value != "" {
		brokers := strings.Split(value, ",")
		for i := range brokers {
			brokers[i] = strings.TrimSpace(brokers[i])
		}
		consumer, err := ingestion.NewKafka(ingestion.KafkaConfig{
			Brokers:      brokers,
			Topic:        envOr("KAFKA_TOPIC", "financial-events"),
			GroupID:      envOr("KAFKA_GROUP_ID", "fraud-feature-aggregator"),
			MaxAttempts:  envPositiveInt("KAFKA_MAX_ATTEMPTS", 3),
			RetryBackoff: envDuration("KAFKA_RETRY_BACKOFF", 50*time.Millisecond),
			PoisonPolicy: envOr("KAFKA_POISON_POLICY", "halt"),
			DLQTopic:     strings.TrimSpace(os.Getenv("KAFKA_DLQ_TOPIC")),
		})
		return consumer, "kafka", err
	}

	var events []ingestion.Event
	decoder := json.NewDecoder(os.Stdin)
	if err := decoder.Decode(&events); err != nil {
		if errors.Is(err, io.EOF) {
			return ingestion.NewMemory(nil), "stdin", nil
		}
		return nil, "stdin", err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, "stdin", errors.New("stdin contains trailing JSON")
	}
	return ingestion.NewMemory(events), "stdin", nil
}

func configuredMaterializer(requireRemote bool) (aggregations.Materializer, func(context.Context) error, func() error, error) {
	if address := strings.TrimSpace(os.Getenv("REDIS_ADDR")); address != "" {
		redis, err := aggregations.NewRedisMaterializer(address, envDuration("REDIS_TIMEOUT", 100*time.Millisecond))
		if err != nil {
			return nil, nil, nil, err
		}
		return redis, redis.Probe, redis.Close, nil
	}
	if requireRemote {
		return nil, nil, nil, errors.New("REDIS_ADDR is required when Kafka ingestion is configured")
	}
	return aggregations.NewMemoryMaterializer(), nil, nil, nil
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	if value, err := time.ParseDuration(os.Getenv(key)); err == nil && value > 0 {
		return value
	}
	return fallback
}

func envPositiveInt(key string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(os.Getenv(key)))
	if err != nil || value < 1 {
		return fallback
	}
	return value
}
