// Command stream-processor consumes ordered financial events, maintains sliding
// window features, and materializes the gateway's fixed-width Redis vector.
package main

import (
	"context"
	"crypto/tls"
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
				"fraud_stream_event_lag_seconds " + strconv.FormatFloat(math.Float64frombits(s.eventLagBits.Load()), 'g', -1, 64) + "\n" +
				"fraud_stream_materialized_total " + strconv.FormatUint(s.materialized.Load(), 10) + "\n" +
				"fraud_stream_failures_total " + strconv.FormatUint(s.failures.Load(), 10) + "\n",
		))
	})
	return mux
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	developmentInsecure := envBool("FRAUD_DEVELOPMENT_INSECURE")

	consumer, source, err := configuredConsumer(developmentInsecure)
	if err != nil {
		slog.Error("consumer configuration failed", "error", err)
		os.Exit(1)
	}
	defer consumer.Close()

	sink, probe, closeSink, err := configuredMaterializer(source == "kafka", developmentInsecure)
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
	httpAddr := envOr("STREAM_PROCESSOR_HTTP_ADDR", ":8082")
	httpServer, httpCert, httpKey, err := configuredHTTPServer(httpAddr, state.handler(), developmentInsecure)
	if err != nil {
		slog.Error("operational HTTP configuration failed", "error", err)
		os.Exit(1)
	}
	httpErr := make(chan error, 1)
	go func() {
		slog.Info("stream processor operational HTTP listening", "addr", httpAddr)
		var err error
		if developmentInsecure {
			err = httpServer.ListenAndServe()
		} else {
			err = httpServer.ListenAndServeTLS(httpCert, httpKey)
		}
		if err != nil && err != http.ErrServerClosed {
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

func configuredHTTPServer(address string, handler http.Handler, developmentInsecure bool) (*http.Server, string, string, error) {
	certFile := strings.TrimSpace(os.Getenv("STREAM_PROCESSOR_TLS_CERT_FILE"))
	keyFile := strings.TrimSpace(os.Getenv("STREAM_PROCESSOR_TLS_KEY_FILE"))
	if !developmentInsecure && (certFile == "" || keyFile == "") {
		return nil, "", "", errors.New("STREAM_PROCESSOR_TLS_CERT_FILE and STREAM_PROCESSOR_TLS_KEY_FILE are required unless FRAUD_DEVELOPMENT_INSECURE=true")
	}
	server := &http.Server{
		Addr:              address,
		Handler:           handler,
		TLSConfig:         &tls.Config{MinVersion: tls.VersionTLS13},
		ReadHeaderTimeout: 2 * time.Second,
		ReadTimeout:       5 * time.Second,
		WriteTimeout:      5 * time.Second,
		IdleTimeout:       30 * time.Second,
		MaxHeaderBytes:    16 << 10,
	}
	return server, certFile, keyFile, nil
}

func configuredConsumer(developmentInsecure bool) (ingestion.Consumer, string, error) {
	if value := strings.TrimSpace(os.Getenv("KAFKA_BROKERS")); value != "" {
		brokers := strings.Split(value, ",")
		for i := range brokers {
			brokers[i] = strings.TrimSpace(brokers[i])
		}
		config, err := configuredKafka(brokers, developmentInsecure)
		if err != nil {
			return nil, "kafka", err
		}
		consumer, err := ingestion.NewKafka(config)
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

func configuredKafka(brokers []string, developmentInsecure bool) (ingestion.KafkaConfig, error) {
	saslUsername := strings.TrimSpace(os.Getenv("KAFKA_SASL_USERNAME"))
	saslPassword := os.Getenv("KAFKA_SASL_PASSWORD")
	clientCert := strings.TrimSpace(os.Getenv("KAFKA_TLS_CERT_FILE"))
	clientKey := strings.TrimSpace(os.Getenv("KAFKA_TLS_KEY_FILE"))
	if !developmentInsecure && !((saslUsername != "" && saslPassword != "") || (clientCert != "" && clientKey != "")) {
		return ingestion.KafkaConfig{}, errors.New("production Kafka requires SCRAM credentials or a TLS client certificate")
	}
	return ingestion.KafkaConfig{
		Brokers:             brokers,
		Topic:               envOr("KAFKA_TOPIC", "financial-events"),
		GroupID:             envOr("KAFKA_GROUP_ID", "fraud-feature-aggregator"),
		MaxAttempts:         envPositiveInt("KAFKA_MAX_ATTEMPTS", 3),
		RetryBackoff:        envDuration("KAFKA_RETRY_BACKOFF", 50*time.Millisecond),
		PoisonPolicy:        envOr("KAFKA_POISON_POLICY", "halt"),
		DevelopmentInsecure: developmentInsecure,
		TLSCAFile:           strings.TrimSpace(os.Getenv("KAFKA_TLS_CA_FILE")),
		TLSServerName:       strings.TrimSpace(os.Getenv("KAFKA_TLS_SERVER_NAME")),
		TLSClientCertFile:   clientCert,
		TLSClientKeyFile:    clientKey,
		SASLUsername:        saslUsername,
		SASLPassword:        saslPassword,
		DLQTopic:            strings.TrimSpace(os.Getenv("KAFKA_DLQ_TOPIC")),
	}, nil
}

func configuredMaterializer(requireRemote, developmentInsecure bool) (aggregations.Materializer, func(context.Context) error, func() error, error) {
	if address := strings.TrimSpace(os.Getenv("REDIS_ADDR")); address != "" {
		config, err := configuredRedisMaterializer(address, developmentInsecure)
		if err != nil {
			return nil, nil, nil, err
		}
		redis, err := aggregations.NewRedisMaterializerWithConfig(config)
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

func configuredRedisMaterializer(address string, developmentInsecure bool) (aggregations.RedisConfig, error) {
	username := strings.TrimSpace(os.Getenv("REDIS_USERNAME"))
	password := os.Getenv("REDIS_PASSWORD")
	certFile := strings.TrimSpace(os.Getenv("REDIS_TLS_CERT_FILE"))
	keyFile := strings.TrimSpace(os.Getenv("REDIS_TLS_KEY_FILE"))
	if username != "" && password == "" {
		return aggregations.RedisConfig{}, errors.New("REDIS_USERNAME requires REDIS_PASSWORD")
	}
	if !developmentInsecure && password == "" && (certFile == "" || keyFile == "") {
		return aggregations.RedisConfig{}, errors.New("production Redis requires REDIS_PASSWORD or a TLS client certificate")
	}
	return aggregations.RedisConfig{
		Address:    address,
		Timeout:    envDuration("REDIS_TIMEOUT", 100*time.Millisecond),
		Username:   username,
		Password:   password,
		UseTLS:     !developmentInsecure,
		CAFile:     strings.TrimSpace(os.Getenv("REDIS_TLS_CA_FILE")),
		ServerName: strings.TrimSpace(os.Getenv("REDIS_TLS_SERVER_NAME")),
		CertFile:   certFile,
		KeyFile:    keyFile,
	}, nil
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

func envBool(key string) bool {
	value, err := strconv.ParseBool(strings.TrimSpace(os.Getenv(key)))
	return err == nil && value
}
