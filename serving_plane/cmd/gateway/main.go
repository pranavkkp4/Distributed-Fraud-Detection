package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/feature_store"
	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/gateway"
	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/worker"
)

func main() {
	if len(os.Args) == 2 && os.Args[1] == "-healthcheck" {
		if err := probeHTTPReady(
			firstEnv("FRAUD_GATEWAY_ADDR", "GATEWAY_ADDR"),
			":8080",
			envBool("FRAUD_DEVELOPMENT_INSECURE"),
			os.Getenv("FRAUD_GATEWAY_TLS_CA_FILE"),
			os.Getenv("FRAUD_GATEWAY_TLS_SERVER_NAME"),
		); err != nil {
			slog.Error("gateway healthcheck failed", "error", err)
			os.Exit(1)
		}
		return
	}
	developmentInsecure := envBool("FRAUD_DEVELOPMENT_INSECURE")
	authToken, err := requiredSecret("FRAUD_AUTH_TOKEN", developmentInsecure)
	if err != nil {
		slog.Error("gateway configuration failed", "error", err)
		os.Exit(1)
	}
	var features featurestore.Store
	var redisStore *featurestore.Redis
	if address := strings.TrimSpace(os.Getenv("REDIS_ADDR")); address != "" {
		var err error
		redisStore, err = featurestore.NewRedis(address, worker.ModelFeatureWidth, duration("REDIS_TIMEOUT", 5*time.Millisecond))
		if err != nil {
			slog.Error("Redis feature store setup", "error", err)
			os.Exit(1)
		}
		defer redisStore.Close()
		features = redisStore
		if developmentInsecure {
			features = featurestore.Fallback{Primary: redisStore, Local: developmentFeatures()}
		}
	} else {
		if !developmentInsecure {
			slog.Error("gateway configuration failed", "error", "REDIS_ADDR is required unless FRAUD_DEVELOPMENT_INSECURE=true")
			os.Exit(1)
		}
		features = developmentFeatures()
	}

	engine, clients, err := configuredEngine(developmentInsecure)
	if err != nil {
		slog.Error("worker pool setup", "error", err)
		os.Exit(1)
	}
	for _, client := range clients {
		defer client.Close()
	}

	service, err := gateway.New(features, engine, gateway.Config{
		AuthToken:        authToken,
		Timeout:          duration("FRAUD_TIMEOUT", 40*time.Millisecond),
		FeatureTimeout:   duration("FRAUD_FEATURE_TIMEOUT", 10*time.Millisecond),
		QueueDelay:       duration("FRAUD_QUEUE_DELAY", 2*time.Millisecond),
		InferenceTimeout: duration("FRAUD_INFERENCE_TIMEOUT", 20*time.Millisecond),
	}, slog.Default())
	if err != nil {
		slog.Error("gateway setup", "error", err)
		os.Exit(1)
	}
	defer service.Close()
	startupCtx, startupCancel := context.WithTimeout(context.Background(), duration("FRAUD_STARTUP_TIMEOUT", 5*time.Second))
	if prober, ok := features.(featurestore.Prober); ok {
		err = prober.Probe(startupCtx)
	}
	if err == nil {
		err = engine.Healthy(startupCtx)
	}
	startupCancel()
	if err != nil {
		slog.Error("gateway dependency probe failed", "error", err)
		os.Exit(1)
	}

	address := firstEnv("FRAUD_GATEWAY_ADDR", "GATEWAY_ADDR")
	if address == "" {
		address = ":8080"
	}
	server := &http.Server{Addr: address, Handler: service.Handler(), TLSConfig: &tls.Config{MinVersion: tls.VersionTLS13}, ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	httpCert, httpKey := os.Getenv("FRAUD_GATEWAY_TLS_CERT_FILE"), os.Getenv("FRAUD_GATEWAY_TLS_KEY_FILE")
	if !developmentInsecure && (httpCert == "" || httpKey == "") {
		slog.Error("gateway TLS configuration failed", "error", "FRAUD_GATEWAY_TLS_CERT_FILE and FRAUD_GATEWAY_TLS_KEY_FILE are required")
		os.Exit(1)
	}
	serveErrors := make(chan error, 1)
	go func() {
		slog.Info("gateway listening", "addr", address, "remote_workers", len(clients))
		var serveErr error
		if developmentInsecure {
			serveErr = server.ListenAndServe()
		} else {
			serveErr = server.ListenAndServeTLS(httpCert, httpKey)
		}
		if serveErr != nil && serveErr != http.ErrServerClosed {
			serveErrors <- serveErr
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	serveFailed := false
	select {
	case <-ctx.Done():
	case err := <-serveErrors:
		slog.Error("gateway HTTP failed", "error", err)
		serveFailed = true
		stop()
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = server.Shutdown(shutdownCtx)
	if serveFailed {
		os.Exit(1)
	}
}

func developmentFeatures() *featurestore.Memory {
	memory := featurestore.NewMemory(worker.ModelFeatureWidth)
	features := make([]float64, worker.ModelFeatureWidth)
	copy(features, []float64{0.1, 0.3, 0.2})
	if err := memory.Put("demo-account", features); err != nil {
		panic(err) // fixed-width static development fixture; a failure is a programmer error
	}
	return memory
}

func configuredEngine(developmentInsecure bool) (worker.Inference, []*worker.GRPCClient, error) {
	addresses := firstEnv("FRAUD_WORKER_ADDRS", "FRAUD_WORKER_ADDR")
	if strings.TrimSpace(addresses) == "" {
		if !developmentInsecure {
			return nil, nil, errors.New("remote worker endpoints are required outside development insecure mode")
		}
		return worker.DefaultCPUScorer(), nil, nil
	}
	workerToken, err := requiredSecret("FRAUD_WORKER_AUTH_TOKEN", developmentInsecure)
	if err != nil {
		return nil, nil, err
	}
	parts := strings.Split(addresses, ",")
	workers := make([]worker.Inference, 0, len(parts))
	clients := make([]*worker.GRPCClient, 0, len(parts))
	for _, rawAddress := range parts {
		address := strings.TrimSpace(rawAddress)
		if address == "" {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), duration("FRAUD_WORKER_DIAL_TIMEOUT", 2*time.Second))
		client, err := worker.DialWithTransport(ctx, address, workerToken, worker.TransportConfig{
			Insecure:   developmentInsecure,
			ServerName: os.Getenv("FRAUD_WORKER_TLS_SERVER_NAME"),
			CAFile:     os.Getenv("FRAUD_WORKER_TLS_CA_FILE"),
			CertFile:   os.Getenv("FRAUD_GATEWAY_MTLS_CERT_FILE"),
			KeyFile:    os.Getenv("FRAUD_GATEWAY_MTLS_KEY_FILE"),
		})
		cancel()
		if err != nil {
			slog.Warn("skipping worker endpoint", "addr", address, "error", err)
			continue
		}
		clients = append(clients, client)
		workers = append(workers, client)
	}
	pool, err := worker.NewPool(workers, duration("FRAUD_WORKER_PROBE_INTERVAL", time.Second))
	if err != nil {
		for _, client := range clients {
			_ = client.Close()
		}
		return nil, nil, err
	}
	return pool, clients, nil
}

func requiredSecret(key string, developmentInsecure bool) (string, error) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" && !developmentInsecure {
		return "", fmt.Errorf("%s is required unless FRAUD_DEVELOPMENT_INSECURE=true", key)
	}
	return value, nil
}

func envBool(key string) bool {
	value, err := strconv.ParseBool(strings.TrimSpace(os.Getenv(key)))
	return err == nil && value
}

// probeHTTPReady verifies the local readiness endpoint using the same transport
// policy as the listener. Production HTTPS probes verify certificates using the
// system root store unless an explicit CA bundle is configured.
func probeHTTPReady(configuredAddress, fallback string, developmentInsecure bool, caFile, serverName string) error {
	address := strings.TrimSpace(configuredAddress)
	if address == "" {
		address = fallback
	}
	_, port, err := net.SplitHostPort(address)
	if err != nil || port == "" {
		return errors.New("healthcheck address must include a port")
	}
	scheme := "http"
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if !developmentInsecure {
		roots, err := probeRoots(caFile)
		if err != nil {
			return err
		}
		scheme = "https"
		transport.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS13, RootCAs: roots, ServerName: strings.TrimSpace(serverName)}
	}
	client := &http.Client{Timeout: 500 * time.Millisecond, Transport: transport}
	response, err := client.Get(scheme + "://127.0.0.1:" + port + "/readyz")
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("readiness returned %s", response.Status)
	}
	return nil
}

func probeRoots(caFile string) (*x509.CertPool, error) {
	caFile = strings.TrimSpace(caFile)
	if caFile == "" {
		// A nil RootCAs value deliberately uses the platform trust store.
		return nil, nil
	}
	pem, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("read gateway healthcheck TLS CA: %w", err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(pem) {
		return nil, errors.New("gateway healthcheck TLS CA contains no certificates")
	}
	return roots, nil
}

func firstEnv(keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return ""
}

func duration(key string, fallback time.Duration) time.Duration {
	if value, err := time.ParseDuration(os.Getenv(key)); err == nil && value > 0 {
		return value
	}
	return fallback
}
