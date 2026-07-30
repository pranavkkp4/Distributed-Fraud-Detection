// Command worker serves the gRPC fraud inference worker protocol and a small
// operational HTTP endpoint for Kubernetes probes and Prometheus scraping.
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
	"sync/atomic"
	"syscall"
	"time"

	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/worker"
)

type measuredEngine struct {
	delegate        worker.Inference
	batches         atomic.Uint64
	items           atomic.Uint64
	failures        atomic.Uint64
	durationNanos   atomic.Uint64
	durationBuckets [7]atomic.Uint64
}

func (m *measuredEngine) Infer(ctx context.Context, requests []worker.Request) ([]worker.Result, error) {
	started := time.Now()
	defer func() { m.observeDuration(time.Since(started)) }()
	m.batches.Add(1)
	m.items.Add(uint64(len(requests)))
	results, err := m.delegate.Infer(ctx, requests)
	if err != nil {
		m.failures.Add(1)
	}
	return results, err
}

var workerDurationBounds = [...]time.Duration{100 * time.Microsecond, 500 * time.Microsecond, time.Millisecond, 5 * time.Millisecond, 10 * time.Millisecond, 20 * time.Millisecond}

func (m *measuredEngine) observeDuration(duration time.Duration) {
	m.durationNanos.Add(uint64(duration))
	for i, bound := range workerDurationBounds {
		if duration <= bound {
			m.durationBuckets[i].Add(1)
		}
	}
	m.durationBuckets[len(m.durationBuckets)-1].Add(1)
}

func (m *measuredEngine) Healthy(ctx context.Context) error {
	return m.delegate.Healthy(ctx)
}

func (m *measuredEngine) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 100*time.Millisecond)
		defer cancel()
		if err := m.Healthy(ctx); err != nil {
			http.Error(w, "not ready", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		var output strings.Builder
		fmt.Fprintf(&output, "fraud_worker_requests_total %d\n", m.items.Load())
		fmt.Fprintf(&output, "fraud_worker_errors_total %d\n", m.failures.Load())
		fmt.Fprintf(&output, "fraud_worker_batches_total %d\n", m.batches.Load())
		labels := [...]string{"0.0001", "0.0005", "0.001", "0.005", "0.01", "0.02", "+Inf"}
		for i, label := range labels {
			fmt.Fprintf(&output, "fraud_worker_request_duration_seconds_bucket{le=%q} %d\n", label, m.durationBuckets[i].Load())
		}
		fmt.Fprintf(&output, "fraud_worker_request_duration_seconds_sum %g\n", float64(m.durationNanos.Load())/float64(time.Second))
		fmt.Fprintf(&output, "fraud_worker_request_duration_seconds_count %d\n", m.batches.Load())
		_, _ = w.Write([]byte(output.String()))
	})
	return mux
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "-healthcheck" {
		if err := probeReady(
			envOr("FRAUD_WORKER_HTTP_ADDR", ":9091"),
			envBool("FRAUD_DEVELOPMENT_INSECURE"),
			os.Getenv("FRAUD_WORKER_HTTP_TLS_CA_FILE"),
			os.Getenv("FRAUD_WORKER_HTTP_TLS_SERVER_NAME"),
		); err != nil {
			slog.Error("worker healthcheck failed", "error", err)
			os.Exit(1)
		}
		return
	}
	grpcAddr := envOr("FRAUD_WORKER_ADDR", ":50051")
	httpAddr := envOr("FRAUD_WORKER_HTTP_ADDR", ":9091")
	developmentInsecure := envBool("FRAUD_DEVELOPMENT_INSECURE")
	authToken := strings.TrimSpace(os.Getenv("FRAUD_WORKER_AUTH_TOKEN"))
	if authToken == "" && !developmentInsecure {
		slog.Error("worker configuration failed", "error", "FRAUD_WORKER_AUTH_TOKEN is required unless FRAUD_DEVELOPMENT_INSECURE=true")
		os.Exit(1)
	}
	listener, err := net.Listen("tcp", grpcAddr)
	if err != nil {
		slog.Error("gRPC listen failed", "error", err)
		os.Exit(1)
	}

	engine := &measuredEngine{delegate: worker.DefaultCPUScorer()}
	grpcServer, grpcServeErrors, err := worker.ServeWithTransportObserved(listener, engine, authToken, worker.TransportConfig{
		Insecure:          developmentInsecure,
		CAFile:            os.Getenv("FRAUD_GATEWAY_MTLS_CA_FILE"),
		CertFile:          os.Getenv("FRAUD_WORKER_TLS_CERT_FILE"),
		KeyFile:           os.Getenv("FRAUD_WORKER_TLS_KEY_FILE"),
		RequireClientCert: !developmentInsecure,
	})
	if err != nil {
		_ = listener.Close()
		slog.Error("worker TLS configuration failed", "error", err)
		os.Exit(1)
	}
	httpServer := &http.Server{Addr: httpAddr, Handler: engine.handler(), TLSConfig: &tls.Config{MinVersion: tls.VersionTLS13}, ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	httpCert, httpKey := os.Getenv("FRAUD_WORKER_HTTP_TLS_CERT_FILE"), os.Getenv("FRAUD_WORKER_HTTP_TLS_KEY_FILE")
	if !developmentInsecure && (httpCert == "" || httpKey == "") {
		grpcServer.Stop()
		_ = listener.Close()
		slog.Error("worker HTTP TLS configuration failed", "error", "FRAUD_WORKER_HTTP_TLS_CERT_FILE and FRAUD_WORKER_HTTP_TLS_KEY_FILE are required")
		os.Exit(1)
	}
	serveErrors := make(chan error, 1)
	go func() {
		slog.Info("worker operational HTTP listening", "addr", httpAddr)
		var serveErr error
		if developmentInsecure {
			serveErr = httpServer.ListenAndServe()
		} else {
			serveErr = httpServer.ListenAndServeTLS(httpCert, httpKey)
		}
		if serveErr != nil && serveErr != http.ErrServerClosed {
			serveErrors <- serveErr
		}
	}()
	slog.Info("gRPC worker listening", "addr", grpcAddr)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	serveFailed := false
	select {
	case <-ctx.Done():
	case err := <-serveErrors:
		slog.Error("worker operational HTTP failed", "error", err)
		serveFailed = true
		stop()
	case err := <-grpcServeErrors:
		if err == nil {
			err = errors.New("gRPC server stopped unexpectedly")
		}
		slog.Error("worker gRPC failed", "error", err)
		serveFailed = true
		stop()
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)
	done := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(done)
	}()
	select {
	case <-done:
	case <-shutdownCtx.Done():
		grpcServer.Stop()
	}
	if serveFailed {
		os.Exit(1)
	}
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envBool(key string) bool {
	value, err := strconv.ParseBool(strings.TrimSpace(os.Getenv(key)))
	return err == nil && value
}

// probeReady verifies the local operational endpoint. HTTPS is mandatory when
// development insecure mode is disabled; platform roots or an explicit CA file
// are used without disabling certificate verification.
func probeReady(address string, developmentInsecure bool, caFile, serverName string) error {
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
		// nil means crypto/tls verifies against the platform trust store.
		return nil, nil
	}
	pem, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("read worker healthcheck TLS CA: %w", err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(pem) {
		return nil, errors.New("worker healthcheck TLS CA contains no certificates")
	}
	return roots, nil
}
