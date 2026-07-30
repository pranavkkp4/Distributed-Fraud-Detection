// Package gateway serves authenticated fraud-scoring HTTP requests.
package gateway

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/feature_store"
	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/scheduler"
	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/worker"
)

type Config struct {
	AuthToken        string
	Timeout          time.Duration
	FeatureTimeout   time.Duration
	QueueDelay       time.Duration
	InferenceTimeout time.Duration
	DenyScore        float64
}
type Gateway struct {
	features         featurestore.Store
	engine           worker.Inference
	scheduler        *scheduler.Scheduler[worker.Request, worker.Result]
	config           Config
	logger           *slog.Logger
	requests         atomic.Uint64
	denied           atomic.Uint64
	failures         atomic.Uint64
	featureFallbacks atomic.Uint64
	queueFallbacks   atomic.Uint64
	workerFallbacks  atomic.Uint64
	requestNanos     atomic.Uint64
	requestBuckets   [7]atomic.Uint64
}
type scoreRequest struct {
	EntityID string  `json:"entity_id"`
	Amount   float64 `json:"amount"`
}
type scoreResponse struct {
	RequestID          string               `json:"request_id"`
	Score              float64              `json:"score"`
	Confidence         float64              `json:"confidence"`
	Decision           string               `json:"decision"`
	Explanation        []worker.Explanation `json:"explanation"`
	Model              string               `json:"model"`
	CalibrationVersion string               `json:"calibration_version"`
	PolicyVersion      string               `json:"policy_version"`
	FeatureVersion     string               `json:"feature_version"`
	Fallback           bool                 `json:"fallback"`
	FallbackReason     string               `json:"fallback_reason"`
}

func New(features featurestore.Store, engine worker.Inference, cfg Config, logger *slog.Logger) (*Gateway, error) {
	if features == nil || engine == nil {
		return nil, errors.New("features and inference engine are required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 40 * time.Millisecond
	}
	if cfg.FeatureTimeout <= 0 {
		cfg.FeatureTimeout = 10 * time.Millisecond
	}
	if cfg.QueueDelay <= 0 {
		cfg.QueueDelay = 2 * time.Millisecond
	}
	if cfg.InferenceTimeout <= 0 {
		cfg.InferenceTimeout = 20 * time.Millisecond
	}
	if cfg.DenyScore <= 0 {
		cfg.DenyScore = 0.5
	}
	if logger == nil {
		logger = slog.Default()
	}
	s, err := scheduler.New[worker.Request, worker.Result](256, 32, cfg.QueueDelay, func(ctx context.Context, reqs []worker.Request) ([]worker.Result, error) {
		inferenceCtx, cancel := context.WithTimeout(ctx, cfg.InferenceTimeout)
		defer cancel()
		return engine.Infer(inferenceCtx, reqs)
	})
	if err != nil {
		return nil, err
	}
	return &Gateway{features: features, engine: engine, scheduler: s, config: cfg, logger: logger}, nil
}
func (g *Gateway) Close() { g.scheduler.Close() }
func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/score", g.score)
	mux.HandleFunc("GET /healthz", g.health)
	mux.HandleFunc("GET /readyz", g.ready)
	mux.HandleFunc("GET /metrics", g.metrics)
	return g.requestID(mux)
}
func (g *Gateway) score(w http.ResponseWriter, r *http.Request) {
	if !g.authorized(r) {
		g.denied.Add(1)
		g.write(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
		return
	}
	g.requests.Add(1)
	started := time.Now()
	defer func() { g.observeRequest(time.Since(started)) }()
	defer r.Body.Close()
	var in scoreRequest
	d := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10))
	d.DisallowUnknownFields()
	if err := d.Decode(&in); err != nil || d.Decode(&struct{}{}) != io.EOF || !validEntityID(in.EntityID) || in.Amount < 0 {
		g.write(w, http.StatusBadRequest, map[string]string{"error": "invalid request"})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), g.config.Timeout)
	defer cancel()
	ctx = worker.WithRequestMetadata(ctx, requestID(r.Context()), r.Header.Get("Authorization"))
	featureCtx, featureCancel := context.WithTimeout(ctx, g.config.FeatureTimeout)
	features, err := g.features.Get(featureCtx, in.EntityID)
	featureCancel()
	if err != nil {
		g.failures.Add(1)
		g.respondFallback(w, r, worker.FallbackFeatureUnavailable)
		return
	}
	result, err := g.scheduler.Submit(ctx, worker.Request{Features: features, TransactionAmount: in.Amount})
	if err != nil {
		g.failures.Add(1)
		reason := worker.FallbackWorkerUnavailable
		if errors.Is(err, scheduler.ErrQueueDelay) || errors.Is(err, scheduler.ErrFull) {
			reason = worker.FallbackQueueDeadline
		}
		g.respondFallback(w, r, reason)
		return
	}
	decision := "allow"
	if result.Score >= g.config.DenyScore {
		decision = "deny"
	}
	g.write(w, http.StatusOK, scoreResponse{RequestID: requestID(r.Context()), Score: result.Score, Confidence: result.Confidence, Decision: decision, Explanation: result.Explanation, Model: result.Model, CalibrationVersion: result.CalibrationVersion, PolicyVersion: result.PolicyVersion, FeatureVersion: result.FeatureVersion, Fallback: result.Fallback, FallbackReason: result.FallbackReason})
}
func (g *Gateway) respondFallback(w http.ResponseWriter, r *http.Request, reason string) {
	g.logger.Warn("fail-closed fraud fallback", "request_id", requestID(r.Context()), "reason", reason)
	switch reason {
	case worker.FallbackFeatureUnavailable:
		g.featureFallbacks.Add(1)
	case worker.FallbackQueueDeadline:
		g.queueFallbacks.Add(1)
	default:
		g.workerFallbacks.Add(1)
	}
	g.write(w, http.StatusServiceUnavailable, scoreResponse{RequestID: requestID(r.Context()), Score: 1, Confidence: 1, Decision: "deny", Explanation: []worker.Explanation{}, Model: "fail-closed", CalibrationVersion: "unavailable", PolicyVersion: "fail-closed-v1", FeatureVersion: "rolling-features-v1-32", Fallback: true, FallbackReason: reason})
}
func (g *Gateway) health(w http.ResponseWriter, r *http.Request) {
	g.write(w, http.StatusOK, map[string]string{"status": "ok"})
}
func (g *Gateway) ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), g.config.Timeout)
	defer cancel()
	if store, ok := g.features.(featurestore.Prober); ok {
		if err := store.Probe(ctx); err != nil {
			g.write(w, http.StatusServiceUnavailable, map[string]string{"status": "feature store unavailable"})
			return
		}
	}
	if err := g.engine.Healthy(ctx); err != nil {
		g.write(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready"})
		return
	}
	g.write(w, http.StatusOK, map[string]string{"status": "ready"})
}
func (g *Gateway) metrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	var output strings.Builder
	fmt.Fprintf(&output, "fraud_gateway_requests_total %d\n", g.requests.Load())
	fmt.Fprintf(&output, "fraud_gateway_auth_denied_total %d\n", g.denied.Load())
	fmt.Fprintf(&output, "fraud_gateway_fallback_total{reason=%q} %d\n", worker.FallbackFeatureUnavailable, g.featureFallbacks.Load())
	fmt.Fprintf(&output, "fraud_gateway_fallback_total{reason=%q} %d\n", worker.FallbackQueueDeadline, g.queueFallbacks.Load())
	fmt.Fprintf(&output, "fraud_gateway_fallback_total{reason=%q} %d\n", worker.FallbackWorkerUnavailable, g.workerFallbacks.Load())
	fmt.Fprintf(&output, "fraud_gateway_fail_closed_total %d\n", g.failures.Load())
	requestLabels := [...]string{"0.0001", "0.0005", "0.001", "0.005", "0.01", "0.04", "+Inf"}
	for i, label := range requestLabels {
		fmt.Fprintf(&output, "fraud_gateway_request_duration_seconds_bucket{le=%q} %d\n", label, g.requestBuckets[i].Load())
	}
	fmt.Fprintf(&output, "fraud_gateway_request_duration_seconds_sum %g\n", float64(g.requestNanos.Load())/float64(time.Second))
	fmt.Fprintf(&output, "fraud_gateway_request_duration_seconds_count %d\n", g.requests.Load())
	schedulerMetrics := g.scheduler.Metrics()
	queueLabels := [...]string{"0.0001", "0.0005", "0.001", "0.002", "+Inf"}
	for i, label := range queueLabels {
		fmt.Fprintf(&output, "fraud_scheduler_queue_duration_seconds_bucket{le=%q} %d\n", label, schedulerMetrics.QueueBuckets[i])
	}
	fmt.Fprintf(&output, "fraud_scheduler_queue_duration_seconds_sum %g\n", float64(schedulerMetrics.QueueNanos)/float64(time.Second))
	fmt.Fprintf(&output, "fraud_scheduler_queue_duration_seconds_count %d\n", schedulerMetrics.QueueCount)
	batchLabels := [...]string{"1", "4", "8", "16", "32", "+Inf"}
	for i, label := range batchLabels {
		fmt.Fprintf(&output, "fraud_scheduler_batch_size_bucket{le=%q} %d\n", label, schedulerMetrics.BatchBuckets[i])
	}
	fmt.Fprintf(&output, "fraud_scheduler_batch_size_sum %d\n", schedulerMetrics.BatchItems)
	fmt.Fprintf(&output, "fraud_scheduler_batch_size_count %d\n", schedulerMetrics.BatchCount)
	_, _ = w.Write([]byte(output.String()))
}

var requestDurationBounds = [...]time.Duration{100 * time.Microsecond, 500 * time.Microsecond, time.Millisecond, 5 * time.Millisecond, 10 * time.Millisecond, 40 * time.Millisecond}

func (g *Gateway) observeRequest(duration time.Duration) {
	g.requestNanos.Add(uint64(duration))
	for i, bound := range requestDurationBounds {
		if duration <= bound {
			g.requestBuckets[i].Add(1)
		}
	}
	g.requestBuckets[len(g.requestBuckets)-1].Add(1)
}
func (g *Gateway) write(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func (g *Gateway) authorized(r *http.Request) bool {
	if g.config.AuthToken == "" {
		return true
	}
	const p = "Bearer "
	v := r.Header.Get("Authorization")
	return strings.HasPrefix(v, p) && subtle.ConstantTimeCompare([]byte(strings.TrimPrefix(v, p)), []byte(g.config.AuthToken)) == 1
}
func (g *Gateway) requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if !validRequestID(id) {
			id = newRequestID()
		}
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), requestIDKey{}, id)))
	})
}

var requestIDSequence atomic.Uint64

func newRequestID() string {
	var value [16]byte
	if _, err := rand.Read(value[:]); err == nil {
		return hex.EncodeToString(value[:])
	}
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), requestIDSequence.Add(1))
}

// gRPC text metadata is ASCII. Restrict caller-supplied IDs to a printable,
// whitespace-free subset so audit identity cannot break the worker hop.
func validRequestID(value string) bool {
	if len(value) == 0 || len(value) > 128 {
		return false
	}
	for i := range len(value) {
		if value[i] < 0x21 || value[i] > 0x7e {
			return false
		}
	}
	return true
}

func validEntityID(value string) bool {
	return value != "" && len(value) <= 256 && utf8.ValidString(value) && strings.TrimSpace(value) == value && strings.IndexFunc(value, unicode.IsControl) < 0
}

type requestIDKey struct{}

func requestID(ctx context.Context) string { v, _ := ctx.Value(requestIDKey{}).(string); return v }
