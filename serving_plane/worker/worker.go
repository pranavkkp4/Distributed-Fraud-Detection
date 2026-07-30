// Package worker exposes inference worker and health-pool abstractions.
package worker

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

type Request struct {
	Features          []float64 `json:"features"`
	TransactionAmount float64   `json:"transaction_amount"`
}
type Result struct {
	Score              float64       `json:"score"`
	Confidence         float64       `json:"confidence"`
	Explanation        []Explanation `json:"explanation"`
	Model              string        `json:"model"`
	CalibrationVersion string        `json:"calibration_version"`
	PolicyVersion      string        `json:"policy_version"`
	FeatureVersion     string        `json:"feature_version"`
	Fallback           bool          `json:"fallback"`
	FallbackReason     string        `json:"fallback_reason"`
}

// Explanation contains only a bounded feature index and clipped contribution;
// raw feature values are never returned to callers.
type Explanation struct {
	FeatureIndex int     `json:"feature_index"`
	Impact       float64 `json:"impact"`
}
type Inference interface {
	Infer(context.Context, []Request) ([]Result, error)
	Healthy(context.Context) error
}

const (
	ModelFeatureWidth          = 32
	MaxBatchSize               = 64
	MaxFeatureWidth            = 256
	MaxRPCMessageBytes         = 512 << 10
	MaxExplanations            = 3
	FallbackFeatureUnavailable = "feature_unavailable"
	FallbackQueueDeadline      = "queue_deadline"
	FallbackWorkerUnavailable  = "worker_unavailable"
)

func validateRequests(requests []Request, allowEmpty bool) error {
	if (!allowEmpty && len(requests) == 0) || len(requests) > MaxBatchSize {
		return fmt.Errorf("batch size must be between 1 and %d", MaxBatchSize)
	}
	for i, request := range requests {
		if len(request.Features) == 0 || len(request.Features) > MaxFeatureWidth {
			return fmt.Errorf("request %d feature width must be between 1 and %d", i, MaxFeatureWidth)
		}
		if request.TransactionAmount < 0 || math.IsNaN(request.TransactionAmount) || math.IsInf(request.TransactionAmount, 0) {
			return fmt.Errorf("request %d has invalid transaction amount", i)
		}
		for _, feature := range request.Features {
			if math.IsNaN(feature) || math.IsInf(feature, 0) {
				return fmt.Errorf("request %d contains a non-finite feature", i)
			}
		}
	}
	return nil
}

func validateResults(results []Result, expected int) error {
	if len(results) != expected {
		return errors.New("inference result count mismatch")
	}
	for i, result := range results {
		if result.Score < 0 || result.Score > 1 || math.IsNaN(result.Score) || math.IsInf(result.Score, 0) {
			return fmt.Errorf("result %d has invalid score", i)
		}
		if result.Confidence < 0 || result.Confidence > 1 || math.IsNaN(result.Confidence) || math.IsInf(result.Confidence, 0) {
			return fmt.Errorf("result %d has invalid confidence", i)
		}
		if len(result.Explanation) > MaxExplanations {
			return fmt.Errorf("result %d has too many explanations", i)
		}
		for _, detail := range result.Explanation {
			if detail.FeatureIndex < 0 || detail.FeatureIndex >= MaxFeatureWidth || detail.Impact < -1 || detail.Impact > 1 || math.IsNaN(detail.Impact) || math.IsInf(detail.Impact, 0) {
				return fmt.Errorf("result %d has invalid explanation", i)
			}
		}
		if !validVersion(result.Model) || !validVersion(result.CalibrationVersion) || !validVersion(result.PolicyVersion) || !validVersion(result.FeatureVersion) {
			return fmt.Errorf("result %d has invalid version metadata", i)
		}
		if result.Fallback != (result.FallbackReason != "") || !validFallbackReason(result.FallbackReason) {
			return fmt.Errorf("result %d has invalid fallback reason", i)
		}
	}
	return nil
}

func validVersion(value string) bool {
	return value != "" && len(value) <= 64 && utf8.ValidString(value) && strings.IndexFunc(value, unicode.IsControl) < 0
}

func validFallbackReason(value string) bool {
	return value == "" || value == FallbackFeatureUnavailable || value == FallbackQueueDeadline || value == FallbackWorkerUnavailable
}

// CPUScorer is deterministic and intentionally small for local and fail-closed operation.
type CPUScorer struct {
	Weights            []float64
	Bias               float64
	ModelName          string
	CalibrationVersion string
	PolicyVersion      string
	FeatureVersion     string
}

// DefaultCPUScorer matches the online model contract's 32-value feature width.
func DefaultCPUScorer() CPUScorer {
	weights := make([]float64, ModelFeatureWidth)
	copy(weights, []float64{1.2, 0.8, 1.1})
	return CPUScorer{Weights: weights, Bias: -1}
}

func (s CPUScorer) Infer(ctx context.Context, requests []Request) ([]Result, error) {
	if len(s.Weights) == 0 {
		return nil, errors.New("scorer has no weights")
	}
	if err := validateRequests(requests, true); err != nil {
		return nil, err
	}
	out := make([]Result, len(requests))
	for i, r := range requests {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(r.Features) != len(s.Weights) {
			return nil, errors.New("feature width mismatch")
		}
		z := s.Bias
		contributions := make([]Explanation, len(r.Features))
		for j, x := range r.Features {
			contribution := x * s.Weights[j]
			z += contribution
			contributions[j] = Explanation{FeatureIndex: j, Impact: contribution}
		}
		z += 0.0001 * r.TransactionAmount
		score := 1 / (1 + math.Exp(-z))
		out[i] = Result{
			Score:              score,
			Confidence:         math.Abs(score-0.5) * 2,
			Explanation:        boundedExplanations(contributions),
			Model:              s.name(),
			CalibrationVersion: s.calibrationVersion(),
			PolicyVersion:      s.policyVersion(),
			FeatureVersion:     s.featureVersion(),
		}
	}
	return out, nil
}
func (s CPUScorer) Healthy(context.Context) error {
	if len(s.Weights) == 0 {
		return errors.New("no weights")
	}
	return nil
}
func (s CPUScorer) name() string {
	if s.ModelName != "" {
		return s.ModelName
	}
	return "deterministic-cpu-v1"
}

func (s CPUScorer) calibrationVersion() string {
	if s.CalibrationVersion != "" {
		return s.CalibrationVersion
	}
	return "sigmoid-v1"
}

func (s CPUScorer) policyVersion() string {
	if s.PolicyVersion != "" {
		return s.PolicyVersion
	}
	return "fraud-threshold-v1"
}

func (s CPUScorer) featureVersion() string {
	if s.FeatureVersion != "" {
		return s.FeatureVersion
	}
	return "rolling-features-v1-32"
}

func boundedExplanations(values []Explanation) []Explanation {
	sort.Slice(values, func(i, j int) bool {
		left, right := math.Abs(values[i].Impact), math.Abs(values[j].Impact)
		if left == right {
			return values[i].FeatureIndex < values[j].FeatureIndex
		}
		return left > right
	})
	if len(values) > MaxExplanations {
		values = values[:MaxExplanations]
	}
	for i := range values {
		values[i].Impact = math.Max(-1, math.Min(1, values[i].Impact))
	}
	return values
}

// Pool selects healthy workers round-robin and avoids a down worker until its probe interval elapses.
type Pool struct {
	workers    []Inference
	probeEvery time.Duration
	mu         sync.Mutex
	next       uint64
	unhealthy  map[int]time.Time
}

func NewPool(workers []Inference, probeEvery time.Duration) (*Pool, error) {
	if len(workers) == 0 {
		return nil, errors.New("empty worker pool")
	}
	if probeEvery <= 0 {
		probeEvery = time.Second
	}
	return &Pool{workers: append([]Inference(nil), workers...), probeEvery: probeEvery, unhealthy: map[int]time.Time{}}, nil
}
func (p *Pool) Infer(ctx context.Context, requests []Request) ([]Result, error) {
	p.mu.Lock()
	start := int(p.next % uint64(len(p.workers)))
	p.next++
	p.mu.Unlock()
	var last error
	for offset := 0; offset < len(p.workers); offset++ {
		i := (start + offset) % len(p.workers)
		attemptCtx, cancel := p.attemptContext(ctx, len(p.workers)-offset)
		if !p.available(attemptCtx, i) {
			cancel()
			continue
		}
		r, err := p.workers[i].Infer(attemptCtx, requests)
		cancel()
		if err == nil {
			return r, nil
		}
		last = err
		if ctx.Err() == nil {
			p.markDown(i)
		}
	}
	if last == nil {
		last = errors.New("no healthy inference worker")
	}
	return nil, last
}

// attemptContext prevents a single stalled endpoint from consuming the entire
// request budget before the pool can fail over. It never extends the caller's
// deadline and divides the remaining time among viable attempts.
func (p *Pool) attemptContext(ctx context.Context, remaining int) (context.Context, context.CancelFunc) {
	if remaining < 1 {
		remaining = 1
	}
	if deadline, ok := ctx.Deadline(); ok {
		budget := time.Until(deadline) / time.Duration(remaining)
		if budget <= 0 {
			return context.WithCancel(ctx)
		}
		return context.WithTimeout(ctx, budget)
	}
	return context.WithTimeout(ctx, 100*time.Millisecond)
}
func (p *Pool) Healthy(ctx context.Context) error {
	for i, candidate := range p.workers {
		attemptCtx, cancel := p.attemptContext(ctx, len(p.workers)-i)
		err := candidate.Healthy(attemptCtx)
		cancel()
		if err == nil {
			p.mu.Lock()
			delete(p.unhealthy, i)
			p.mu.Unlock()
			return nil
		}
		p.markDown(i)
	}
	return errors.New("no healthy inference worker")
}
func (p *Pool) available(ctx context.Context, i int) bool {
	p.mu.Lock()
	until, down := p.unhealthy[i]
	p.mu.Unlock()
	if !down {
		return true
	}
	if time.Now().Before(until) {
		return false
	}
	if err := p.workers[i].Healthy(ctx); err != nil {
		p.markDown(i)
		return false
	}
	p.mu.Lock()
	delete(p.unhealthy, i)
	p.mu.Unlock()
	return true
}
func (p *Pool) markDown(i int) {
	p.mu.Lock()
	p.unhealthy[i] = time.Now().Add(p.probeEvery)
	p.mu.Unlock()
}
