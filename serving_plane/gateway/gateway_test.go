package gateway

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/feature_store"
	"github.com/pranavkkp4/distributed-fraud-detection/serving_plane/worker"
)

func TestAuthScoreAndResultContract(t *testing.T) {
	memory := featurestore.NewMemory(2)
	_ = memory.Put("e", []float64{1, 2})
	service, err := New(memory, worker.CPUScorer{Weights: []float64{1, 1}}, Config{AuthToken: "x", Timeout: time.Second, QueueDelay: 20 * time.Millisecond}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	handler := service.Handler()

	request := httptest.NewRequest(http.MethodPost, "/v1/score", bytes.NewBufferString(`{"entity_id":"e","amount":1}`))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusUnauthorized {
		t.Fatal(recorder.Code)
	}

	request = httptest.NewRequest(http.MethodPost, "/v1/score", bytes.NewBufferString(`{"entity_id":"e","amount":1}`))
	request.Header.Set("Authorization", "Bearer x")
	request.Header.Set("X-Request-ID", "known")
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || recorder.Header().Get("X-Request-ID") != "known" {
		t.Fatalf("%d %s", recorder.Code, recorder.Body.String())
	}
	var response scoreResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Confidence < 0 || response.Confidence > 1 || len(response.Explanation) > worker.MaxExplanations {
		t.Fatalf("unbounded response: %#v", response)
	}
	if response.Model == "" || response.CalibrationVersion == "" || response.PolicyVersion == "" || response.FeatureVersion == "" {
		t.Fatalf("missing response versions: %#v", response)
	}
}

func TestStrictJSONAndEntityValidation(t *testing.T) {
	memory := featurestore.NewMemory(1)
	_ = memory.Put("e", []float64{1})
	service, err := New(memory, worker.CPUScorer{Weights: []float64{1}}, Config{Timeout: time.Second}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()

	for _, body := range []string{
		`{"entity_id":"e","amount":1} {}`,
		`{"entity_id":"` + strings.Repeat("a", 257) + `","amount":1}`,
		`{"entity_id":" e","amount":1}`,
	} {
		request := httptest.NewRequest(http.MethodPost, "/v1/score", strings.NewReader(body))
		recorder := httptest.NewRecorder()
		service.Handler().ServeHTTP(recorder, request)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("body %q returned %d", body, recorder.Code)
		}
	}
}

func TestInvalidRequestIDIsReplacedWithWorkerSafeRandomID(t *testing.T) {
	service, err := New(featurestore.NewMemory(1), worker.CPUScorer{Weights: []float64{1}}, Config{Timeout: time.Second}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	request.Header.Set("X-Request-ID", "not worker safe")
	recorder := httptest.NewRecorder()
	service.Handler().ServeHTTP(recorder, request)
	requestID := recorder.Header().Get("X-Request-ID")
	if requestID == "not worker safe" || len(requestID) != 32 {
		t.Fatalf("replacement request ID = %q", requestID)
	}
	if _, err := hex.DecodeString(requestID); err != nil {
		t.Fatalf("replacement request ID is not random hex: %v", err)
	}
}

type blockingStore struct {
	remaining chan time.Duration
}

func (s blockingStore) Get(ctx context.Context, _ string) ([]float64, error) {
	if deadline, ok := ctx.Deadline(); ok {
		s.remaining <- time.Until(deadline)
	}
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestFeaturePhaseTimeout(t *testing.T) {
	remaining := make(chan time.Duration, 1)
	service, err := New(blockingStore{remaining: remaining}, worker.CPUScorer{Weights: []float64{1}}, Config{
		Timeout:          200 * time.Millisecond,
		FeatureTimeout:   15 * time.Millisecond,
		QueueDelay:       5 * time.Millisecond,
		InferenceTimeout: 50 * time.Millisecond,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	request := httptest.NewRequest(http.MethodPost, "/v1/score", strings.NewReader(`{"entity_id":"e","amount":1}`))
	recorder := httptest.NewRecorder()
	service.Handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d", recorder.Code)
	}
	var response scoreResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if !response.Fallback || response.FallbackReason != worker.FallbackFeatureUnavailable {
		t.Fatalf("fallback contract = %#v", response)
	}
	if budget := <-remaining; budget <= 0 || budget > 25*time.Millisecond {
		t.Fatalf("feature phase budget = %s", budget)
	}
}

type budgetEngine struct {
	remaining chan time.Duration
}

func (e budgetEngine) Infer(ctx context.Context, requests []worker.Request) ([]worker.Result, error) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return nil, errors.New("missing inference deadline")
	}
	e.remaining <- time.Until(deadline)
	results := make([]worker.Result, len(requests))
	for i := range results {
		results[i] = worker.Result{Score: 0.1, Confidence: 0.8, Model: "test", CalibrationVersion: "c1", PolicyVersion: "p1", FeatureVersion: "f1", Explanation: []worker.Explanation{}}
	}
	return results, nil
}

func (e budgetEngine) Healthy(context.Context) error { return nil }

func TestQueueDelayDoesNotConsumeInferenceBudget(t *testing.T) {
	memory := featurestore.NewMemory(1)
	_ = memory.Put("e", []float64{1})
	remaining := make(chan time.Duration, 1)
	service, err := New(memory, budgetEngine{remaining: remaining}, Config{
		Timeout:          200 * time.Millisecond,
		FeatureTimeout:   20 * time.Millisecond,
		QueueDelay:       20 * time.Millisecond,
		InferenceTimeout: 50 * time.Millisecond,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	request := httptest.NewRequest(http.MethodPost, "/v1/score", strings.NewReader(`{"entity_id":"e","amount":1}`))
	recorder := httptest.NewRecorder()
	service.Handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", recorder.Code, recorder.Body.String())
	}
	if budget := <-remaining; budget < 35*time.Millisecond || budget > 60*time.Millisecond {
		t.Fatalf("inference phase budget = %s", budget)
	}
}
