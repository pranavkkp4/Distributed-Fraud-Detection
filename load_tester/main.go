// Command fraud-load-tester generates bounded, reproducible, Poisson-arrival
// HTTP traffic and writes measurements, rather than claimed benchmark results.
package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/HdrHistogram/hdrhistogram-go"
)

type Config struct {
	URL           string        `json:"url"`
	Token         string        `json:"-"`
	Duration      time.Duration `json:"duration"`
	Requests      int           `json:"requests"`
	Concurrency   int           `json:"concurrency"`
	Rate          float64       `json:"rate_per_second"`
	Timeout       time.Duration `json:"timeout"`
	Seed          uint64        `json:"seed"`
	Entities      []string      `json:"entities"`
	EntityWeights []int         `json:"entity_weights"`
	PayloadMode   string        `json:"payload_mode"`
	OutputDir     string        `json:"output_dir"`
}

func (c Config) Validate() error {
	if !strings.HasPrefix(c.URL, "http://") && !strings.HasPrefix(c.URL, "https://") {
		return errors.New("url must be http(s)")
	}
	if c.Requests < 1 && c.Duration <= 0 {
		return errors.New("set a positive -requests or -duration")
	}
	if c.Requests < 0 || c.Duration < 0 || c.Concurrency < 1 || c.Rate <= 0 || c.Timeout <= 0 {
		return errors.New("requests/duration/concurrency/rate/timeout must be positive")
	}
	if len(c.Entities) == 0 {
		return errors.New("at least one entity is required")
	}
	if c.PayloadMode != "" && c.PayloadMode != "entity" && c.PayloadMode != "transaction" {
		return errors.New("payload-mode must be entity or transaction")
	}
	if len(c.EntityWeights) != 0 && len(c.EntityWeights) != len(c.Entities) {
		return errors.New("entity weights must match entities")
	}
	for _, weight := range c.EntityWeights {
		if weight < 1 {
			return errors.New("entity weights must be positive")
		}
	}
	return nil
}

// ArrivalSchedule returns cumulative exponential inter-arrival offsets.  A
// seeded schedule is intentionally deterministic for repeatable investigations.
func ArrivalSchedule(n int, rate float64, seed uint64) ([]time.Duration, error) {
	if n < 0 || rate <= 0 {
		return nil, errors.New("n must be non-negative and rate must be positive")
	}
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	offsets := make([]time.Duration, n)
	var elapsed float64
	for i := range offsets {
		u := r.Float64()
		// Float64 is [0,1), and max avoids an extremely unusual log(0).
		if u <= 0 {
			u = math.SmallestNonzeroFloat64
		}
		elapsed += -math.Log(u) / rate
		offsets[i] = time.Duration(elapsed * float64(time.Second))
	}
	return offsets, nil
}

type Result struct {
	latency time.Duration
	status  int
	err     error
}
type job struct {
	index     int
	scheduled time.Time
}
type Summary struct {
	StartedAt      string           `json:"started_at"`
	FinishedAt     string           `json:"finished_at"`
	ElapsedSeconds float64          `json:"elapsed_seconds"`
	Scheduled      int              `json:"scheduled"`
	Completed      int              `json:"completed"`
	Successful     int              `json:"successful"`
	Errors         int              `json:"errors"`
	Throughput     float64          `json:"throughput_per_second"`
	StatusCounts   map[string]int   `json:"status_counts"`
	Percentiles    map[string]int64 `json:"percentiles_us"`
}

func Run(ctx context.Context, cfg Config) (*Summary, *hdrhistogram.Histogram, error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	requested := cfg.Requests
	if requested == 0 {
		requested = int(math.Ceil(cfg.Duration.Seconds() * cfg.Rate))
	}
	offsets, err := ArrivalSchedule(requested, cfg.Rate, cfg.Seed)
	if err != nil {
		return nil, nil, err
	}
	transport := &http.Transport{MaxIdleConns: cfg.Concurrency, MaxIdleConnsPerHost: cfg.Concurrency, MaxConnsPerHost: cfg.Concurrency, IdleConnTimeout: 30 * time.Second, TLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS13}}
	client := &http.Client{Transport: transport, Timeout: cfg.Timeout}
	defer transport.CloseIdleConnections()
	// The client deadline bounds ordinary observations; retain a one-hour range
	// so a pathological completed request is still represented rather than
	// silently dropped by the histogram.
	hist := hdrhistogram.New(1, int64(time.Hour/time.Microsecond), 3)
	results := make(chan Result, cfg.Concurrency)
	jobs := make(chan job)
	var workers sync.WaitGroup
	for range cfg.Concurrency {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for work := range jobs {
				results <- perform(ctx, client, cfg, work)
			}
		}()
	}
	started := time.Now()
	var sent atomic.Int64
	go func() {
		defer close(jobs)
		for i, offset := range offsets {
			if cfg.Duration > 0 && offset > cfg.Duration {
				break
			}
			timer := time.NewTimer(time.Until(started.Add(offset)))
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
			jobs <- job{index: i, scheduled: started.Add(offset)}
			sent.Add(1)
		}
	}()
	go func() { workers.Wait(); close(results) }()
	s := &Summary{StartedAt: started.UTC().Format(time.RFC3339Nano), StatusCounts: map[string]int{}, Percentiles: map[string]int64{}}
	for r := range results {
		s.Completed++
		if r.err != nil {
			s.Errors++
			s.StatusCounts["transport_error"]++
		} else {
			s.StatusCounts[strconv.Itoa(r.status)]++
			if r.status >= 200 && r.status < 300 {
				s.Successful++
			} else {
				s.Errors++
			}
		}
		// The value starts at planned open-loop arrival, so it includes client
		// admission/worker-queue delay. It is an observed request, not a synthetic
		// coordinated-omission correction.
		_ = hist.RecordValue(max64(1, r.latency.Microseconds()))
	}
	s.Scheduled = int(sent.Load())
	finished := time.Now()
	s.FinishedAt = finished.UTC().Format(time.RFC3339Nano)
	s.ElapsedSeconds = finished.Sub(started).Seconds()
	if s.ElapsedSeconds > 0 {
		s.Throughput = float64(s.Completed) / s.ElapsedSeconds
	}
	for _, p := range []struct {
		name  string
		value float64
	}{{"p50", 50}, {"p90", 90}, {"p99", 99}, {"p99_9", 99.9}, {"max", 100}} {
		s.Percentiles[p.name] = hist.ValueAtQuantile(p.value)
	}
	return s, hist, nil
}

func perform(ctx context.Context, client *http.Client, cfg Config, work job) Result {
	body, err := requestPayload(cfg, work)
	if err != nil {
		return Result{latency: time.Since(work.scheduled), err: err}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.URL, bytes.NewReader(body))
	if err != nil {
		return Result{latency: time.Since(work.scheduled), err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.Token)
	}
	resp, err := client.Do(req)
	latency := time.Since(work.scheduled)
	if err != nil {
		return Result{latency: latency, err: err}
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	return Result{latency: latency, status: resp.StatusCode}
}

// requestPayload only allocates the JSON byte slice required for HTTP. The
// transaction identifier derives from seed and index, making it deterministic
// and unique within a run without UUID or clock allocation.
func requestPayload(cfg Config, work job) ([]byte, error) {
	entity := entityForIndex(cfg, work.index)
	if cfg.PayloadMode == "transaction" {
		occurred := work.scheduled.UnixNano()
		if occurred <= 0 {
			occurred = 1_700_000_000_000_000_000 + int64(work.index)
		}
		payload := struct {
			TransactionID    string `json:"transaction_id"`
			AccountID        string `json:"account_id"`
			AmountMicros     int64  `json:"amount_micros"`
			Currency         string `json:"currency"`
			OccurredAtNS     int64  `json:"occurred_at_ns"`
			MerchantCategory string `json:"merchant_category"`
		}{
			TransactionID: fmt.Sprintf("load-%016x-%08x", cfg.Seed, work.index),
			AccountID:     entity, AmountMicros: 1_000_000 + int64(work.index%100)*10_000,
			Currency: "USD", OccurredAtNS: occurred,
			MerchantCategory: [...]string{"grocery", "fuel", "travel", "retail"}[work.index%4],
		}
		return json.Marshal(payload)
	}
	payload := struct {
		EntityID string  `json:"entity_id"`
		Amount   float64 `json:"amount"`
	}{EntityID: entity, Amount: 10 + float64(work.index%100)}
	return json.Marshal(payload)
}
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func WriteArtifacts(dir string, cfg Config, s *Summary, h *hdrhistogram.Histogram) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	manifest := struct {
		Config      Config             `json:"config"`
		Summary     *Summary           `json:"summary"`
		Histogram   []hdrhistogram.Bar `json:"histogram"`
		Environment struct {
			GOOS      string `json:"goos"`
			GOARCH    string `json:"goarch"`
			GoVersion string `json:"go_version"`
			CPUCount  int    `json:"cpu_count"`
		} `json:"environment"`
	}{Config: cfg, Summary: s, Histogram: h.Distribution()}
	manifest.Environment.GOOS, manifest.Environment.GOARCH = runtime.GOOS, runtime.GOARCH
	manifest.Environment.GoVersion, manifest.Environment.CPUCount = runtime.Version(), runtime.NumCPU()
	b, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0644); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(dir, "distribution.csv"))
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	_ = w.Write([]string{"quantile", "latency_us"})
	for _, p := range []float64{0, 25, 50, 75, 90, 95, 99, 99.9, 100} {
		_ = w.Write([]string{strconv.FormatFloat(p, 'f', 1, 64), strconv.FormatInt(h.ValueAtQuantile(p), 10)})
	}
	w.Flush()
	if err = w.Error(); err != nil {
		_ = f.Close()
		return err
	}
	_ = f.Close()
	return os.WriteFile(filepath.Join(dir, "latency-cdf.svg"), CDFSVG(h), 0644)
}
func CDFSVG(h *hdrhistogram.Histogram) []byte {
	pts := make([]string, 0, 101)
	max := h.Max()
	if max < 1 {
		max = 1
	}
	for i := 0; i <= 100; i++ {
		x := 40 + float64(h.ValueAtQuantile(float64(i)))/float64(max)*540
		y := 260 - float64(i)/100*220
		pts = append(pts, fmt.Sprintf("%.1f,%.1f", x, y))
	}
	return []byte(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300" viewBox="0 0 600 300"><rect width="100%%" height="100%%" fill="white"/><text x="40" y="24" font-family="sans-serif" font-size="16">Observed end-to-end latency CDF</text><path d="M40 40V260H580" fill="none" stroke="black"/><polyline points="%s" fill="none" stroke="#1565c0" stroke-width="2"/><text x="40" y="282" font-family="sans-serif" font-size="11">latency (μs), max %d</text><text x="4" y="50" font-family="sans-serif" font-size="11">100%%</text><text x="8" y="260" font-family="sans-serif" font-size="11">0%%</text></svg>`, strings.Join(pts, " "), h.Max()))
}

func main() {
	c := Config{}
	entities := flag.String("entities", "demo-account", "comma-separated entity IDs, optionally name:weight")
	flag.StringVar(&c.URL, "url", "http://127.0.0.1:8080/v1/score", "gateway POST URL")
	flag.StringVar(&c.Token, "token", "", "bearer token")
	flag.DurationVar(&c.Duration, "duration", 0, "run duration (alternative to requests)")
	flag.IntVar(&c.Requests, "requests", 1000, "number of requests (0 uses duration)")
	flag.IntVar(&c.Concurrency, "concurrency", 100, "maximum in-flight requests")
	flag.Float64Var(&c.Rate, "rate", 1000, "Poisson mean arrivals/second")
	flag.DurationVar(&c.Timeout, "timeout", 5*time.Second, "per-request timeout")
	flag.Uint64Var(&c.Seed, "seed", 1, "deterministic Poisson seed")
	flag.StringVar(&c.OutputDir, "output", "load_tester/results", "artifact directory")
	flag.StringVar(&c.PayloadMode, "payload-mode", "entity", "entity (gateway) or transaction (Rust Transaction API) body")
	flag.Parse()
	var parseErr error
	c.Entities, c.EntityWeights, parseErr = parseEntities(*entities)
	if parseErr != nil {
		fmt.Fprintln(os.Stderr, "load test:", parseErr)
		os.Exit(2)
	}
	s, h, err := Run(context.Background(), c)
	if err == nil {
		err = WriteArtifacts(c.OutputDir, c, s, h)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "load test:", err)
		os.Exit(2)
	}
	b, _ := json.MarshalIndent(s, "", "  ")
	fmt.Println(string(b))
}

// parseEntities accepts name[:weight] items, e.g. "vip:8,regular:2".  The
// deterministic weighted round-robin selector makes the traffic mix explicit
// without a second random source that could obscure Poisson reproducibility.
func parseEntities(s string) ([]string, []int, error) {
	parts := strings.Split(s, ",")
	names, weights := make([]string, 0, len(parts)), make([]int, 0, len(parts))
	for _, part := range parts {
		fields := strings.Split(strings.TrimSpace(part), ":")
		if len(fields) > 2 || fields[0] == "" {
			return nil, nil, errors.New("entities must be name[:weight]")
		}
		weight := 1
		if len(fields) == 2 {
			var err error
			weight, err = strconv.Atoi(fields[1])
			if err != nil || weight < 1 {
				return nil, nil, errors.New("entity weight must be a positive integer")
			}
		}
		names, weights = append(names, fields[0]), append(weights, weight)
	}
	return names, weights, nil
}
func entityForIndex(c Config, index int) string {
	if len(c.EntityWeights) == 0 {
		return c.Entities[index%len(c.Entities)]
	}
	total := 0
	for _, weight := range c.EntityWeights {
		total += weight
	}
	position := index % total
	for i, weight := range c.EntityWeights {
		if position < weight {
			return c.Entities[i]
		}
		position -= weight
	}
	return c.Entities[len(c.Entities)-1]
}
