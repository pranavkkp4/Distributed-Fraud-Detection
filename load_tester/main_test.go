package main

import (
	"context"
	"encoding/json"
	"github.com/HdrHistogram/hdrhistogram-go"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestArrivalScheduleDeterministicAndIncreasing(t *testing.T) {
	a, e := ArrivalSchedule(20, 100, 7)
	if e != nil {
		t.Fatal(e)
	}
	b, _ := ArrivalSchedule(20, 100, 7)
	for i := range a {
		if a[i] != b[i] || (i > 0 && a[i] <= a[i-1]) {
			t.Fatalf("non deterministic/increasing at %d", i)
		}
	}
	if _, e = ArrivalSchedule(1, 0, 1); e == nil {
		t.Fatal("expected validation")
	}
}
func TestConfigValidation(t *testing.T) {
	c := Config{URL: "http://x", Requests: 1, Concurrency: 1, Rate: 1, Timeout: time.Second, Entities: []string{"e"}}
	if c.Validate() != nil {
		t.Fatal("valid config rejected")
	}
	c.Rate = 0
	if c.Validate() == nil {
		t.Fatal("invalid rate accepted")
	}
	c.Rate, c.PayloadMode = 1, "unknown"
	if c.Validate() == nil {
		t.Fatal("invalid payload mode accepted")
	}
}
func TestPayloadModes(t *testing.T) {
	base := Config{Seed: 7, Entities: []string{"account-a"}}
	work := job{index: 3, scheduled: time.Unix(1_700_000_000, 42)}
	entityBytes, err := requestPayload(base, work)
	if err != nil {
		t.Fatal(err)
	}
	var entity map[string]any
	if err = json.Unmarshal(entityBytes, &entity); err != nil || len(entity) != 2 || entity["entity_id"] != "account-a" {
		t.Fatalf("bad entity payload: %s (%v)", entityBytes, err)
	}
	base.PayloadMode = "transaction"
	transactionBytes, err := requestPayload(base, work)
	if err != nil {
		t.Fatal(err)
	}
	var transaction map[string]any
	if err = json.Unmarshal(transactionBytes, &transaction); err != nil || len(transaction) != 6 {
		t.Fatalf("bad transaction payload: %s (%v)", transactionBytes, err)
	}
	if transaction["transaction_id"] != "load-0000000000000007-00000003" || transaction["account_id"] != "account-a" || transaction["amount_micros"] == nil || transaction["currency"] != "USD" || transaction["occurred_at_ns"] == nil || transaction["merchant_category"] == nil {
		t.Fatalf("missing transaction fields: %#v", transaction)
	}
}
func TestWeightedEntities(t *testing.T) {
	names, weights, err := parseEntities("vip:2,regular:1")
	if err != nil {
		t.Fatal(err)
	}
	c := Config{Entities: names, EntityWeights: weights}
	got := []string{entityForIndex(c, 0), entityForIndex(c, 1), entityForIndex(c, 2)}
	if strings.Join(got, ",") != "vip,vip,regular" {
		t.Fatalf("unexpected distribution: %v", got)
	}
	if _, _, err := parseEntities("vip:0"); err == nil {
		t.Fatal("zero weight accepted")
	}
}
func TestArtifactsAndSVG(t *testing.T) {
	h := hdrhistogram.New(1, 1000000, 3)
	for _, v := range []int64{10, 20, 30, 40, 100} {
		_ = h.RecordValue(v)
	}
	s := &Summary{Percentiles: map[string]int64{"p99": h.ValueAtQuantile(99)}}
	d := t.TempDir()
	c := Config{URL: "http://x", Requests: 1, Concurrency: 1, Rate: 1, Timeout: time.Second, Entities: []string{"e"}}
	if e := WriteArtifacts(d, c, s, h); e != nil {
		t.Fatal(e)
	}
	svg := string(CDFSVG(h))
	if !strings.Contains(svg, "Observed end-to-end latency CDF") || !strings.Contains(svg, "latency (") || !strings.Contains(svg, "polyline") {
		t.Fatal("invalid svg")
	}
	var v map[string]any
	b, e := os.ReadFile(filepath.Join(d, "manifest.json"))
	if e != nil {
		t.Fatal(e)
	}
	if json.Unmarshal(b, &v) != nil || v["summary"] == nil {
		t.Fatal("invalid manifest")
	}
	if v["environment"] == nil {
		t.Fatal("manifest missing environment")
	}
}

func TestScheduledLatencyIncludesSaturationQueue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil || len(payload) != 2 || payload["entity_id"] == nil || payload["amount"] == nil {
			t.Errorf("gateway-incompatible payload: %#v, %v", payload, err)
		}
		time.Sleep(20 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	c := Config{URL: srv.URL, Requests: 3, Concurrency: 1, Rate: 100000, Timeout: time.Second, Seed: 2, Entities: []string{"e"}}
	s, h, err := Run(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	if s.Completed != 3 || s.Successful != 3 {
		t.Fatalf("unexpected outcome: %#v", s)
	}
	// Request 3 waits behind two 20 ms responses; latency begins at its planned
	// arrival, therefore max must include that client-side queue delay.
	if h.Max() < int64((35*time.Millisecond)/time.Microsecond) {
		t.Fatalf("queue delay was excluded: max=%dus", h.Max())
	}
}
