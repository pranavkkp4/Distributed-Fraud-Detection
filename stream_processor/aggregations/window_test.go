package aggregations

import (
	"context"
	"errors"
	"math"
	"strconv"
	"testing"
	"time"

	"github.com/pranavkkp4/distributed-fraud-detection/stream_processor/ingestion"
)

func TestSlidingWindowIdempotency(t *testing.T) {
	now := time.Now()
	w := New(time.Minute)
	e := ingestion.Event{ID: "1", EntityID: "a", Amount: 10, OccurredAt: now, Partition: 0, Offset: 1}
	f, changed, err := w.Apply(e)
	if err != nil || !changed || f.EventCount != 1 {
		t.Fatal(f, changed, err)
	}
	_, changed, err = w.Apply(e)
	if err != nil || changed {
		t.Fatal(changed, err)
	}
	f, changed, err = w.Apply(ingestion.Event{ID: "2", EntityID: "a", Amount: 30, OccurredAt: now.Add(time.Second), Partition: 0, Offset: 2})
	if err != nil || !changed || f.EventCount != 2 || f.TotalAmount != 40 || f.MaxAmount != 30 {
		t.Fatal(f, changed, err)
	}
	sink := NewMemoryMaterializer()
	if err := sink.Put(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if got, ok := sink.Get("a"); !ok || got.EventCount != 2 {
		t.Fatal(got, ok)
	}
}

func TestLateEventPreservesWatermark(t *testing.T) {
	window := New(time.Minute)
	base := time.Date(2026, 1, 1, 0, 10, 0, 0, time.UTC)
	first := ingestion.Event{ID: "new", EntityID: "account", Amount: 10, OccurredAt: base, Partition: 0, Offset: 1}
	if _, _, err := window.Apply(first); err != nil {
		t.Fatal(err)
	}
	late := ingestion.Event{ID: "late", EntityID: "account", Amount: 5, OccurredAt: base.Add(-30 * time.Second), Partition: 0, Offset: 2}
	features, changed, err := window.Apply(late)
	if err != nil || !changed {
		t.Fatal(features, changed, err)
	}
	if !features.WindowEnd.Equal(base) {
		t.Fatalf("watermark regressed to %s, want %s", features.WindowEnd, base)
	}
	if features.EventCount != 2 || features.TotalAmount != 15 {
		t.Fatalf("unexpected late-event features: %#v", features)
	}
}

func TestDedupePurgedWithWindow(t *testing.T) {
	window := New(time.Minute)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := range 50 {
		event := ingestion.Event{ID: "event-" + strconv.Itoa(i), EntityID: "account", Amount: 1, OccurredAt: base.Add(time.Duration(i) * time.Second), Partition: 0, Offset: int64(i + 1)}
		if _, _, err := window.Apply(event); err != nil {
			t.Fatal(err)
		}
	}
	if size := window.DedupeSize(); size != 50 {
		t.Fatalf("dedupe size = %d, want 50", size)
	}
	advance := ingestion.Event{ID: "advance", EntityID: "account", Amount: 1, OccurredAt: base.Add(3 * time.Minute), Partition: 0, Offset: 51}
	if _, _, err := window.Apply(advance); err != nil {
		t.Fatal(err)
	}
	if size := window.DedupeSize(); size != 1 {
		t.Fatalf("dedupe size after expiry = %d, want 1", size)
	}
}

func TestBoundedInactiveEntitiesAndEventCapacity(t *testing.T) {
	window := NewWithLimits(time.Hour, 2, 3)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i, entity := range []string{"a", "b", "c"} {
		if _, _, err := window.Apply(ingestion.Event{ID: "entity-" + entity, EntityID: entity, Amount: 1, OccurredAt: base.Add(time.Duration(i) * time.Second), Partition: i, Offset: 1}); err != nil {
			t.Fatal(err)
		}
	}
	if got := window.EntityCount(); got != 2 {
		t.Fatalf("entity count = %d, want bounded 2", got)
	}
	if _, _, err := window.Apply(ingestion.Event{ID: "capacity-1", EntityID: "c", Amount: 1, OccurredAt: base.Add(10 * time.Second), Partition: 10, Offset: 1}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := window.Apply(ingestion.Event{ID: "capacity-2", EntityID: "c", Amount: 1, OccurredAt: base.Add(11 * time.Second), Partition: 11, Offset: 1}); !errors.Is(err, ErrStateCapacity) {
		t.Fatalf("capacity error = %v", err)
	}
	if got := window.DedupeSize(); got > 3 {
		t.Fatalf("dedupe count = %d, exceeds configured event capacity", got)
	}
}

type failOnceSink struct{ failed bool }

func (s *failOnceSink) Put(context.Context, Features) error {
	if !s.failed {
		s.failed = true
		return errors.New("temporary sink failure")
	}
	return nil
}

func TestMaterializationFailureDoesNotAdvanceState(t *testing.T) {
	window := New(time.Minute)
	sink := &failOnceSink{}
	event := ingestion.Event{ID: "1", EntityID: "account", Amount: 10, OccurredAt: time.Now(), Partition: 0, Offset: 1}
	if _, _, err := window.ApplyAndMaterialize(context.Background(), event, sink); err == nil {
		t.Fatal("expected first materialization to fail")
	}
	features, changed, err := window.ApplyAndMaterialize(context.Background(), event, sink)
	if err != nil || !changed || features.EventCount != 1 {
		t.Fatalf("retry = %#v, %v, %v", features, changed, err)
	}
}

func TestMaterializationFailureDoesNotEvictInactiveEntity(t *testing.T) {
	window := NewWithLimits(time.Minute, 1, 10)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if _, _, err := window.Apply(ingestion.Event{ID: "a-1", EntityID: "a", Amount: 10, OccurredAt: base, Partition: 0, Offset: 1}); err != nil {
		t.Fatal(err)
	}
	sink := &failOnceSink{}
	_, _, err := window.ApplyAndMaterialize(
		context.Background(),
		ingestion.Event{ID: "b-1", EntityID: "b", Amount: 20, OccurredAt: base.Add(time.Second), Partition: 1, Offset: 1},
		sink,
	)
	if err == nil {
		t.Fatal("expected materialization failure")
	}
	if features, ok := window.Snapshot("a", base); !ok || features.TotalAmount != 10 {
		t.Fatalf("failed sink evicted committed entity: %#v, %v", features, ok)
	}
	if got := window.EntityCount(); got != 1 {
		t.Fatalf("entity count changed after failed sink: %d", got)
	}
}

func TestAggregateOverflowFailsClosed(t *testing.T) {
	window := New(time.Minute)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	first := ingestion.Event{ID: "first", EntityID: "account", Amount: math.MaxFloat64, OccurredAt: base, Partition: 0, Offset: 1}
	if _, _, err := window.Apply(first); err != nil {
		t.Fatal(err)
	}

	second := ingestion.Event{ID: "second", EntityID: "account", Amount: math.MaxFloat64, OccurredAt: base.Add(time.Second), Partition: 0, Offset: 2}
	if _, _, err := window.Apply(second); !errors.Is(err, ErrNumericRange) {
		t.Fatalf("overflow error = %v, want %v", err, ErrNumericRange)
	}

	features, ok := window.Snapshot("account", base.Add(time.Second))
	if !ok || features.EventCount != 1 || features.TotalAmount != math.MaxFloat64 {
		t.Fatalf("overflow changed committed state: %#v, %v", features, ok)
	}
	if got := window.DedupeSize(); got != 1 {
		t.Fatalf("overflow changed deduplication state: %d", got)
	}
}

func TestPartitionAndOffsetBoundsFailClosed(t *testing.T) {
	window := New(time.Minute)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	tests := []ingestion.Event{
		{ID: "negative-partition", EntityID: "account", Amount: 1, OccurredAt: base, Partition: -1, Offset: 0},
		{ID: "partition-cap", EntityID: "account", Amount: 1, OccurredAt: base, Partition: MaxTrackedPartitions, Offset: 0},
		{ID: "negative-offset", EntityID: "account", Amount: 1, OccurredAt: base, Partition: 0, Offset: -1},
	}
	for _, event := range tests {
		if _, _, err := window.Apply(event); err == nil {
			t.Fatalf("event %q unexpectedly accepted", event.ID)
		}
	}
	if got := window.EntityCount(); got != 0 {
		t.Fatalf("invalid events changed entity state: %d", got)
	}
}
