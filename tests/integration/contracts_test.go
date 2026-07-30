// Package integration validates public cross-plane contracts without Docker, Redis, or Kafka.
package integration

import (
	"context"
	"errors"
	"testing"
	"time"

	featurestore "github.com/pranavkkp4/distributed-fraud-detection/serving_plane/feature_store"
	"github.com/pranavkkp4/distributed-fraud-detection/stream_processor/aggregations"
	"github.com/pranavkkp4/distributed-fraud-detection/stream_processor/ingestion"
)

func TestFeatureStoreFixedWidthAndVersionedKeyContract(t *testing.T) {
	store := featurestore.NewMemory(3)
	const versionedEntityKey = "features:v1:account-42"

	if err := store.Put(versionedEntityKey, []float64{1.25, 0, 99}); err != nil {
		t.Fatalf("put fixed-width versioned key: %v", err)
	}
	if err := store.Put("features:v2:account-42", []float64{1, 2}); err == nil {
		t.Fatal("short feature vector was accepted")
	}

	features, err := store.Get(context.Background(), versionedEntityKey)
	if err != nil {
		t.Fatalf("get fixed-width versioned key: %v", err)
	}
	if len(features) != 3 || features[0] != 1.25 || features[2] != 99 {
		t.Fatalf("unexpected fixed-width feature vector: %#v", features)
	}
	features[0] = -1 // Callers must not mutate a stored version's snapshot.
	again, err := store.Get(context.Background(), versionedEntityKey)
	if err != nil || again[0] != 1.25 {
		t.Fatalf("store returned an aliased or missing snapshot: %#v, %v", again, err)
	}
	if _, err := store.Get(context.Background(), "features:v3:account-42"); !errors.Is(err, featurestore.ErrNotFound) {
		t.Fatalf("unknown feature version must not resolve to another version: %v", err)
	}
}

func TestAggregationOrderingAndIdempotencyContract(t *testing.T) {
	base := time.Date(2025, time.January, 1, 12, 0, 0, 0, time.UTC)
	window := aggregations.New(5 * time.Minute)

	first := ingestion.Event{ID: "event-1", EntityID: "account-42", Amount: 10, OccurredAt: base, Partition: 0, Offset: 11}
	features, changed, err := window.Apply(first)
	if err != nil || !changed || features.EventCount != 1 || features.TotalAmount != 10 {
		t.Fatalf("first ordered event: features=%+v changed=%v err=%v", features, changed, err)
	}
	if _, changed, err = window.Apply(first); err != nil || changed {
		t.Fatalf("duplicate event must be idempotent: changed=%v err=%v", changed, err)
	}

	stale := ingestion.Event{ID: "event-stale", EntityID: "account-42", Amount: 900, OccurredAt: base.Add(time.Second), Partition: 0, Offset: 10}
	if _, changed, err = window.Apply(stale); err != nil || changed {
		t.Fatalf("stale same-partition offset must not change state: changed=%v err=%v", changed, err)
	}

	next := ingestion.Event{ID: "event-2", EntityID: "account-42", Amount: 30, OccurredAt: base.Add(2 * time.Second), Partition: 0, Offset: 12}
	features, changed, err = window.Apply(next)
	if err != nil || !changed || features.EventCount != 2 || features.TotalAmount != 40 || features.MaxAmount != 30 {
		t.Fatalf("next ordered event: features=%+v changed=%v err=%v", features, changed, err)
	}

	otherPartition := ingestion.Event{ID: "event-3", EntityID: "account-42", Amount: 5, OccurredAt: base.Add(3 * time.Second), Partition: 1, Offset: 1}
	features, changed, err = window.Apply(otherPartition)
	if err != nil || !changed || features.EventCount != 3 || features.TotalAmount != 45 {
		t.Fatalf("partition offsets must be independent: features=%+v changed=%v err=%v", features, changed, err)
	}
}
