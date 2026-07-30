// Package aggregations calculates ordered, idempotent sliding-window fraud features.
package aggregations

import (
	"context"
	"errors"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/pranavkkp4/distributed-fraud-detection/stream_processor/ingestion"
)

type Features struct {
	EntityID    string    `json:"entity_id"`
	EventCount  int       `json:"event_count"`
	TotalAmount float64   `json:"total_amount"`
	MaxAmount   float64   `json:"max_amount"`
	WindowEnd   time.Time `json:"window_end"`
}

type Materializer interface {
	Put(context.Context, Features) error
}

var (
	ErrNumericRange  = errors.New("aggregation numeric range exceeded")
	ErrStateCapacity = errors.New("aggregation state capacity exceeded")
)

const MaxTrackedPartitions = 100_000

type event struct {
	id     string
	at     time.Time
	amount float64
}

type Window struct {
	Duration    time.Duration
	mu          sync.Mutex
	byEntity    map[string][]event
	seen        map[string]struct{}
	offsets     map[int]int64
	watermark   map[string]time.Time
	lastActive  map[string]time.Time
	eventCount  int
	maxEntities int
	maxEvents   int
}

func New(duration time.Duration) *Window {
	return NewWithLimits(duration, 100_000, 1_000_000)
}

// NewWithLimits makes state retention explicit. maxEvents bounds both active
// window records and their deduplication IDs; overflow fails closed instead of
// silently dropping records and corrupting a financial aggregate.
func NewWithLimits(duration time.Duration, maxEntities, maxEvents int) *Window {
	if duration <= 0 {
		duration = 5 * time.Minute
	}
	if maxEntities < 1 {
		maxEntities = 1
	}
	if maxEvents < 1 {
		maxEvents = 1
	}
	return &Window{
		Duration:    duration,
		byEntity:    map[string][]event{},
		seen:        map[string]struct{}{},
		offsets:     map[int]int64{},
		watermark:   map[string]time.Time{},
		lastActive:  map[string]time.Time{},
		maxEntities: maxEntities,
		maxEvents:   maxEvents,
	}
}

// Apply advances in-memory state. Duplicate IDs and stale partition offsets are
// no-ops. A per-entity event-time watermark prevents late records from moving a
// feature window backward.
func (w *Window) Apply(in ingestion.Event) (Features, bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	pending, err := w.prepareLocked(in)
	if err != nil || !pending.accepted {
		return Features{}, false, err
	}
	w.commitLocked(in, pending)
	return pending.features, pending.changed, nil
}

// ApplyAndMaterialize advances aggregation state only after the idempotent sink
// succeeds. A Kafka redelivery after a sink failure therefore recomputes and
// retries the same vector instead of accidentally committing a no-op.
func (w *Window) ApplyAndMaterialize(ctx context.Context, in ingestion.Event, sink Materializer) (Features, bool, error) {
	if sink == nil {
		return Features{}, false, errors.New("materializer is required")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	pending, err := w.prepareLocked(in)
	if err != nil || !pending.accepted {
		return Features{}, false, err
	}
	if pending.changed {
		if err := sink.Put(ctx, pending.features); err != nil {
			return Features{}, false, err
		}
	}
	w.commitLocked(in, pending)
	return pending.features, pending.changed, nil
}

type pendingUpdate struct {
	accepted    bool
	changed     bool
	features    Features
	events      []event
	watermark   time.Time
	expired     []string
	evictEntity string
}

func (w *Window) prepareLocked(in ingestion.Event) (pendingUpdate, error) {
	if !validEvent(in) {
		return pendingUpdate{}, errors.New("invalid financial event")
	}
	if _, duplicate := w.seen[in.ID]; duplicate {
		return pendingUpdate{}, nil
	}
	if high, ok := w.offsets[in.Partition]; ok && in.Offset <= high {
		return pendingUpdate{}, nil
	}
	evictEntity := ""
	if _, known := w.watermark[in.EntityID]; !known && len(w.watermark) >= w.maxEntities {
		evictEntity = w.oldestEntityLocked()
	}

	watermark := w.watermark[in.EntityID]
	if in.OccurredAt.After(watermark) {
		watermark = in.OccurredAt
	}
	cutoff := watermark.Add(-w.Duration)
	current := w.byEntity[in.EntityID]
	first := sort.Search(len(current), func(i int) bool { return !current[i].at.Before(cutoff) })
	expired := make([]string, first)
	for i := range first {
		expired[i] = current[i].id
	}
	events := append([]event(nil), current[first:]...)
	changed := !in.OccurredAt.Before(cutoff)
	if changed {
		events = append(events, event{id: in.ID, at: in.OccurredAt, amount: in.Amount})
		sort.SliceStable(events, func(i, j int) bool { return events[i].at.Before(events[j].at) })
	}
	projectedEventCount := w.eventCount - first
	if evictEntity != "" {
		projectedEventCount -= len(w.byEntity[evictEntity])
	}
	if changed {
		projectedEventCount++
	}
	if projectedEventCount > w.maxEvents {
		return pendingUpdate{}, ErrStateCapacity
	}

	features := Features{EntityID: in.EntityID, EventCount: len(events), WindowEnd: watermark}
	for _, item := range events {
		if features.TotalAmount > math.MaxFloat64-item.amount {
			return pendingUpdate{}, ErrNumericRange
		}
		features.TotalAmount += item.amount
		if item.amount > features.MaxAmount {
			features.MaxAmount = item.amount
		}
	}
	return pendingUpdate{
		accepted:    true,
		changed:     changed || len(expired) > 0,
		features:    features,
		events:      events,
		watermark:   watermark,
		expired:     expired,
		evictEntity: evictEntity,
	}, nil
}

func (w *Window) commitLocked(in ingestion.Event, pending pendingUpdate) {
	if pending.evictEntity != "" {
		w.evictEntityLocked(pending.evictEntity)
	}
	for _, id := range pending.expired {
		delete(w.seen, id)
	}
	if !in.OccurredAt.Before(pending.watermark.Add(-w.Duration)) {
		w.seen[in.ID] = struct{}{}
	}
	w.offsets[in.Partition] = in.Offset
	w.watermark[in.EntityID] = pending.watermark
	w.eventCount += len(pending.events) - len(w.byEntity[in.EntityID])
	w.byEntity[in.EntityID] = pending.events
	w.lastActive[in.EntityID] = pending.watermark
}

func (w *Window) oldestEntityLocked() string {
	var entity string
	var oldest time.Time
	for candidate, active := range w.lastActive {
		if entity == "" || active.Before(oldest) || (active.Equal(oldest) && candidate < entity) {
			entity, oldest = candidate, active
		}
	}
	return entity
}

func (w *Window) evictEntityLocked(entity string) {
	for _, item := range w.byEntity[entity] {
		delete(w.seen, item.id)
	}
	w.eventCount -= len(w.byEntity[entity])
	delete(w.byEntity, entity)
	delete(w.watermark, entity)
	delete(w.lastActive, entity)
}

func validEvent(in ingestion.Event) bool {
	return in.ID != "" && len(in.ID) <= 256 && utf8.ValidString(in.ID) && strings.IndexFunc(in.ID, unicode.IsControl) < 0 &&
		in.EntityID != "" && len(in.EntityID) <= 256 &&
		utf8.ValidString(in.EntityID) && strings.TrimSpace(in.EntityID) == in.EntityID && strings.IndexFunc(in.EntityID, unicode.IsControl) < 0 &&
		in.Amount >= 0 && !math.IsNaN(in.Amount) && !math.IsInf(in.Amount, 0) && !in.OccurredAt.IsZero() &&
		in.Partition >= 0 && in.Partition < MaxTrackedPartitions && in.Offset >= 0
}

// Snapshot returns a safe feature copy at the later of now or the entity's
// watermark. It does not mutate stream state.
func (w *Window) Snapshot(entity string, now time.Time) (Features, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	watermark := w.watermark[entity]
	if now.After(watermark) {
		watermark = now
	}
	events := w.byEntity[entity]
	if len(events) == 0 {
		return Features{}, false
	}
	cutoff := watermark.Add(-w.Duration)
	features := Features{EntityID: entity, WindowEnd: watermark}
	for _, item := range events {
		if item.at.Before(cutoff) {
			continue
		}
		features.EventCount++
		features.TotalAmount += item.amount
		if item.amount > features.MaxAmount {
			features.MaxAmount = item.amount
		}
	}
	return features, true
}

// DedupeSize exposes the active-window ID cardinality for monitoring/tests.
func (w *Window) DedupeSize() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.seen)
}

// EntityCount exposes bounded active entity state for monitoring/tests.
func (w *Window) EntityCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.watermark)
}
