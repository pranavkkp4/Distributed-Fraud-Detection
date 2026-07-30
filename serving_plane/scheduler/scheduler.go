// Package scheduler provides bounded, deadline-aware continuous batching.
package scheduler

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrClosed     = errors.New("scheduler is closed")
	ErrFull       = errors.New("scheduler queue is full")
	ErrQueueDelay = errors.New("scheduler queue delay exceeded")
)

// Result returns either the batch result or an error.
type Result[R any] struct {
	Value R
	Err   error
}

// Handler is called serially with bounded batches. It must return one result per item.
type Handler[T any, R any] func(context.Context, []T) ([]R, error)

const (
	stateQueued uint32 = iota
	stateDispatched
	stateExpired
)

// Scheduler limits queueing time and queue size before invoking Handler.
type Scheduler[T any, R any] struct {
	queue        chan *item[T, R]
	done         chan struct{}
	once         sync.Once
	mu           sync.RWMutex
	closed       bool
	maxBatch     int
	maxDelay     time.Duration
	handler      Handler[T, R]
	queueCount   atomic.Uint64
	queueNanos   atomic.Uint64
	queueBuckets [5]atomic.Uint64
	batchCount   atomic.Uint64
	batchItems   atomic.Uint64
	batchBuckets [6]atomic.Uint64
}

type item[T any, R any] struct {
	ctx         context.Context
	value       T
	reply       chan Result[R]
	dispatched  chan struct{}
	queueExpiry time.Time
	state       atomic.Uint32
}

// New starts a continuous batcher. maxDelay applies only while an item is
// waiting to enter a handler invocation. Callers enforce their own deadlines
// while waiting; a batch handler receives the latest finite deadline shared by
// the batch so one short request cannot cancel otherwise viable work.
func New[T any, R any](maxQueue, maxBatch int, maxDelay time.Duration, handler Handler[T, R]) (*Scheduler[T, R], error) {
	if maxQueue < 1 || maxBatch < 1 || maxDelay <= 0 || handler == nil {
		return nil, errors.New("invalid scheduler configuration")
	}
	s := &Scheduler[T, R]{
		queue:    make(chan *item[T, R], maxQueue),
		done:     make(chan struct{}),
		maxBatch: maxBatch,
		maxDelay: maxDelay,
		handler:  handler,
	}
	go s.run()
	return s, nil
}

// Submit waits for its value's result while honoring ctx before and during
// inference. It returns ErrQueueDelay only if the item was not admitted to a
// batch within maxDelay.
func (s *Scheduler[T, R]) Submit(ctx context.Context, value T) (R, error) {
	var zero R
	if err := ctx.Err(); err != nil {
		return zero, err
	}

	it := &item[T, R]{
		ctx:         ctx,
		value:       value,
		reply:       make(chan Result[R], 1),
		dispatched:  make(chan struct{}),
		queueExpiry: time.Now().Add(s.maxDelay),
	}

	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return zero, ErrClosed
	}
	select {
	case s.queue <- it:
		s.mu.RUnlock()
	default:
		s.mu.RUnlock()
		return zero, ErrFull
	}

	timer := time.NewTimer(time.Until(it.queueExpiry))
	defer stopTimer(timer)
	for {
		select {
		case result := <-it.reply:
			return result.Value, result.Err
		case <-it.dispatched:
			return waitForResult(ctx, it.reply)
		case <-timer.C:
			if it.state.CompareAndSwap(stateQueued, stateExpired) {
				return zero, ErrQueueDelay
			}
			return waitForResult(ctx, it.reply)
		case <-ctx.Done():
			it.state.CompareAndSwap(stateQueued, stateExpired)
			return zero, ctx.Err()
		case <-s.done:
			select {
			case result := <-it.reply:
				return result.Value, result.Err
			default:
				return zero, ErrClosed
			}
		}
	}
}

func waitForResult[R any](ctx context.Context, reply <-chan Result[R]) (R, error) {
	var zero R
	select {
	case result := <-reply:
		return result.Value, result.Err
	case <-ctx.Done():
		return zero, ctx.Err()
	}
}

func stopTimer(timer *time.Timer) {
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
}

func (s *Scheduler[T, R]) run() {
	defer close(s.done)
	for {
		first, ok := s.nextQueued()
		if !ok {
			return
		}

		batch := []*item[T, R]{first}
		deadline := first.queueExpiry
		timer := time.NewTimer(s.collectionWait(deadline))
		queueOpen := true
	collect:
		for len(batch) < s.maxBatch && queueOpen {
			select {
			case next, open := <-s.queue:
				if !open {
					queueOpen = false
					break collect
				}
				if next.state.Load() != stateQueued {
					continue
				}
				batch = append(batch, next)
				if next.queueExpiry.Before(deadline) {
					deadline = next.queueExpiry
					stopTimer(timer)
					timer.Reset(s.collectionWait(deadline))
				}
			case <-timer.C:
				break collect
			}
		}
		stopTimer(timer)
		s.dispatch(batch)
		if !queueOpen {
			return
		}
	}
}

func (s *Scheduler[T, R]) nextQueued() (*item[T, R], bool) {
	for {
		it, ok := <-s.queue
		if !ok {
			return nil, false
		}
		if it.state.Load() == stateQueued {
			return it, true
		}
	}
}

// Wake slightly before the absolute cap so timer scheduling jitter cannot turn
// a healthy batch into a queue timeout. The expiry check in dispatch remains
// the authoritative cap, including when the scheduler is busy in a handler.
func (s *Scheduler[T, R]) collectionWait(deadline time.Time) time.Duration {
	wait := time.Until(deadline)
	// Reserve enough time for timer scheduling jitter (notably on Windows).
	// The strict expiry check in dispatch is still the authoritative cap.
	guard := s.maxDelay / 2
	if guard < 50*time.Microsecond {
		guard = 50 * time.Microsecond
	}
	if wait > guard {
		wait -= guard
	}
	if wait < 0 {
		return 0
	}
	return wait
}

func (s *Scheduler[T, R]) dispatch(batch []*item[T, R]) {
	now := time.Now()
	active := make([]*item[T, R], 0, len(batch))
	values := make([]T, 0, len(batch))
	for _, it := range batch {
		if err := it.ctx.Err(); err != nil {
			if it.state.CompareAndSwap(stateQueued, stateExpired) {
				it.reply <- Result[R]{Err: err}
			}
			continue
		}
		if !now.Before(it.queueExpiry) {
			if it.state.CompareAndSwap(stateQueued, stateExpired) {
				it.reply <- Result[R]{Err: ErrQueueDelay}
			}
			continue
		}
		if !it.state.CompareAndSwap(stateQueued, stateDispatched) {
			continue
		}
		close(it.dispatched)
		s.observeQueue(now.Sub(it.queueExpiry.Add(-s.maxDelay)))
		active = append(active, it)
		values = append(values, it.value)
	}
	if len(active) == 0 {
		return
	}
	s.observeBatch(len(active))

	// Callers enforce their own deadline while waiting for the reply. The latest
	// finite deadline bounds a fully deadline-aware batch without letting one
	// nearly-expired request poison every other request. If any caller is
	// unbounded, the handler remains responsible for its own execution bound.
	batchCtx := context.WithoutCancel(active[0].ctx)
	var latestDeadline time.Time
	hasUnboundedCaller := false
	for _, it := range active {
		deadline, ok := it.ctx.Deadline()
		if !ok {
			hasUnboundedCaller = true
			continue
		}
		if deadline.After(latestDeadline) {
			latestDeadline = deadline
		}
	}
	var cancel context.CancelFunc
	if !hasUnboundedCaller && !latestDeadline.IsZero() {
		batchCtx, cancel = context.WithDeadline(batchCtx, latestDeadline)
		defer cancel()
	}

	results, err := s.handler(batchCtx, values)
	if err == nil && len(results) != len(active) {
		err = errors.New("batch handler returned incorrect result count")
	}
	for i, it := range active {
		if ctxErr := it.ctx.Err(); ctxErr != nil {
			it.reply <- Result[R]{Err: ctxErr}
		} else if err != nil {
			it.reply <- Result[R]{Err: err}
		} else {
			it.reply <- Result[R]{Value: results[i]}
		}
	}
}

var queueBounds = [...]time.Duration{100 * time.Microsecond, 500 * time.Microsecond, time.Millisecond, 2 * time.Millisecond}
var batchBounds = [...]int{1, 4, 8, 16, 32}

func (s *Scheduler[T, R]) observeQueue(duration time.Duration) {
	if duration < 0 {
		duration = 0
	}
	s.queueCount.Add(1)
	s.queueNanos.Add(uint64(duration))
	for i, bound := range queueBounds {
		if duration <= bound {
			s.queueBuckets[i].Add(1)
		}
	}
	s.queueBuckets[len(s.queueBuckets)-1].Add(1)
}

func (s *Scheduler[T, R]) observeBatch(size int) {
	s.batchCount.Add(1)
	s.batchItems.Add(uint64(size))
	for i, bound := range batchBounds {
		if size <= bound {
			s.batchBuckets[i].Add(1)
		}
	}
	s.batchBuckets[len(s.batchBuckets)-1].Add(1)
}

type Metrics struct {
	QueueCount   uint64
	QueueNanos   uint64
	QueueBuckets [5]uint64
	BatchCount   uint64
	BatchItems   uint64
	BatchBuckets [6]uint64
}

func (s *Scheduler[T, R]) Metrics() Metrics {
	metrics := Metrics{
		QueueCount: s.queueCount.Load(),
		QueueNanos: s.queueNanos.Load(),
		BatchCount: s.batchCount.Load(),
		BatchItems: s.batchItems.Load(),
	}
	for i := range metrics.QueueBuckets {
		metrics.QueueBuckets[i] = s.queueBuckets[i].Load()
	}
	for i := range metrics.BatchBuckets {
		metrics.BatchBuckets[i] = s.batchBuckets[i].Load()
	}
	return metrics
}

// Close stops accepting new work and drains work already admitted to the queue.
func (s *Scheduler[T, R]) Close() {
	s.once.Do(func() {
		s.mu.Lock()
		s.closed = true
		close(s.queue)
		s.mu.Unlock()
	})
}
