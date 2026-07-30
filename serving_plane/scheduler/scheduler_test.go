package scheduler

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestBatchesAndCancellation(t *testing.T) {
	calls := make(chan int, 1)
	s, err := New[int, int](4, 2, 40*time.Millisecond, func(_ context.Context, values []int) ([]int, error) {
		calls <- len(values)
		return values, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	done := make(chan error, 2)
	for _, n := range []int{1, 2} {
		go func(n int) {
			value, submitErr := s.Submit(context.Background(), n)
			if submitErr == nil && value != n {
				submitErr = errors.New("wrong value")
			}
			done <- submitErr
		}(n)
	}
	for range 2 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
	if size := <-calls; size != 2 {
		t.Fatalf("batch size = %d, want 2", size)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Submit(ctx, 3); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled submit error = %v", err)
	}
}

func TestQueueDelayDoesNotConsumeHandlerBudget(t *testing.T) {
	started := make(chan struct{})
	s, err := New[int, int](2, 1, 20*time.Millisecond, func(ctx context.Context, values []int) ([]int, error) {
		close(started)
		select {
		case <-time.After(45 * time.Millisecond):
			return values, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	value, err := s.Submit(ctx, 7)
	if err != nil || value != 7 {
		t.Fatalf("Submit() = %d, %v; handler must retain its caller-context budget", value, err)
	}
	<-started
}

func TestBusySchedulerExpiresQueuedItemOnly(t *testing.T) {
	firstStarted := make(chan struct{})
	release := make(chan struct{})
	s, err := New[int, int](4, 1, 20*time.Millisecond, func(_ context.Context, values []int) ([]int, error) {
		if values[0] == 1 {
			close(firstStarted)
			<-release
		}
		return values, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	firstDone := make(chan error, 1)
	go func() {
		_, submitErr := s.Submit(context.Background(), 1)
		firstDone <- submitErr
	}()
	<-firstStarted

	started := time.Now()
	if _, err := s.Submit(context.Background(), 2); !errors.Is(err, ErrQueueDelay) {
		t.Fatalf("queued submit error = %v, want %v", err, ErrQueueDelay)
	}
	if elapsed := time.Since(started); elapsed > 100*time.Millisecond {
		t.Fatalf("queue timeout took %s", elapsed)
	}
	close(release)
	if err := <-firstDone; err != nil {
		t.Fatalf("dispatched request failed: %v", err)
	}
}

func TestShortCallerDeadlineDoesNotPoisonBatch(t *testing.T) {
	handlerDeadline := make(chan time.Duration, 1)
	s, err := New[int, int](4, 2, 40*time.Millisecond, func(ctx context.Context, values []int) ([]int, error) {
		deadline, ok := ctx.Deadline()
		if !ok {
			return nil, errors.New("bounded callers must produce a bounded batch")
		}
		handlerDeadline <- time.Until(deadline)
		select {
		case <-time.After(100 * time.Millisecond):
			return values, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	shortCtx, shortCancel := context.WithTimeout(context.Background(), 70*time.Millisecond)
	defer shortCancel()
	longCtx, longCancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer longCancel()
	type outcome struct {
		value int
		err   error
	}
	shortResult, longResult := make(chan outcome, 1), make(chan outcome, 1)
	go func() {
		value, submitErr := s.Submit(shortCtx, 1)
		shortResult <- outcome{value, submitErr}
	}()
	time.Sleep(time.Millisecond)
	go func() {
		value, submitErr := s.Submit(longCtx, 2)
		longResult <- outcome{value, submitErr}
	}()

	if remaining := <-handlerDeadline; remaining < 250*time.Millisecond {
		t.Fatalf("handler inherited the short caller deadline: %s", remaining)
	}
	if result := <-shortResult; !errors.Is(result.err, context.DeadlineExceeded) {
		t.Fatalf("short caller = %#v", result)
	}
	if result := <-longResult; result.err != nil || result.value != 2 {
		t.Fatalf("long caller = %#v", result)
	}
}

func TestConcurrentSubmitAndClose(t *testing.T) {
	s, err := New[int, int](64, 8, 10*time.Millisecond, func(_ context.Context, values []int) ([]int, error) {
		return values, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for i := range 128 {
		wg.Add(1)
		go func(value int) {
			defer wg.Done()
			_, submitErr := s.Submit(context.Background(), value)
			if submitErr != nil && !errors.Is(submitErr, ErrClosed) && !errors.Is(submitErr, ErrFull) && !errors.Is(submitErr, ErrQueueDelay) {
				t.Errorf("unexpected submit error: %v", submitErr)
			}
		}(i)
	}
	s.Close()
	wg.Wait()
}
