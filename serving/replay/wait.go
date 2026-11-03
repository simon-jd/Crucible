package replay

import (
	"context"
	"time"
)

// Experimental blocking segment bound; due and request deadline remain absolute.
const maxWaitSegment = 100 * time.Millisecond

type waitTimer interface {
	C() <-chan time.Time
	Stop() bool
	Reset(time.Duration) bool
}
type waitClock interface {
	Now() time.Time
	NewTimer(time.Duration) waitTimer
}
type realWaitClock struct{}
type realWaitTimer struct{ timer *time.Timer }

func (realWaitClock) Now() time.Time { return time.Now() }
func (realWaitClock) NewTimer(d time.Duration) waitTimer {
	return realWaitTimer{time.NewTimer(d)}
}
func (t realWaitTimer) C() <-chan time.Time        { return t.timer.C }
func (t realWaitTimer) Stop() bool                 { return t.timer.Stop() }
func (t realWaitTimer) Reset(d time.Duration) bool { return t.timer.Reset(d) }

// The two times precede recorder locking; never use the timer channel's payload.
func observeWaitStop(timer waitTimer, clock waitClock) (time.Time, time.Time) {
	selected := clock.Now()
	timer.Stop()
	return selected, clock.Now()
}

func waitUntil(ctx context.Context, due time.Time, phase *phaseRequest) {
	waitUntilClock(ctx, due, phase, realWaitClock{})
}

// One timer is reused for this request. Go1.27's synchronous timer channels
// guarantee that Stop/Reset do not leave a stale value requiring a drain.
func waitUntilClock(ctx context.Context, due time.Time, phase *phaseRequest, clock waitClock) {
	now := clock.Now()
	remaining := due.Sub(now)
	if remaining <= 0 {
		if phase != nil {
			phase.at("no_timer", now)
		}
		return
	}
	timer := clock.NewTimer(min(remaining, maxWaitSegment))
	for {
		select {
		case <-timer.C():
		case <-ctx.Done():
		}
		selected, stopped := observeWaitStop(timer, clock)
		remaining = due.Sub(stopped)
		if ctx.Err() != nil || remaining <= 0 {
			if phase != nil {
				phase.finalWake(selected, stopped)
			}
			return
		}
		// Refresh immediately before Reset, with no recorder lock between them.
		// If bookkeeping crossed due, this remains the last actual wake pair.
		remaining = due.Sub(clock.Now())
		if ctx.Err() != nil || remaining <= 0 {
			if phase != nil {
				phase.finalWake(selected, stopped)
			}
			return
		}
		timer.Reset(min(remaining, maxWaitSegment))
		// Count only after the next blocking segment is armed. Recorder delay
		// may delay observing its wake, but cannot rebase an already-armed timer.
		if phase != nil {
			phase.intermediateWake()
		}
	}
}
