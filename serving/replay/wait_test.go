package replay

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

type waitStep struct {
	wake, stopCost, beforeResetCost, selectCost time.Duration
	cancel                                      bool
}
type scriptedWaitClock struct {
	t         *testing.T
	base, now time.Time
	steps     []waitStep
	cancel    context.CancelFunc
	delays    []time.Duration
	created   int
	stops     int
	resets    int
	nowReads  int
	timer     *scriptedWaitTimer
}
type scriptedWaitTimer struct {
	clock   *scriptedWaitClock
	channel chan time.Time
	step    int
	stopped bool
}

func newScriptedWait(t *testing.T, steps ...waitStep) *scriptedWaitClock {
	t.Helper()
	base := time.Unix(1000, 0)
	return &scriptedWaitClock{t: t, base: base, now: base, steps: steps}
}
func (c *scriptedWaitClock) Now() time.Time {
	if c.timer != nil {
		c.nowReads++
		if c.nowReads == 3 {
			c.now = c.now.Add(c.steps[c.timer.step-1].beforeResetCost)
		}
	}
	return c.now
}
func (c *scriptedWaitClock) NewTimer(d time.Duration) waitTimer {
	c.created++
	if c.created != 1 {
		c.t.Fatal("created another timer instead of reusing it")
	}
	c.timer = &scriptedWaitTimer{clock: c, channel: make(chan time.Time, 1)}
	c.timer.arm(d)
	return c.timer
}
func (t *scriptedWaitTimer) C() <-chan time.Time {
	// Models progress after Reset/counting but before select observes its wake.
	t.clock.now = t.clock.now.Add(t.clock.steps[t.step-1].selectCost)
	return t.channel
}
func (t *scriptedWaitTimer) Stop() bool {
	t.clock.stops++
	t.stopped = true
	t.clock.now = t.clock.now.Add(t.clock.steps[t.step-1].stopCost)
	return false
}
func (t *scriptedWaitTimer) Reset(d time.Duration) bool {
	if !t.stopped {
		t.clock.t.Fatal("Reset without Stop")
	}
	t.clock.resets++
	t.arm(d)
	return false
}
func (t *scriptedWaitTimer) arm(d time.Duration) {
	c := t.clock
	if d <= 0 || t.step == len(c.steps) {
		c.t.Fatal("unexpected additional/nonblocking wait", d, t.step)
	}
	c.delays = append(c.delays, d)
	s := c.steps[t.step]
	t.step++
	t.stopped = false
	c.now = c.base.Add(s.wake)
	c.nowReads = 0
	if s.cancel {
		if c.cancel == nil {
			c.t.Fatal("missing cancellation callback")
		}
		c.cancel()
	} else {
		// Deliberately stale timer payload: only actual Now may decide wake time.
		t.channel <- c.base.Add(-time.Hour)
	}
}

func waitRecorder(t *testing.T, start time.Time) *Diagnostics {
	t.Helper()
	d, err := NewDiagnostics(1)
	if err != nil {
		t.Fatal(err)
	}
	if err = d.begin(start, 1); err != nil {
		t.Fatal(err)
	}
	return d
}
func finishWaitRecord(r *phaseRequest, at time.Time) {
	r.at("send_entry", at)
	r.at("get_conn", at)
	r.capture(phaseEvent{Phase: "got_conn", AtNS: at.Sub(r.parent.start).Nanoseconds(), Connection: 1, Reused: true})
	seq := r.writeEntry(1, 2, at)
	r.writeReturn(1, seq, 2, 2, nil, at)
}

func TestWaitSegmentsBoundAndAbsoluteDue(t *testing.T) {
	cases := []struct {
		name    string
		steps   []waitStep
		delays  []time.Duration
		interim uint32
	}{
		{"segments", []waitStep{{wake: 100 * time.Millisecond}, {wake: 200 * time.Millisecond}, {wake: 250 * time.Millisecond}}, []time.Duration{100 * time.Millisecond, 100 * time.Millisecond, 50 * time.Millisecond}, 2},
		{"early-wake", []waitStep{{wake: 80 * time.Millisecond}, {wake: 180 * time.Millisecond}, {wake: 249 * time.Millisecond}, {wake: 250 * time.Millisecond}}, []time.Duration{100 * time.Millisecond, 100 * time.Millisecond, 70 * time.Millisecond, time.Millisecond}, 3},
		{"delayed-wakes", []waitStep{{wake: 110 * time.Millisecond}, {wake: 225 * time.Millisecond}, {wake: 260 * time.Millisecond}}, []time.Duration{100 * time.Millisecond, 100 * time.Millisecond, 25 * time.Millisecond}, 2},
		{"late-first-wake", []waitStep{{wake: 273 * time.Millisecond}}, []time.Duration{100 * time.Millisecond}, 0},
		{"fresh-reset-remainder", []waitStep{{wake: 100 * time.Millisecond, beforeResetCost: 100 * time.Millisecond}, {wake: 250 * time.Millisecond}}, []time.Duration{100 * time.Millisecond, 50 * time.Millisecond}, 1},
		{"bookkeeping-crosses-due", []waitStep{{wake: 100 * time.Millisecond, beforeResetCost: 200 * time.Millisecond}}, []time.Duration{100 * time.Millisecond}, 0},
		{"after-reset-counter-delay", []waitStep{{wake: 100 * time.Millisecond}, {wake: 200 * time.Millisecond, selectCost: 80 * time.Millisecond}}, []time.Duration{100 * time.Millisecond, 100 * time.Millisecond}, 1},
		{"stop-crosses-due", []waitStep{{wake: 249 * time.Millisecond, stopCost: 2 * time.Millisecond}}, []time.Duration{100 * time.Millisecond}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clock := newScriptedWait(t, tc.steps...)
			d := waitRecorder(t, clock.base)
			due := clock.base.Add(250 * time.Millisecond)
			waitUntilClock(context.Background(), due, d.request(0), clock)
			if !reflect.DeepEqual(clock.delays, tc.delays) || clock.created != 1 || clock.stops != len(tc.steps) || clock.resets != len(tc.steps)-1 || clock.now.Before(due) {
				t.Fatalf("delays=%v creates=%d stops=%d resets=%d now=%v", clock.delays, clock.created, clock.stops, clock.resets, clock.now)
			}
			row := d.Snapshot().Requests[0]
			last := tc.steps[len(tc.steps)-1]
			if len(row.Events) != 2 || row.IntermediateWakes != tc.interim || row.Events[0].Phase != "post_select" || row.Events[0].AtNS != int64(last.wake+last.selectCost) || row.Events[1].AtNS != int64(last.wake+last.selectCost+last.stopCost) {
				t.Fatalf("final-only actual wake, not stale payload: %+v", row)
			}
			finishWaitRecord(d.request(0), clock.now)
			capture := d.Snapshot()
			if capture.Version != 2 || !capture.CaptureValid {
				t.Fatalf("%+v", capture)
			}
		})
	}
}

func TestWaitZeroAndOverdueCreateNoTimer(t *testing.T) {
	for _, delay := range []time.Duration{0, -time.Second} {
		t.Run(delay.String(), func(t *testing.T) {
			clock := newScriptedWait(t)
			d := waitRecorder(t, clock.base)
			waitUntilClock(context.Background(), clock.base.Add(delay), d.request(0), clock)
			row := d.Snapshot().Requests[0]
			if clock.created != 0 || len(row.Events) != 1 || row.Events[0].Phase != "no_timer" || row.IntermediateWakes != 0 {
				t.Fatal(clock.created, row)
			}
		})
	}
}

func TestWaitCancellationAmongSegments(t *testing.T) {
	for _, intermediate := range []bool{false, true} {
		t.Run(fmt.Sprint(intermediate), func(t *testing.T) {
			steps := []waitStep{{wake: 150 * time.Millisecond, cancel: true}}
			if intermediate {
				steps = append([]waitStep{{wake: 100 * time.Millisecond}}, steps...)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			clock := newScriptedWait(t, steps...)
			clock.cancel = cancel
			d := waitRecorder(t, clock.base)
			waitUntilClock(ctx, clock.base.Add(time.Second), d.request(0), clock)
			d.request(0).at("cancelled", clock.now)
			capture := d.Snapshot()
			if ctx.Err() == nil || clock.resets != len(steps)-1 || !capture.CaptureValid || capture.Requests[0].IntermediateWakes != uint32(len(steps)-1) || len(capture.Requests[0].Events) != 3 {
				t.Fatal(ctx.Err(), clock.resets, capture)
			}
		})
	}
}

func TestWaitOptionalCounterSaturationAndDisabled(t *testing.T) {
	steps := []waitStep{{wake: 100 * time.Millisecond}, {wake: 200 * time.Millisecond}, {wake: 250 * time.Millisecond}}
	clock := newScriptedWait(t, steps...)
	d := waitRecorder(t, clock.base)
	d.request(0).intermediateWakes = maxIntermediateWakes - 1
	waitUntilClock(context.Background(), clock.base.Add(250*time.Millisecond), d.request(0), clock)
	finishWaitRecord(d.request(0), clock.now)
	capture := d.Snapshot()
	if capture.CaptureValid || capture.Requests[0].IntermediateWakes != maxIntermediateWakes || !strings.Contains(strings.Join(capture.Requests[0].Problems, ","), "intermediate_wake_overflow") || len(capture.Requests[0].Events) != 7 {
		t.Fatalf("counter wrapped or filled event slots: %+v", capture)
	}
	plain := newScriptedWait(t, steps...)
	waitUntilClock(context.Background(), plain.base.Add(250*time.Millisecond), nil, plain)
	if !reflect.DeepEqual(plain.delays, clock.delays) || plain.stops != clock.stops {
		t.Fatal("ordinary and diagnostic waits differ")
	}
}

func TestWaitConcurrentCountersAndSnapshots(t *testing.T) {
	d := phaseRecorder(t, 1)
	var workers sync.WaitGroup
	for i := 0; i < 16; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for j := 0; j < 32; j++ {
				d.request(0).intermediateWake()
				_ = d.Snapshot()
			}
		}()
	}
	workers.Wait()
	at := time.Now()
	d.request(0).finalWake(at, at)
	finishWaitRecord(d.request(0), at)
	capture := d.Snapshot()
	if !capture.CaptureValid || capture.Requests[0].IntermediateWakes != 512 || len(capture.Requests[0].Events) != 7 {
		t.Fatal(capture)
	}
}

type waitRoundTripper func(*http.Request) (*http.Response, error)

func (f waitRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestWaitKeepsAbsoluteRequestDeadlineAndLegacySend(t *testing.T) {
	clock := newScriptedWait(t, waitStep{wake: 275 * time.Millisecond})
	due := time.Now().Add(time.Second)
	clock.base, clock.now = due.Add(-250*time.Millisecond), due.Add(-250*time.Millisecond)
	waitUntilClock(context.Background(), due, nil, clock)
	deadline := due.Add(2 * time.Second)
	called := false
	client := &http.Client{Transport: waitRoundTripper(func(req *http.Request) (*http.Response, error) {
		called = true
		actual, ok := req.Context().Deadline()
		if !ok || actual != deadline {
			t.Fatal("deadline rebased to late wake", actual, deadline)
		}
		return &http.Response{StatusCode: 503, Body: io.NopCloser(strings.NewReader("")), Header: http.Header{}}, nil
	})}
	r := emptyRecord(250)
	// Original eight-argument helper remains callable; no HTTP connection needed.
	send(context.Background(), client, "http://fixture.invalid", []byte("{}"), deadline, clock.base, &r, 1)
	if !called || r.Status != "http_error" || !clock.now.After(due) {
		t.Fatal(called, r, clock.now)
	}
}
