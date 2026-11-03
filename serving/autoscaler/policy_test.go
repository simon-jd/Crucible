package main

import (
	"testing"
	"time"
)

func testPolicy(t *testing.T) (*Policy, *time.Time) {
	t.Helper()
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	p, err := NewPolicy(Config{Min: 1, Max: 8, QueueLow: 1, QueueHigh: 4, TTFTLow: 0.2, TTFTHigh: 0.8, FreshFor: 10 * time.Second}, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	return p, &now
}
func metric(now time.Time, q, ttft float64) Metrics {
	return Metrics{ObservedAt: now, QueuePerReplica: q, P95TTFTSeconds: ttft}
}
func TestAutoscalerHysteresis(t *testing.T) {
	p, now := testPolicy(t)
	cur := 2
	for i := 0; i < 120; i++ {
		*now = now.Add(time.Second)
		q := 3.9
		if i%2 == 0 {
			q = 4.1
		}
		want := p.Decide(metric(*now, q, 0.5), cur)
		if want < cur {
			t.Fatal("oscillation scaled down")
		}
		cur = want
	}
	if cur != 4 {
		t.Fatalf("want 4 with 60s cooldown, got %d", cur)
	}
}
func TestScaleDownCooldown(t *testing.T) {
	p, now := testPolicy(t)
	if got := p.Decide(metric(*now, 0, 0.1), 4); got != 4 {
		t.Fatal(got)
	}
	*now = now.Add(299 * time.Second)
	if got := p.Decide(metric(*now, 0, 0.1), 4); got != 4 {
		t.Fatal(got)
	}
	*now = now.Add(time.Second)
	if got := p.Decide(metric(*now, 0, 0.1), 4); got != 3 {
		t.Fatal(got)
	}
}
func TestInvalidAndStaleMetrics(t *testing.T) {
	p, now := testPolicy(t)
	for _, m := range []Metrics{metric(now.Add(-11*time.Second), 8, 2), metric(now.Add(time.Second), 8, 2), metric(*now, -1, 2), metric(*now, 8, -1)} {
		if got := p.Decide(m, 2); got != 2 {
			t.Fatal(got)
		}
	}
}
func TestTTFTAndQueueSignals(t *testing.T) {
	for _, m := range []Metrics{{QueuePerReplica: 5, P95TTFTSeconds: 0.1}, {QueuePerReplica: 0, P95TTFTSeconds: 1}} {
		p, now := testPolicy(t)
		m.ObservedAt = *now
		if got := p.Decide(m, 2); got != 3 {
			t.Fatal(got)
		}
	}
}
