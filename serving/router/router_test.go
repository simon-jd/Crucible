package router

import (
	"testing"
	"time"
)

func fixture(t *testing.T) (*Router, *time.Time, Scope) {
	t.Helper()
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	r, err := New(Config{QueueThreshold: 4, MaxEntries: 2, MaxPrefixBytes: 4096, FreshFor: time.Minute, Clock: func() time.Time { return now }}, []Backend{{ID: "a", URL: "http://127.0.0.1:1"}, {ID: "b", URL: "http://127.0.0.1:2"}})
	if err != nil {
		t.Fatal(err)
	}
	scope := Scope{Model: "m", Revision: "r1", Adapter: "base", Tenant: "tenant"}
	for _, id := range []string{"a", "b"} {
		if err := r.Update(id, Snapshot{ObservedAt: now, Queue: 0}); err != nil {
			t.Fatal(err)
		}
	}
	return r, &now, scope
}
func TestPrefixAffinity(t *testing.T) {
	r, now, scope := fixture(t)
	if err := r.Update("b", Snapshot{ObservedAt: *now, Queue: 1, Prefixes: []Prefix{{Scope: scope, Text: "hello "}}}); err != nil {
		t.Fatal(err)
	}
	lease, err := r.Acquire(scope, "hello world")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if lease.Backend.ID != "b" || lease.MatchedBytes != 6 {
		t.Fatalf("got %+v", lease)
	}
}
func TestSpillWhenOverloaded(t *testing.T) {
	r, now, scope := fixture(t)
	if err := r.Update("b", Snapshot{ObservedAt: *now, Queue: 5, Prefixes: []Prefix{{Scope: scope, Text: "hello "}}}); err != nil {
		t.Fatal(err)
	}
	lease, err := r.Acquire(scope, "hello world")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if lease.Backend.ID != "a" {
		t.Fatalf("got %s", lease.Backend.ID)
	}
}

func TestCacheScopeFreshnessAndBounds(t *testing.T) {
	r, now, s := fixture(t)
	if err := r.Update("b", Snapshot{ObservedAt: *now, Prefixes: []Prefix{{Scope: s, Text: "a"}, {Scope: s, Text: "ab"}, {Scope: s, Text: "abc"}}}); err == nil {
		t.Fatal("oversized cache accepted")
	}
	if err := r.Update("b", Snapshot{ObservedAt: *now, Prefixes: []Prefix{{Scope: s, Text: "hello"}}}); err != nil {
		t.Fatal(err)
	}
	other := s
	other.Tenant = "other"
	lease, err := r.Acquire(other, "hello")
	if err != nil {
		t.Fatal(err)
	}
	if lease.MatchedBytes != 0 {
		t.Fatal("cross-tenant affinity")
	}
	lease.Release()
	lease.Release()
	*now = now.Add(time.Minute + time.Nanosecond)
	if _, err := r.Acquire(s, "hello"); err == nil {
		t.Fatal("stale backend selected")
	}
}
func TestReservations(t *testing.T) {
	r, _, s := fixture(t)
	first, err := r.Acquire(s, "x")
	if err != nil {
		t.Fatal(err)
	}
	second, err := r.Acquire(s, "x")
	if err != nil {
		t.Fatal(err)
	}
	if first.Backend.ID == second.Backend.ID {
		t.Fatal("inflight not counted")
	}
	first.Release()
	second.Release()
	if got := r.Inflight(); got != 0 {
		t.Fatal(got)
	}
}

func TestLongestPrefixAndExactThreshold(t *testing.T) {
	r, now, s := fixture(t)
	if err := r.Update("a", Snapshot{ObservedAt: *now, Queue: 0, Prefixes: []Prefix{{Scope: s, Text: "hello"}}}); err != nil {
		t.Fatal(err)
	}
	if err := r.Update("b", Snapshot{ObservedAt: *now, Queue: 4, Prefixes: []Prefix{{Scope: s, Text: "hello world"}}}); err != nil {
		t.Fatal(err)
	}
	lease, err := r.Acquire(s, "hello world!")
	if err != nil {
		t.Fatal(err)
	}
	if lease.Backend.ID != "b" || lease.MatchedBytes != 11 {
		t.Fatal(lease)
	}
	lease.Release()
	if err := r.Update("b", Snapshot{ObservedAt: *now, Queue: 5, Prefixes: []Prefix{{Scope: s, Text: "hello world"}}}); err != nil {
		t.Fatal(err)
	}
	lease, err = r.Acquire(s, "hello world!")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if lease.Backend.ID != "a" || lease.MatchedBytes != 5 {
		t.Fatal(lease)
	}
}
func TestInvalidSnapshotsRejected(t *testing.T) {
	r, now, _ := fixture(t)
	for _, s := range []Snapshot{{ObservedAt: now.Add(time.Second)}, {ObservedAt: now.Add(-2 * time.Minute)}, {ObservedAt: *now, Queue: -1}} {
		if r.Update("a", s) == nil {
			t.Fatal("invalid snapshot accepted")
		}
	}
}
