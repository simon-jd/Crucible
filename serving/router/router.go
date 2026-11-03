// Package router implements independent CPU-side prefix affinity. Prefix records
// are trusted adapter observations, never proof that a GPU KV block still exists.
package router

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"net/url"
	"sync"
	"time"
)

type Scope struct{ Model, Revision, Adapter, Tenant string }
type Prefix struct {
	Scope Scope
	Text  string
}
type Snapshot struct {
	ObservedAt time.Time
	Queue      int
	Prefixes   []Prefix
}
type Backend struct{ ID, URL string }
type Config struct {
	QueueThreshold, MaxEntries, MaxPrefixBytes int
	FreshFor                                   time.Duration
	Clock                                      func() time.Time
}
type entry struct {
	scope  Scope
	size   int
	digest [32]byte
}
type state struct {
	backend         Backend
	observed        time.Time
	ejected         time.Time // set by Eject; cleared only by a snapshot observed after it
	queue, inflight int
	prefixes        []entry
}
type Router struct {
	mu                   sync.Mutex
	cfg                  Config
	states               []*state
	ejections, dialRetry int
}
type Lease struct {
	Backend      Backend
	MatchedBytes int
	release      func()
	once         sync.Once
}

func (l *Lease) Release() { l.once.Do(l.release) }

func New(c Config, backends []Backend) (*Router, error) {
	if c.Clock == nil {
		c.Clock = time.Now
	}
	if c.QueueThreshold < 0 || c.MaxEntries < 1 || c.MaxEntries > 4096 || c.MaxPrefixBytes < 1 || c.MaxPrefixBytes > 65536 || c.FreshFor <= 0 || len(backends) == 0 || len(backends) > 256 {
		return nil, errors.New("invalid router bounds")
	}
	r := &Router{cfg: c}
	seen := map[string]bool{}
	for _, b := range backends {
		u, err := url.Parse(b.URL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || b.ID == "" || seen[b.ID] {
			return nil, fmt.Errorf("invalid backend %q", b.ID)
		}
		seen[b.ID] = true
		r.states = append(r.states, &state{backend: b})
	}
	return r, nil
}
func validScope(s Scope) bool {
	return s.Model != "" && s.Revision != "" && s.Adapter != "" && s.Tenant != "" && len(s.Model)+len(s.Revision)+len(s.Adapter)+len(s.Tenant) <= 1024
}

// Update atomically replaces the complete cache snapshot. Empty Prefixes clears
// affinity. Queue is adapter-observed waiting depth; it may overlap reservations.
func (r *Router) Update(id string, snapshot Snapshot) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.cfg.Clock()
	if snapshot.Queue < 0 || snapshot.Queue > 1000000000 || snapshot.ObservedAt.IsZero() || snapshot.ObservedAt.After(now) || now.Sub(snapshot.ObservedAt) > r.cfg.FreshFor || len(snapshot.Prefixes) > r.cfg.MaxEntries {
		return errors.New("invalid or stale snapshot")
	}
	var target *state
	for _, s := range r.states {
		if s.backend.ID == id {
			target = s
		}
	}
	if target == nil {
		return errors.New("unknown backend")
	}
	if snapshot.ObservedAt.Before(target.observed) {
		return errors.New("out-of-order snapshot")
	}
	prefixes := make([]entry, 0, len(snapshot.Prefixes))
	for _, p := range snapshot.Prefixes {
		if !validScope(p.Scope) || len(p.Text) == 0 || len(p.Text) > r.cfg.MaxPrefixBytes {
			return errors.New("invalid prefix")
		}
		prefixes = append(prefixes, entry{p.Scope, len(p.Text), sha256.Sum256([]byte(p.Text))})
	}
	if !target.ejected.IsZero() && snapshot.ObservedAt.After(target.ejected) {
		target.ejected = time.Time{}
	}
	target.queue = snapshot.Queue
	target.observed = snapshot.ObservedAt
	target.prefixes = prefixes
	return nil
}

// Pressure is max(observed waiting queue, local outstanding reservations): a
// conservative floor, not their sum, because the populations can overlap.
func pressure(s *state) int { return max(s.queue, s.inflight) }
func (r *Router) Acquire(scope Scope, prompt string) (*Lease, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !validScope(scope) {
		return nil, errors.New("invalid scope")
	}
	now := r.cfg.Clock()
	var preferred, least *state
	best := -1
	leastMatch := 0
	digests := map[int][32]byte{}
	for _, s := range r.states {
		if s.observed.IsZero() || s.observed.After(now) || now.Sub(s.observed) > r.cfg.FreshFor || !s.ejected.IsZero() {
			continue
		}
		matched := 0
		for _, e := range s.prefixes {
			if e.scope != scope || e.size > len(prompt) {
				continue
			}
			hash, ok := digests[e.size]
			if !ok {
				hash = sha256.Sum256([]byte(prompt[:e.size]))
				digests[e.size] = hash
			}
			if hash == e.digest && e.size > matched {
				matched = e.size
			}
		}
		if least == nil || pressure(s) < pressure(least) {
			least = s
			leastMatch = matched
		}
		if preferred == nil || matched > best || (matched == best && pressure(s) < pressure(preferred)) {
			preferred = s
			best = matched
		}
	}
	if preferred == nil {
		return nil, errors.New("no fresh backend")
	}
	if pressure(preferred) > r.cfg.QueueThreshold {
		preferred = least
		best = leastMatch
	}
	preferred.inflight++
	chosen := preferred
	return &Lease{Backend: chosen.backend, MatchedBytes: best, release: func() { r.mu.Lock(); defer r.mu.Unlock(); chosen.inflight-- }}, nil
}
func (r *Router) Inflight() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, s := range r.states {
		n += s.inflight
	}
	return n
}

// Eject stops selecting a backend after a proxy transport failure, without waiting
// for its last snapshot to age out. A snapshot observed after the ejection readmits
// it; re-serving an older scrape (what the adapter does for a dead replica) does not.
func (r *Router) Eject(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, s := range r.states {
		if s.backend.ID == id {
			s.ejected = r.cfg.Clock()
			r.ejections++
		}
	}
}
func (r *Router) Ejections() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.ejections
}
func (r *Router) DialRetries() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.dialRetry
}
func (r *Router) countDialRetry() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.dialRetry++
}
