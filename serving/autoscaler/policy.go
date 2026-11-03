package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

type Metrics struct {
	ObservedAt      time.Time `json:"observed_at"`
	QueuePerReplica float64   `json:"queue_per_replica"`
	P95TTFTSeconds  float64   `json:"p95_ttft_seconds"`
	Replicas        int       `json:"replicas"` // observed desired replicas, not ready replicas
}
type Config struct {
	Min, Max                               int
	QueueLow, QueueHigh, TTFTLow, TTFTHigh float64
	FreshFor                               time.Duration
}
type Policy struct {
	mu                  sync.Mutex
	cfg                 Config
	clock               func() time.Time
	started, lastChange time.Time
	observed            int
	seen                bool
}

func finite(n float64) bool { return !math.IsNaN(n) && !math.IsInf(n, 0) }
func NewPolicy(c Config, clock func() time.Time) (*Policy, error) {
	if c.Min < 1 || c.Max < c.Min || !finite(c.QueueLow) || !finite(c.QueueHigh) || !finite(c.TTFTLow) || !finite(c.TTFTHigh) || c.QueueLow < 0 || c.QueueHigh <= c.QueueLow || c.TTFTLow < 0 || c.TTFTHigh <= c.TTFTLow || c.FreshFor <= 0 {
		return nil, errors.New("invalid policy config")
	}
	if clock == nil {
		clock = time.Now
	}
	return &Policy{cfg: c, clock: clock, started: clock()}, nil
}

// Decide uses observed desired replicas as cur. Decisions alone never consume a
// cooldown: a changed cur must first be observed (or confirmed by a scale ACK).
func (p *Policy) Decide(m Metrics, cur int) (want int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	now := p.clock()
	if !p.seen {
		p.observed = cur
		p.seen = true
	} else if p.observed != cur {
		p.observed = cur
		p.lastChange = now
	}
	if now.Before(p.started) || (!p.lastChange.IsZero() && now.Before(p.lastChange)) || m.ObservedAt.IsZero() || m.ObservedAt.After(now) || now.Sub(m.ObservedAt) > p.cfg.FreshFor || !finite(m.QueuePerReplica) || !finite(m.P95TTFTSeconds) || m.QueuePerReplica < 0 || m.P95TTFTSeconds < 0 {
		return cur
	}
	if cur < p.cfg.Min {
		return p.cfg.Min
	}
	if cur > p.cfg.Max {
		return p.cfg.Max
	}
	if m.QueuePerReplica > p.cfg.QueueHigh || m.P95TTFTSeconds > p.cfg.TTFTHigh {
		if p.lastChange.IsZero() || now.Sub(p.lastChange) >= 60*time.Second {
			return min(cur+1, p.cfg.Max)
		}
	}
	anchor := p.lastChange
	if anchor.IsZero() {
		anchor = p.started
	}
	if m.QueuePerReplica < p.cfg.QueueLow && m.P95TTFTSeconds < p.cfg.TTFTLow && now.Sub(anchor) >= 300*time.Second {
		return max(cur-1, p.cfg.Min)
	}
	return cur
}

// ObserveApplied is only called after a checked conditional-scale ACK.
func (p *Policy) ObserveApplied(cur int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.seen || p.observed != cur {
		p.observed = cur
		p.seen = true
		p.lastChange = p.clock()
	}
}

// Missing/null signals are not equivalent to an idle serving system.
func (m *Metrics) UnmarshalJSON(data []byte) error {
	type plain Metrics
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, key := range []string{"observed_at", "queue_per_replica", "p95_ttft_seconds", "replicas"} {
		v, ok := fields[key]
		if !ok || string(v) == "null" {
			return fmt.Errorf("missing metric %s", key)
		}
	}
	var value plain
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*m = Metrics(value)
	return nil
}
