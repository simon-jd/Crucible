package replay

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

const MaxPhaseSidecarBytes = 60 << 10
const maxPhaseEvents = 48
const maxPhaseWrites = 16
const maxPhaseRequests = 256
const maxIntermediateWakes = 65535

// Diagnostics is optional, single-run, bounded observation, not sender validity.
// Request locks never nest with emission locks; no hot-path formatting or I/O.
// Snapshot/WriteSidecar are called only after the shared Run implementation ends.
type Diagnostics struct {
	mu             sync.Mutex
	started        bool
	start          time.Time
	requests       []phaseRequest
	nextConnection atomic.Uint64
}
type phaseRequest struct {
	mu                   sync.Mutex
	parent               *Diagnostics
	index                int
	events               [maxPhaseEvents]phaseEvent
	count, writes        int
	overflow             bool
	intermediateWakes    uint32
	intermediateOverflow bool
}
type phaseEvent struct {
	Phase      string `json:"phase"`
	AtNS       int64  `json:"at_ns"`
	Connection uint64 `json:"connection,omitempty"`
	Write      int    `json:"write,omitempty"`
	Length     int    `json:"length,omitempty"`
	N          int    `json:"n,omitempty"`
	Reused     bool   `json:"reused,omitempty"`
	WasIdle    bool   `json:"was_idle,omitempty"`
	IdleNS     int64  `json:"idle_ns,omitempty"`
	Network    string `json:"network,omitempty"`
	Address    string `json:"address,omitempty"`
	Error      string `json:"error,omitempty"`
	err        error
}
type PhaseRequest struct {
	Index             int          `json:"index"`
	Events            []phaseEvent `json:"events"`
	Problems          []string     `json:"problems,omitempty"`
	IntermediateWakes uint32       `json:"intermediate_wakes"`
}
type PhaseCapture struct {
	Version      int            `json:"version"`
	CaptureValid bool           `json:"capture_valid"`
	Problems     []string       `json:"problems,omitempty"`
	Requests     []PhaseRequest `json:"requests"`
}

func NewDiagnostics(count int) (*Diagnostics, error) {
	if count < 1 || count > maxPhaseRequests {
		return nil, fmt.Errorf("diagnostic request bound: %d", count)
	}
	d := &Diagnostics{requests: make([]phaseRequest, count)}
	for i := range d.requests {
		d.requests[i].parent = d
		d.requests[i].index = i
	}
	return d, nil
}
func (d *Diagnostics) begin(start time.Time, count int) error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.started || len(d.requests) != count {
		return errors.New("diagnostics reused or wrong request count")
	}
	d.start, d.started = start, true
	return nil
}
func (d *Diagnostics) request(i int) *phaseRequest {
	if d == nil {
		return nil
	}
	return &d.requests[i]
}
func shortPhaseString(s string) string {
	if len(s) > 256 {
		return s[:256]
	}
	return s
}
func (r *phaseRequest) appendLocked(e phaseEvent) {
	if r.count == len(r.events) {
		r.overflow = true
		return
	}
	r.events[r.count] = e
	r.count++
}
func (r *phaseRequest) at(phase string, at time.Time) {
	r.capture(phaseEvent{Phase: phase, AtNS: at.Sub(r.parent.start).Nanoseconds()})
}
func (r *phaseRequest) capture(e phaseEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.appendLocked(e)
}
func (r *phaseRequest) writeEntry(id uint64, length int, at time.Time) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.writes++
	if r.writes > maxPhaseWrites {
		r.overflow = true
		return r.writes
	}
	r.appendLocked(phaseEvent{Phase: "write_entry", AtNS: at.Sub(r.parent.start).Nanoseconds(), Connection: id, Write: r.writes, Length: length})
	return r.writes
}
func (r *phaseRequest) writeReturn(id uint64, seq, length, n int, err error, at time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if seq > maxPhaseWrites {
		r.overflow = true
		return
	}
	r.appendLocked(phaseEvent{Phase: "write_return", AtNS: at.Sub(r.parent.start).Nanoseconds(), Connection: id, Write: seq, Length: length, N: n, err: err})
}
func (d *Diagnostics) connectionID(c *emissionConn) uint64 {
	id := c.diagnosticID.Load()
	if id == 0 {
		c.diagnosticID.CompareAndSwap(0, d.nextConnection.Add(1))
		id = c.diagnosticID.Load()
	}
	return id
}

// Version2 records only the final blocking wake (or cancellation wake).
// Intermediate waits use a saturating optional counter, never event slots.
func (r *phaseRequest) finalWake(selected, stopped time.Time) {
	r.at("post_select", selected)
	r.at("timer_stopped", stopped)
}
func (r *phaseRequest) intermediateWake() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.intermediateWakes == maxIntermediateWakes {
		r.intermediateOverflow = true
		return
	}
	r.intermediateWakes++
}

func (d *Diagnostics) Snapshot() PhaseCapture {
	d.mu.Lock()
	started := d.started
	d.mu.Unlock()
	out := PhaseCapture{Version: 2, CaptureValid: started, Requests: make([]PhaseRequest, len(d.requests))}
	if !started {
		out.Problems = append(out.Problems, "run_not_started")
	}
	for i := range d.requests {
		r := &d.requests[i]
		r.mu.Lock()
		row := PhaseRequest{Index: i, Events: append([]phaseEvent(nil), r.events[:r.count]...), IntermediateWakes: r.intermediateWakes}
		intermediateOverflow := r.intermediateOverflow
		overflow := r.overflow
		r.mu.Unlock()
		if intermediateOverflow {
			row.Problems = append(row.Problems, "intermediate_wake_overflow")
		}
		if overflow {
			row.Problems = append(row.Problems, "capture_overflow")
		}
		counts := map[string]int{}
		positions := map[string]int{}
		var connection uint64
		entries := map[int]phaseEvent{}
		returns := map[int]bool{}
		for j := range row.Events {
			e := &row.Events[j]
			if e.err != nil {
				e.Error = shortPhaseString(e.err.Error())
				e.err = nil
			}
			counts[e.Phase]++
			if counts[e.Phase] == 1 {
				positions[e.Phase] = j
			}
			if e.AtNS < 0 {
				row.Problems = append(row.Problems, "negative_time")
			}
			if e.Phase == "got_conn" && counts[e.Phase] == 1 {
				connection = e.Connection
			}
			if e.Phase == "write_entry" {
				if _, exists := entries[e.Write]; exists {
					row.Problems = append(row.Problems, "repeated_write_entry")
				} else {
					entries[e.Write] = *e
				}
			}
			if e.Phase == "write_return" {
				entry, exists := entries[e.Write]
				if !exists || returns[e.Write] || entry.Connection != e.Connection || entry.Length != e.Length || e.AtNS < entry.AtNS {
					row.Problems = append(row.Problems, "unmatched_write_return")
				}
				returns[e.Write] = true
			}
		}
		requireOne := func(name string) {
			if counts[name] == 0 {
				row.Problems = append(row.Problems, "missing_"+name)
			}
			if counts[name] > 1 {
				row.Problems = append(row.Problems, "repeated_"+name)
			}
		}
		ordered := func(left, right string) {
			if counts[left] > 0 && counts[right] > 0 {
				a, b := positions[left], positions[right]
				if a >= b || row.Events[a].AtNS > row.Events[b].AtNS {
					row.Problems = append(row.Problems, "phase_order:"+left+"->"+right)
				}
			}
		}
		timerEnd := "timer_stopped"
		if counts["no_timer"] > 0 {
			timerEnd = "no_timer"
			requireOne("no_timer")
			if row.IntermediateWakes != 0 {
				row.Problems = append(row.Problems, "no_timer_with_intermediate_wakes")
			}
			if counts["post_select"]+counts["timer_stopped"] != 0 {
				row.Problems = append(row.Problems, "conflicting_timer_path")
			}
		} else {
			requireOne("post_select")
			requireOne("timer_stopped")
			ordered("post_select", "timer_stopped")
		}
		if counts["cancelled"]+counts["admission_rejected"] > 0 {
			if counts["cancelled"]+counts["admission_rejected"] != 1 || counts["send_entry"] != 0 {
				row.Problems = append(row.Problems, "conflicting_admission_path")
			}
		} else {
			for _, name := range []string{"send_entry", "get_conn", "got_conn"} {
				requireOne(name)
			}
			ordered(timerEnd, "send_entry")
			ordered("send_entry", "get_conn")
			ordered("get_conn", "got_conn")
			ordered("get_conn", "connect_start")
			ordered("connect_start", "connect_done")
			ordered("connect_done", "got_conn")
			ordered("got_conn", "write_entry")
			if counts["got_conn"] == 1 && !row.Events[positions["got_conn"]].Reused {
				requireOne("connect_start")
				requireOne("connect_done")
			}
			if counts["connect_start"] != counts["connect_done"] || counts["connect_start"] > 1 {
				row.Problems = append(row.Problems, "ambiguous_connect_callbacks")
			}
			if len(entries) == 0 || len(entries) != len(returns) {
				row.Problems = append(row.Problems, "missing_write_pair")
			}
			for _, entry := range entries {
				if connection == 0 || entry.Connection != connection || !returns[entry.Write] {
					row.Problems = append(row.Problems, "write_connection_or_pair")
				}
				if counts["got_conn"] > 0 && entry.AtNS < row.Events[positions["got_conn"]].AtNS {
					row.Problems = append(row.Problems, "write_before_got_conn")
				}
			}
		}
		if len(row.Problems) > 0 {
			out.CaptureValid = false
		}
		out.Requests[i] = row
	}
	return out
}

// Publish only a fully serialized bounded file. Existing targets are untouched;
// failed writes retain an explicitly .partial file, never a published sidecar.
func (d *Diagnostics) WriteSidecar(path string) error {
	capture := d.Snapshot()
	data, err := json.Marshal(capture)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if len(data) > MaxPhaseSidecarBytes {
		return errors.New("diagnostic sidecar exceeds 60 KiB")
	}
	if _, err = os.Lstat(path); err == nil {
		return os.ErrExist
	} else if !os.IsNotExist(err) {
		return err
	}
	partial := path + ".partial"
	f, err := os.OpenFile(partial, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	n, writeErr := f.Write(data)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	if n != len(data) {
		return errors.New("short diagnostic sidecar write")
	}
	if closeErr != nil {
		return closeErr
	}
	if err = os.Link(partial, path); err != nil {
		return err
	}
	if err = os.Remove(partial); err != nil {
		return err
	}
	if !capture.CaptureValid {
		return errors.New("diagnostic capture invalid; bounded sidecar retained")
	}
	return nil
}
