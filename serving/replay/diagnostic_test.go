package replay

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func phaseRecorder(t *testing.T, count int) *Diagnostics {
	t.Helper()
	d, err := NewDiagnostics(count)
	if err != nil {
		t.Fatal(err)
	}
	if err = d.begin(time.Now(), count); err != nil {
		t.Fatal(err)
	}
	return d
}
func phaseFixtureRow(d *Diagnostics, i int) {
	r := d.request(i)
	r.at("no_timer", time.Now())
	r.at("send_entry", time.Now())
	r.at("get_conn", time.Now())
	r.capture(phaseEvent{Phase: "got_conn", AtNS: time.Since(d.start).Nanoseconds(), Connection: 1, Reused: true})
	seq := r.writeEntry(1, 2, time.Now())
	r.writeReturn(1, seq, 2, 2, nil, time.Now())
}
func phaseByName(t *testing.T, row PhaseRequest, name string) phaseEvent {
	t.Helper()
	for _, e := range row.Events {
		if e.Phase == name {
			return e
		}
	}
	t.Fatalf("missing %s: %+v", name, row)
	return phaseEvent{}
}

func TestDiagnosticActualWakeAndStop(t *testing.T) {
	d := phaseRecorder(t, 1)
	timer := time.NewTimer(time.Millisecond)
	time.Sleep(25 * time.Millisecond)
	channelTime := <-timer.C
	before := time.Now()
	selected, stoppedAt := observeWaitStop(realWaitTimer{timer}, realWaitClock{})
	d.request(0).finalWake(selected, stoppedAt)
	row := d.Snapshot().Requests[0]
	wake := phaseByName(t, row, "post_select")
	stopped := phaseByName(t, row, "timer_stopped")
	if wake.AtNS < before.Sub(d.start).Nanoseconds() || stopped.AtNS < wake.AtNS {
		t.Fatal(row)
	}
	if wake.AtNS-channelTime.Sub(d.start).Nanoseconds() < int64(10*time.Millisecond) {
		t.Fatal("used adjusted timer payload instead of actual wake", row, channelTime)
	}
	// Default path must not record anything or allocate for optional capture.
	beforeCount := d.request(0).count
	if allocations := testing.AllocsPerRun(20, func() { observeWaitStop(realWaitTimer{timer}, realWaitClock{}) }); allocations != 0 {
		t.Fatal(allocations)
	}
	if d.request(0).count != beforeCount {
		t.Fatal("nil capture changed recorder")
	}
}

func TestDiagnosticRealHTTPPhasesAndReuse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	d, err := NewDiagnostics(2)
	if err != nil {
		t.Fatal(err)
	}
	result, err := RunWithDiagnostics(context.Background(), []Req{{0, 2, 1}, {75, 3, 1}}, DefaultConfig(server.URL, 1), d)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range result.Records {
		if r.Status != "ok" {
			t.Fatal(result)
		}
	}
	capture := d.Snapshot()
	if !capture.CaptureValid {
		t.Fatalf("%+v", capture)
	}
	first := phaseByName(t, capture.Requests[0], "got_conn")
	second := phaseByName(t, capture.Requests[1], "got_conn")
	if first.Reused || !second.Reused || !second.WasIdle || second.IdleNS <= 0 || first.Connection == 0 || second.Connection != first.Connection {
		t.Fatal(first, second)
	}
	for i, row := range capture.Requests {
		got := phaseByName(t, row, "got_conn")
		entry := phaseByName(t, row, "write_entry")
		returned := phaseByName(t, row, "write_return")
		if entry.Connection != got.Connection || returned.Connection != got.Connection || entry.AtNS < got.AtNS || returned.AtNS < entry.AtNS {
			t.Fatal(row)
		}
		if math.Abs(float64(returned.AtNS)/1e6-result.Records[i].SentMs) > 0.000001 {
			t.Fatal("SentMs moved from existing finished observation", row, result.Records[i])
		}
	}
}

func TestDiagnosticUnderlyingWriteAndConcurrentIdentity(t *testing.T) {
	const count = 16
	d := phaseRecorder(t, count)
	var wg sync.WaitGroup
	ids := make(chan uint64, count)
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			local, peer := net.Pipe()
			defer local.Close()
			defer peer.Close()
			conn := &emissionConn{Conn: local}
			id := d.connectionID(conn)
			ids <- id
			if d.connectionID(conn) != id {
				t.Error("unstable reused connection identity")
			}
			r := d.request(i)
			r.at("no_timer", time.Now())
			r.at("send_entry", time.Now())
			r.at("get_conn", time.Now())
			r.capture(phaseEvent{Phase: "got_conn", AtNS: time.Since(d.start).Nanoseconds(), Connection: id, Reused: true})
			emission := newEmission(d.start, 2)
			emission.diagnostic = r
			conn.owner.Store(emission)
			request := []byte("POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}")
			done := make(chan error, 1)
			go func() { _, err := conn.Write(request); done <- err }()
			time.Sleep(10 * time.Millisecond)
			if _, err := io.ReadFull(peer, make([]byte, len(request))); err != nil {
				t.Error(err)
			}
			if err := <-done; err != nil {
				t.Error(err)
			}
			if emission.snapshot() < 0 {
				t.Error("missing framed emission")
			}
			conn.release(emission)
		}(i)
	}
	wg.Wait()
	close(ids)
	seen := map[uint64]bool{}
	for id := range ids {
		if id == 0 || seen[id] {
			t.Fatal("connection identity collision", id)
		}
		seen[id] = true
	}
	snapshot := d.Snapshot()
	if !snapshot.CaptureValid {
		t.Fatalf("%+v", snapshot)
	}
	for _, row := range snapshot.Requests {
		a := phaseByName(t, row, "write_entry")
		b := phaseByName(t, row, "write_return")
		if b.AtNS <= a.AtNS || a.Length != b.N || a.Connection != b.Connection {
			t.Fatal(row)
		}
	}
}

func TestDiagnosticMissingRepeatedOverflowAndOutcomes(t *testing.T) {
	t.Run("missing", func(t *testing.T) {
		d := phaseRecorder(t, 1)
		if d.Snapshot().CaptureValid {
			t.Fatal("missing capture accepted")
		}
	})
	t.Run("repeated", func(t *testing.T) {
		d := phaseRecorder(t, 1)
		phaseFixtureRow(d, 0)
		first := phaseByName(t, d.Snapshot().Requests[0], "get_conn")
		d.request(0).at("get_conn", time.Now())
		s := d.Snapshot()
		if s.CaptureValid || phaseByName(t, s.Requests[0], "get_conn").AtNS != first.AtNS {
			t.Fatal("duplicate overwrote or accepted", s)
		}
	})
	t.Run("overflow", func(t *testing.T) {
		d := phaseRecorder(t, 1)
		phaseFixtureRow(d, 0)
		for i := 0; i < 17; i++ {
			seq := d.request(0).writeEntry(1, 2, time.Now())
			d.request(0).writeReturn(1, seq, 2, 2, nil, time.Now())
		}
		s := d.Snapshot()
		if s.CaptureValid || !strings.Contains(strings.Join(s.Requests[0].Problems, ","), "overflow") {
			t.Fatal(s)
		}
	})
	for _, outcome := range []string{"cancelled", "admission_rejected"} {
		t.Run(outcome, func(t *testing.T) {
			d := phaseRecorder(t, 1)
			d.request(0).at("no_timer", time.Now())
			d.request(0).at(outcome, time.Now())
			s := d.Snapshot()
			if !s.CaptureValid {
				t.Fatal(s)
			}
			phaseByName(t, s.Requests[0], outcome)
		})
	}
	t.Run("recorder-reuse", func(t *testing.T) {
		d := phaseRecorder(t, 1)
		if d.begin(time.Now(), 1) == nil {
			t.Fatal("recorder reused")
		}
	})
}

func TestDiagnosticSidecarBoundAndExclusivePublication(t *testing.T) {
	d := phaseRecorder(t, 1)
	phaseFixtureRow(d, 0)
	path := filepath.Join(t.TempDir(), "phase.json")
	if err := d.WriteSidecar(path); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var capture PhaseCapture
	if err = json.Unmarshal(before, &capture); err != nil || !capture.CaptureValid {
		t.Fatal(err, capture)
	}
	if err = d.WriteSidecar(path); !os.IsExist(err) {
		t.Fatal("existing sidecar not rejected", err)
	}
	after, _ := os.ReadFile(path)
	if string(after) != string(before) {
		t.Fatal("overwrote sidecar")
	}
	if _, err = os.Stat(path + ".partial"); !os.IsNotExist(err) {
		t.Fatal("successful partial retained", err)
	}
	invalid := phaseRecorder(t, 1)
	invalidPath := filepath.Join(t.TempDir(), "invalid.json")
	if err = invalid.WriteSidecar(invalidPath); err == nil {
		t.Fatal("missing capture accepted")
	}
	data, _ := os.ReadFile(invalidPath)
	if err = json.Unmarshal(data, &capture); err != nil || capture.CaptureValid {
		t.Fatal(err, capture)
	}
	large := phaseRecorder(t, maxPhaseRequests)
	for i := range large.requests {
		phaseFixtureRow(large, i)
	}
	largePath := filepath.Join(t.TempDir(), "large.json")
	if err = large.WriteSidecar(largePath); err == nil || !strings.Contains(err.Error(), "60 KiB") {
		t.Fatal(err)
	}
	for _, name := range []string{largePath, largePath + ".partial"} {
		if _, err = os.Stat(name); !os.IsNotExist(err) {
			t.Fatal("oversized data written", name, err)
		}
	}
}

func TestDiagnosticRejectsStageOrderAndNegativeIntervals(t *testing.T) {
	phases := []string{"post_select", "timer_stopped", "send_entry", "get_conn", "connect_start", "connect_done", "got_conn", "write_entry", "write_return"}
	makeRecorder := func() *Diagnostics {
		d := phaseRecorder(t, 1)
		for i, name := range phases {
			e := phaseEvent{Phase: name, AtNS: int64(100 + i*10)}
			if name == "got_conn" || name == "write_entry" || name == "write_return" {
				e.Connection = 1
			}
			if name == "write_entry" || name == "write_return" {
				e.Write = 1
				e.Length = 2
			}
			if name == "write_return" {
				e.N = 2
			}
			d.request(0).capture(e)
		}
		return d
	}
	if s := makeRecorder().Snapshot(); !s.CaptureValid {
		t.Fatalf("valid control: %+v", s)
	}
	for i := 0; i < len(phases)-1; i++ {
		for _, kind := range []string{"position", "negative_interval"} {
			t.Run(phases[i]+"-"+phases[i+1]+"/"+kind, func(t *testing.T) {
				d := makeRecorder()
				r := d.request(0)
				if kind == "position" {
					r.events[i], r.events[i+1] = r.events[i+1], r.events[i]
					// Keep ascending time values: reject semantic order, not just the clock.
					r.events[i].AtNS = int64(100 + i*10)
					r.events[i+1].AtNS = int64(100 + (i+1)*10)
				} else {
					r.events[i+1].AtNS = r.events[i].AtNS - 1
				}
				s := d.Snapshot()
				if s.CaptureValid {
					t.Fatalf("accepted invalid %s: %+v", kind, s)
				}
			})
		}
	}
}
