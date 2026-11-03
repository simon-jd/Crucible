package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestAlternatingColdFixture(t *testing.T) {
	var mu sync.Mutex
	active := ""
	var current Trial
	sources := []string{}
	stops := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch r.URL.Path {
		case "/start":
			var trial Trial
			json.NewDecoder(r.Body).Decode(&trial)
			if active != "" {
				t.Error("prior model not stopped")
			}
			active = trial.ID
			current = trial
			sources = append(sources, trial.Source)
			json.NewEncoder(w).Encode(Receipt{ID: active, Source: current.Source, SHA256: current.SHA256})
		case "/ready":
			json.NewEncoder(w).Encode(Receipt{ID: active, Ready: true, Source: current.Source, SHA256: current.SHA256})
		case "/stop":
			var trial Trial
			json.NewDecoder(r.Body).Decode(&trial)
			if trial.ID != active {
				t.Error("wrong cleanup")
			}
			active = ""
			stops++
			json.NewEncoder(w).Encode(Receipt{ID: trial.ID, Source: trial.Source, SHA256: trial.SHA256})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	result, err := Cold(context.Background(), server.Client(), server.URL, "fixture://nvme", "fixture://object", "cpu-fixture", "loopback adapter, no weights/model/NVMe/object store", strings.Repeat("0", 64), 3, time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(sources, ",") != "object-store,nvme,nvme,object-store,object-store,nvme" || stops != 6 || active != "" || len(result.Trials) != 6 {
		t.Fatal(result, sources, stops)
	}
	bytes, _ := json.Marshal(result)
	t.Log(string(bytes))
}
func TestStageIntegrityBoundsAndCleanup(t *testing.T) {
	data := []byte("fixture weights, not a model")
	hash := sha256.Sum256(data)
	digest := hex.EncodeToString(hash[:])
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; w.Write(data) }))
	defer server.Close()
	dir := t.TempDir()
	path := filepath.Join(dir, "weights")
	if err := Stage(context.Background(), server.Client(), server.URL, path, digest, 3); err == nil {
		t.Fatal("bound not enforced")
	}
	if err := Stage(context.Background(), server.Client(), server.URL, path, strings.Repeat("0", 64), 100); err == nil {
		t.Fatal("checksum not enforced")
	}
	files, _ := os.ReadDir(dir)
	if len(files) != 0 {
		t.Fatal("failed transfer left partial files")
	}
	if err := Stage(context.Background(), server.Client(), server.URL, path, digest, 100); err != nil {
		t.Fatal(err)
	}
	if err := Stage(context.Background(), server.Client(), server.URL, path, digest, 100); err != nil {
		t.Fatal(err)
	}
	if calls != 3 {
		t.Fatal("warm cache downloaded again", calls)
	}
	files, _ = os.ReadDir(dir)
	if len(files) != 1 {
		t.Fatal("temp cache leaked")
	}
}
func TestColdMismatchAndTimeoutStillStop(t *testing.T) {
	for _, mismatch := range []bool{true, false} {
		t.Run(map[bool]string{true: "mismatch", false: "timeout"}[mismatch], func(t *testing.T) {
			var trial Trial
			stopped := false
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/start" {
					json.NewDecoder(r.Body).Decode(&trial)
				}
				receipt := Receipt{ID: trial.ID, Source: trial.Source, SHA256: trial.SHA256}
				if r.URL.Path == "/ready" && mismatch {
					receipt.SHA256 = "wrong"
				}
				if r.URL.Path == "/stop" {
					stopped = true
				}
				json.NewEncoder(w).Encode(receipt)
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
			defer cancel()
			_, err := Cold(ctx, server.Client(), server.URL, "nvme", "object", "cpu-fixture", "fixture", strings.Repeat("0", 64), 3, time.Millisecond)
			if err == nil || !stopped {
				t.Fatal("failed run was not stopped", err, stopped)
			}
		})
	}
}

func TestExecutableSignalStopsTrial(t *testing.T) {
	binary := os.Getenv("TASK14_EXPERIMENTS_BINARY")
	if binary == "" {
		t.Skip("set TASK14_EXPERIMENTS_BINARY to test actual process signals")
	}
	for _, scenario := range []struct {
		name       string
		signal     os.Signal
		stopStatus int
	}{{"sigterm", syscall.SIGTERM, 200}, {"sigint", os.Interrupt, 200}, {"failed-stop", syscall.SIGTERM, 503}} {
		t.Run(scenario.name, func(t *testing.T) {
			var mu sync.Mutex
			var active Trial
			ready := make(chan struct{})
			var once sync.Once
			stops := make(chan Receipt, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				switch r.URL.Path {
				case "/start":
					if err := json.NewDecoder(r.Body).Decode(&active); err != nil {
						t.Error(err)
					}
					json.NewEncoder(w).Encode(Receipt{ID: active.ID, Source: active.Source, SHA256: active.SHA256})
				case "/ready":
					once.Do(func() { close(ready) })
					json.NewEncoder(w).Encode(Receipt{ID: active.ID, Source: active.Source, SHA256: active.SHA256})
				case "/stop":
					var trial Trial
					json.NewDecoder(r.Body).Decode(&trial)
					receipt := Receipt{ID: trial.ID, Source: trial.Source, SHA256: trial.SHA256}
					if !matches(receipt, active) {
						t.Error("stop not bound to active trial")
					}
					stops <- receipt
					w.WriteHeader(scenario.stopStatus)
					json.NewEncoder(w).Encode(receipt)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			cmd := exec.Command(binary, "-mode", "cold", "-adapter-url", server.URL, "-nvme-uri", "fixture://nvme", "-object-uri", "fixture://object", "-sha256", strings.Repeat("0", 64), "-metadata", "owned CPU fixture only", "-timeout", "30s")
			var stderr bytes.Buffer
			cmd.Stderr = &stderr
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- cmd.Wait() }()
			reaped := false
			defer func() {
				if !reaped {
					cmd.Process.Kill()
					<-done
				}
			}()
			select {
			case <-ready:
			case <-time.After(5 * time.Second):
				t.Fatal("child did not acknowledge start")
			}
			if err := cmd.Process.Signal(scenario.signal); err != nil {
				t.Fatal(err)
			}
			select {
			case err := <-done:
				reaped = true
				if err == nil || cmd.ProcessState.ExitCode() != 1 {
					t.Errorf("expected handled cancellation exit1, got %v code%d", err, cmd.ProcessState.ExitCode())
				}
			case <-time.After(7 * time.Second):
				t.Fatal("child not reaped after signal")
			}
			select {
			case <-stops:
			default:
				t.Error("signal exited without stop attempt")
			}
			if !strings.Contains(stderr.String(), "context canceled") {
				t.Error("missing cancellation result", stderr.String())
			}
			if scenario.stopStatus != 200 && !strings.Contains(stderr.String(), "remote cleanup unknown") {
				t.Error("failed stop claimed clean", stderr.String())
			}
		})
	}
}
