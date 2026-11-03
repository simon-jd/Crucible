package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestCooldownExactAndLimits(t *testing.T) {
	p, now := testPolicy(t)
	m := metric(*now, 10, 2)
	if p.Decide(m, 2) != 3 {
		t.Fatal("first up")
	}
	p.ObserveApplied(3)
	*now = now.Add(60*time.Second - time.Nanosecond)
	if p.Decide(metric(*now, 10, 2), 3) != 3 {
		t.Fatal("early up")
	}
	*now = now.Add(time.Nanosecond)
	if p.Decide(metric(*now, 10, 2), 3) != 4 {
		t.Fatal("exact up")
	}
	p.ObserveApplied(4)
	*now = now.Add(300*time.Second - time.Nanosecond)
	if p.Decide(metric(*now, 0, 0.1), 4) != 4 {
		t.Fatal("early down")
	}
	*now = now.Add(time.Nanosecond)
	if p.Decide(metric(*now, 0, 0.1), 4) != 3 {
		t.Fatal("exact down")
	}
	if p.Decide(metric(*now, 10, 2), 8) != 8 {
		t.Fatal("max")
	}
	for _, v := range []float64{math.NaN(), math.Inf(1)} {
		if p.Decide(metric(*now, v, 2), 8) != 8 {
			t.Fatal("invalid")
		}
	}
}
func TestPolicyConcurrent(t *testing.T) {
	p, now := testPolicy(t)
	m := metric(*now, 5, 1)
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				p.Decide(m, 2)
			}
		}()
	}
	wg.Wait()
}
func scaleFixture(n int) Scale {
	var s Scale
	s.APIVersion = "autoscaling/v1"
	s.Kind = "Scale"
	s.Metadata.Name = "model"
	s.Metadata.Namespace = "test"
	s.Metadata.UID = "uid-1"
	s.Metadata.ResourceVersion = "rv1"
	s.Spec.Replicas = n
	return s
}
func TestControllerDryRunAndConditionalScale(t *testing.T) {
	p, now := testPolicy(t)
	scale := scaleFixture(2)
	calls := []string{}
	fail := true
	badAck := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.Method+" "+r.URL.Path)
		if r.URL.Path == "/metrics" {
			m := metric(*now, 8, 1)
			m.Replicas = scale.Spec.Replicas
			json.NewEncoder(w).Encode(m)
			return
		}
		if r.URL.Path != "/apis/apps/v1/namespaces/test/deployments/model/scale" {
			t.Error(r.URL.Path)
			http.NotFound(w, r)
			return
		}
		if r.Method == "PUT" {
			var next Scale
			if err := json.NewDecoder(r.Body).Decode(&next); err != nil {
				t.Error(err)
			}
			if next.Metadata.ResourceVersion != scale.Metadata.ResourceVersion {
				t.Error("missing conditional version")
			}
			if fail {
				http.Error(w, "conflict", 409)
				return
			}
			scale = next
			scale.Metadata.ResourceVersion = "rv2"
			if badAck {
				out := scale
				out.Metadata.UID = "different"
				json.NewEncoder(w).Encode(out)
				return
			}
		}
		json.NewEncoder(w).Encode(scale)
	}))
	defer server.Close()
	c := Controller{Policy: p, HTTP: server.Client(), MetricsURL: server.URL + "/metrics", Scale: &ScaleClient{HTTP: server.Client(), BaseURL: server.URL, Namespace: "test", Name: "model"}}
	d, err := c.Tick(context.Background())
	if err != nil || d.Applied || d.Want != 3 || len(calls) != 1 {
		t.Fatal(d, err, calls)
	}
	c.Apply = true
	if _, err = c.Tick(context.Background()); err == nil {
		t.Fatal("conflict swallowed")
	}
	fail = false
	badAck = true
	if _, err = c.Tick(context.Background()); err == nil {
		t.Fatal("bad ACK accepted")
	}
	// Server accepted desired=3, but the ambiguous ACK was not applied locally.
	if !p.lastChange.IsZero() {
		t.Fatal("unknown ACK consumed cooldown")
	}
	badAck = false
	d, err = c.Tick(context.Background())
	if err != nil || d.Current != 3 || d.Want != 3 || d.Applied {
		t.Fatal(d, err)
	}
	// Each attempt must begin with a GET; never blindly replay the PUT.
	expected := []string{"GET /metrics", "GET /apis/apps/v1/namespaces/test/deployments/model/scale", "GET /metrics", "PUT /apis/apps/v1/namespaces/test/deployments/model/scale", "GET /apis/apps/v1/namespaces/test/deployments/model/scale", "GET /metrics", "PUT /apis/apps/v1/namespaces/test/deployments/model/scale", "GET /apis/apps/v1/namespaces/test/deployments/model/scale", "GET /metrics"}
	if fmt.Sprint(calls) != fmt.Sprint(expected) {
		t.Fatal(calls)
	}
	*now = now.Add(60 * time.Second)
	d, err = c.Tick(context.Background())
	if err != nil || !d.Applied || d.Want != 4 {
		t.Fatal(d, err)
	}
}
func TestFourTimesBurstExperiment(t *testing.T) {
	result, err := SimulateBurst(3)
	if err != nil {
		t.Fatal(err)
	}
	up := result["burst_first_up_seconds"].(map[string]int)
	down := result["burst_first_down_seconds_after_load_drop"].(map[string]int)
	if up["median"] != 0 || up["max"] != 0 || down["median"] != 240 || down["min"] != 240 {
		t.Fatal(result)
	}
	body, _ := json.Marshal(result)
	t.Log(string(body))
	if _, err = SimulateBurst(2); err == nil {
		t.Fatal("insufficient paired rounds")
	}
}
func TestMissingMetricsRejected(t *testing.T) {
	var m Metrics
	for _, s := range []string{`{}`, `{"observed_at":"2026-01-01T00:00:00Z","queue_per_replica":null,"p95_ttft_seconds":0,"replicas":2}`} {
		if json.Unmarshal([]byte(s), &m) == nil {
			t.Fatal("missing metric accepted")
		}
	}
}
