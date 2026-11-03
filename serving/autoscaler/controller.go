package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"
)

type Scale struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Metadata   struct {
		Name            string `json:"name"`
		Namespace       string `json:"namespace"`
		UID             string `json:"uid"`
		ResourceVersion string `json:"resourceVersion"`
	} `json:"metadata"`
	Spec struct {
		Replicas int `json:"replicas"`
	} `json:"spec"`
}
type ScaleClient struct {
	HTTP                            *http.Client
	BaseURL, Namespace, Name, Token string
	Resource                        string // "deployments" (default) or "statefulsets"
}

func (c *ScaleClient) endpoint() (string, error) {
	valid := regexp.MustCompile(`^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$`)
	resource := c.Resource
	if resource == "" {
		resource = "deployments"
	}
	if resource != "deployments" && resource != "statefulsets" {
		return "", errors.New("scale target must be deployments or statefulsets")
	}
	u, err := url.Parse(c.BaseURL)
	if err != nil || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Scheme != "https" && !(u.Scheme == "http" && (u.Hostname() == "127.0.0.1" || u.Hostname() == "::1"))) || !valid.MatchString(c.Namespace) || !valid.MatchString(c.Name) || len(c.Namespace) > 63 || len(c.Name) > 253 {
		return "", errors.New("explicit HTTPS API server and valid target required (HTTP loopback tests only)")
	}
	return strings.TrimRight(c.BaseURL, "/") + "/apis/apps/v1/namespaces/" + c.Namespace + "/" + resource + "/" + c.Name + "/scale", nil
}
func readJSON(resp *http.Response, out any) error {
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("HTTP status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, (1<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 1<<20 {
		return errors.New("response exceeds 1MiB")
	}
	return json.Unmarshal(data, out)
}
func (c *ScaleClient) request(ctx context.Context, method string, scale *Scale) (Scale, error) {
	var out Scale
	endpoint, err := c.endpoint()
	if err != nil {
		return out, err
	}
	var body []byte
	if scale != nil {
		body, err = json.Marshal(scale)
		if err != nil {
			return out, err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return out, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return out, err
	}
	err = readJSON(resp, &out)
	if err != nil {
		return out, err
	}
	if out.APIVersion != "autoscaling/v1" || out.Kind != "Scale" || out.Metadata.Name != c.Name || out.Metadata.Namespace != c.Namespace || out.Metadata.UID == "" || out.Metadata.ResourceVersion == "" || out.Spec.Replicas < 0 {
		return out, errors.New("invalid scale identity/count")
	}
	if scale != nil && (out.Metadata.UID != scale.Metadata.UID || out.Spec.Replicas != scale.Spec.Replicas) {
		return out, errors.New("scale ACK mismatch; outcome unknown, next tick must GET")
	}
	return out, nil
}

type Decision struct {
	Current int    `json:"current"`
	Want    int    `json:"want"`
	Applied bool   `json:"applied"`
	Mode    string `json:"mode"`
}
type Controller struct {
	Policy     *Policy
	HTTP       *http.Client
	MetricsURL string
	Scale      *ScaleClient
	Apply      bool
}

// Tick is serialized by the caller. Every apply tick freshly GETs scale, including
// after a failed/unknown PUT; resourceVersion protects against concurrent writers.
// There is no fire-and-forget pending write or automatic mutation retry.
func (c *Controller) Tick(ctx context.Context) (Decision, error) {
	d := Decision{Mode: "dry-run"}
	var current Scale
	if c.Apply {
		if c.Scale == nil {
			return d, errors.New("apply requires explicit scale client")
		}
		var err error
		current, err = c.Scale.request(ctx, http.MethodGet, nil)
		if err != nil {
			return d, err
		}
		d.Mode = "apply"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.MetricsURL, nil)
	if err != nil {
		return d, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return d, err
	}
	var m Metrics
	if err = readJSON(resp, &m); err != nil {
		return d, err
	}
	if m.Replicas < 1 {
		return d, errors.New("metrics must report observed desired replicas")
	}
	cur := m.Replicas
	if c.Apply {
		cur = current.Spec.Replicas
		if m.Replicas != cur {
			return d, errors.New("metrics replica count differs from fresh desired scale")
		}
	}
	d.Current = cur
	d.Want = c.Policy.Decide(m, cur)
	if !c.Apply || d.Want == cur {
		return d, nil
	}
	current.Spec.Replicas = d.Want
	ack, err := c.Scale.request(ctx, http.MethodPut, &current)
	if err != nil {
		return d, err
	}
	c.Policy.ObserveApplied(ack.Spec.Replicas)
	d.Applied = true
	return d, nil
}

// SimulateBurst uses logical seconds only. It does not measure scheduler, model
// initialization, GPU throughput, or physical replica readiness.
func SimulateBurst(rounds int) (map[string]any, error) {
	if rounds < 3 || rounds > 100 {
		return nil, errors.New("3..100 matched rounds required")
	}
	type event struct{ Second, Desired int }
	type trial struct {
		Round                            int
		Condition                        string
		FirstUpSeconds, FirstDownSeconds int
		Events                           []event
	}
	trials := []trial{}
	ups, downs := []int{}, []int{}
	for round := 0; round < rounds; round++ {
		conditions := []string{"baseline-1x", "burst-4x"}
		if round%2 != 0 {
			conditions[0], conditions[1] = conditions[1], conditions[0]
		}
		for _, condition := range conditions {
			now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
			p, _ := NewPolicy(Config{Min: 1, Max: 8, QueueLow: 1, QueueHigh: 4, TTFTLow: 0.2, TTFTHigh: 0.8, FreshFor: 15 * time.Second}, func() time.Time { return now })
			record := trial{Round: round, Condition: condition, FirstUpSeconds: -1, FirstDownSeconds: -1}
			cur := 2
			for second := 0; second <= 700; second++ {
				q, ttft := 2.0, 0.3
				if condition == "burst-4x" && second >= 10 && second < 190 {
					q = 8
					ttft = 1.2
				}
				if second >= 190 {
					q = 0
					ttft = 0.1
				}
				m := Metrics{ObservedAt: now, QueuePerReplica: q, P95TTFTSeconds: ttft, Replicas: cur}
				want := p.Decide(m, cur)
				if want != cur {
					record.Events = append(record.Events, event{second, want})
					if want > cur && record.FirstUpSeconds < 0 {
						record.FirstUpSeconds = second - 10
					}
					if want < cur && record.FirstDownSeconds < 0 {
						record.FirstDownSeconds = second - 190
					}
					cur = want
					p.ObserveApplied(cur)
				}
				now = now.Add(time.Second)
			}
			trials = append(trials, record)
			if condition == "burst-4x" {
				ups = append(ups, record.FirstUpSeconds)
				downs = append(downs, record.FirstDownSeconds)
			}
		}
	}
	return map[string]any{"label": "synthetic-control-policy-only", "clock": "injected logical seconds", "rounds": rounds, "trials": trials, "burst_first_up_seconds": intSummary(ups), "burst_first_down_seconds_after_load_drop": intSummary(downs)}, nil
}

func intSummary(v []int) map[string]int {
	sort.Ints(v)
	return map[string]int{"median": v[len(v)/2], "min": v[0], "max": v[len(v)-1]}
}
