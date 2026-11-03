// CPU fixture / future cold-start adapter runner. It does not contain a model
// server or claim that a loopback adapter measures GPU model readiness.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

type Trial struct {
	ID     string `json:"id"`
	Source string `json:"source"`
	URI    string `json:"uri"`
	SHA256 string `json:"sha256"`
}
type Receipt struct {
	ID     string `json:"id"`
	Ready  bool   `json:"ready"`
	Source string `json:"source"`
	SHA256 string `json:"sha256"`
}
type Timing struct {
	Round   int     `json:"round"`
	Source  string  `json:"source"`
	Seconds float64 `json:"seconds"`
}
type Summary struct{ Median, Min, Max float64 }
type Result struct {
	SHA256   string             `json:"content_sha256"`
	Label    string             `json:"label"`
	Metadata string             `json:"metadata"`
	Trials   []Timing           `json:"trials"`
	Summary  map[string]Summary `json:"summary"`
}

func summary(values []float64) Summary {
	sort.Float64s(values)
	mid := len(values) / 2
	median := values[mid]
	if len(values)%2 == 0 {
		median = (values[mid-1] + median) / 2
	}
	return Summary{median, values[0], values[len(values)-1]}
}
func exchange(ctx context.Context, client *http.Client, endpoint, method string, payload any) (Receipt, error) {
	var receipt Receipt
	var body []byte
	var err error
	if payload != nil {
		body, err = json.Marshal(payload)
		if err != nil {
			return receipt, err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return receipt, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return receipt, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return receipt, fmt.Errorf("adapter status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 65537))
	if err != nil {
		return receipt, err
	}
	if len(data) > 65536 {
		return receipt, errors.New("adapter response too large")
	}
	err = json.Unmarshal(data, &receipt)
	return receipt, err
}
func matches(r Receipt, t Trial) bool {
	return r.ID == t.ID && r.Source == t.Source && r.SHA256 == t.SHA256
}
func coldTrial(ctx context.Context, client *http.Client, base string, trial Trial, poll time.Duration) (elapsed time.Duration, err error) {
	started := time.Now()
	// Cleanup uses a separate bounded context even when the run was cancelled.
	defer func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		ack, e := exchange(cleanup, client, base+"/stop", "POST", trial)
		if e == nil && !matches(ack, trial) {
			e = errors.New("stop trial mismatch")
		}
		if e != nil {
			e = fmt.Errorf("stop trial %s failed; remote cleanup unknown: %w", trial.ID, e)
		}
		err = errors.Join(err, e)
	}()
	ack, err := exchange(ctx, client, base+"/start", "POST", trial)
	if err != nil {
		return 0, err
	}
	if !matches(ack, trial) {
		return 0, errors.New("start trial mismatch")
	}
	ticker := time.NewTicker(poll)
	defer ticker.Stop()
	for {
		ready, e := exchange(ctx, client, base+"/ready?id="+trial.ID, "GET", nil)
		if e != nil {
			return 0, e
		}
		if !matches(ready, trial) {
			return 0, errors.New("readiness trial mismatch")
		}
		if ready.Ready {
			return time.Since(started), nil
		}
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-ticker.C:
		}
	}
}

// Cold pairs two sources in alternating AB/BA order, >=3 rounds. The external
// adapter must reset model processes/page-cache conditions before each start.
func Cold(ctx context.Context, client *http.Client, base, nvme, object, label, metadata, digest string, rounds int, poll time.Duration) (Result, error) {
	result := Result{SHA256: digest, Label: label, Metadata: metadata, Summary: map[string]Summary{}}
	hash, e := hex.DecodeString(digest)
	if e != nil || len(hash) != sha256.Size {
		return result, errors.New("content SHA256 required")
	}
	if rounds < 3 || rounds > 100 || poll <= 0 || nvme == "" || object == "" || metadata == "" || (label != "cpu-fixture" && label != "real-backend") {
		return result, errors.New("3..100 rounds, metadata, both URIs, positive poll and explicit label required")
	}
	values := map[string][]float64{}
	for round := 0; round < rounds; round++ {
		sources := []string{"object-store", "nvme"}
		if round%2 != 0 {
			sources[0], sources[1] = sources[1], sources[0]
		}
		for _, source := range sources {
			uri := object
			if source == "nvme" {
				uri = nvme
			}
			trial := Trial{ID: fmt.Sprintf("t%d-%d", time.Now().UnixNano(), round), Source: source, URI: uri, SHA256: digest}
			elapsed, err := coldTrial(ctx, client, strings.TrimRight(base, "/"), trial, poll)
			if err != nil {
				return result, err
			}
			seconds := elapsed.Seconds()
			result.Trials = append(result.Trials, Timing{round, source, seconds})
			values[source] = append(values[source], seconds)
		}
	}
	for source, v := range values {
		result.Summary[source] = summary(v)
	}
	return result, nil
}
func fileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hash := sha256.New()
	if _, err = io.Copy(hash, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// Stage writes only a new verified cache object using a same-directory atomic
// link. It never replaces an existing cache file; concurrent producers converge.
func Stage(ctx context.Context, client *http.Client, source, destination, digest string, maxBytes int64) (err error) {
	expected, e := hex.DecodeString(digest)
	if e != nil || len(expected) != sha256.Size || maxBytes <= 0 || maxBytes > 5_000_000_000_000 {
		return errors.New("SHA256 and positive byte bound required")
	}
	digest = strings.ToLower(digest)
	if info, e := os.Lstat(destination); e == nil {
		if !info.Mode().IsRegular() || info.Size() > maxBytes {
			return errors.New("existing cache not regular or over bound")
		}
		got, e := fileDigest(destination)
		if e != nil {
			return e
		}
		if got != digest {
			return errors.New("existing cache digest mismatch")
		}
		return nil
	} else if !os.IsNotExist(e) {
		return e
	}
	req, err := http.NewRequestWithContext(ctx, "GET", source, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("weights source status %d", resp.StatusCode)
	}
	file, err := os.CreateTemp(filepath.Dir(destination), ".weights-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	defer file.Close()
	hash := sha256.New()
	n, err := io.Copy(io.MultiWriter(file, hash), io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return err
	}
	if n > maxBytes {
		return errors.New("weights exceed byte bound")
	}
	if hex.EncodeToString(hash.Sum(nil)) != digest {
		return errors.New("download digest mismatch")
	}
	if err = file.Sync(); err != nil {
		return err
	}
	if err = file.Close(); err != nil {
		return err
	}
	if err = os.Link(file.Name(), destination); os.IsExist(err) {
		got, e := fileDigest(destination)
		if e != nil {
			return e
		}
		if got == digest {
			return nil
		}
	}
	return err
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	mode := flag.String("mode", "cold", "cold or stage")
	endpoint := flag.String("adapter-url", "", "explicit reset/start/readiness/stop adapter")
	nvme := flag.String("nvme-uri", "", "NVMe cache source URI")
	object := flag.String("object-uri", "", "object storage source URI")
	label := flag.String("label", "cpu-fixture", "cpu-fixture or real-backend")
	metadata := flag.String("metadata", "", "model revision, image digest, hardware and matched cache conditions")
	rounds := flag.Int("rounds", 3, "alternating matched pairs, minimum3")
	destination := flag.String("destination", "", "stage destination in an existing NVMe cache directory")
	digest := flag.String("sha256", "", "expected immutable weight SHA256")
	maxBytes := flag.Int64("max-bytes", 1<<30, "stage download upper bound")
	timeout := flag.Duration("timeout", 30*time.Minute, "whole experiment/staging deadline")
	flag.Parse()
	if *timeout <= 0 {
		return errors.New("positive timeout required")
	}
	signalContext, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	ctx, cancel := context.WithTimeout(signalContext, *timeout)
	defer cancel()
	transport := &http.Transport{ResponseHeaderTimeout: 10 * time.Second, IdleConnTimeout: 30 * time.Second}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	if *mode == "stage" {
		return Stage(ctx, client, *object, *destination, *digest, *maxBytes)
	}
	if *mode != "cold" {
		return errors.New("unknown mode")
	}
	result, err := Cold(ctx, client, *endpoint, *nvme, *object, *label, *metadata, *digest, *rounds, 100*time.Millisecond)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}
