package router

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestEjectSkipsBackendUntilNewerSnapshot(t *testing.T) {
	r, now, scope := fixture(t)
	snap := func() Snapshot { return Snapshot{ObservedAt: *now, Prefixes: []Prefix{{Scope: scope, Text: "hello "}}} }
	if err := r.Update("b", snap()); err != nil {
		t.Fatal(err)
	}
	r.Eject("b")
	if r.Ejections() != 1 {
		t.Fatalf("ejections %d", r.Ejections())
	}
	// The adapter re-serves its last good scrape for a dead replica: same ObservedAt.
	if err := r.Update("b", snap()); err != nil {
		t.Fatal(err)
	}
	for range 3 {
		lease, err := r.Acquire(scope, "hello world")
		if err != nil {
			t.Fatal(err)
		}
		if lease.Backend.ID != "a" {
			t.Fatalf("ejected backend chosen: %+v", lease)
		}
		lease.Release()
	}
	*now = now.Add(time.Second)
	if err := r.Update("b", snap()); err != nil {
		t.Fatal(err)
	}
	lease, err := r.Acquire(scope, "hello world")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if lease.Backend.ID != "b" {
		t.Fatalf("newer snapshot did not readmit b: %+v", lease)
	}
}

func TestEjectUnknownBackendIsIgnored(t *testing.T) {
	r, _, _ := fixture(t)
	r.Eject("zz")
	if r.Ejections() != 0 {
		t.Fatal("unknown backend counted")
	}
}

// twoBackendProxy puts `first` ahead of a live backend; on equal pressure the
// router picks the first listed backend.
func twoBackendProxy(t *testing.T, firstURL string, retryDial bool) (*Router, *httptest.Server) {
	t.Helper()
	live := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "ok") }))
	t.Cleanup(live.Close)
	r, err := New(Config{QueueThreshold: 4, MaxEntries: 4, MaxPrefixBytes: 4096, FreshFor: time.Minute}, []Backend{{ID: "first", URL: firstURL}, {ID: "live", URL: live.URL}})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"first", "live"} {
		if err = r.Update(id, Snapshot{ObservedAt: time.Now()}); err != nil {
			t.Fatal(err)
		}
	}
	tr := &http.Transport{ResponseHeaderTimeout: time.Second}
	t.Cleanup(tr.CloseIdleConnections)
	h, err := NewProxy(r, ProxyConfig{Scope: Scope{Model: "m", Revision: "r", Adapter: "base"}, MaxConcurrent: 2, MaxBodyBytes: 256, RequestTimeout: 2 * time.Second, Transport: tr, RetryDialErrors: retryDial})
	if err != nil {
		t.Fatal(err)
	}
	proxy := httptest.NewServer(h)
	t.Cleanup(proxy.Close)
	return r, proxy
}

// refusedURL is an address nothing listens on: the request never reaches a server.
func refusedURL(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	url := "http://" + listener.Addr().String()
	listener.Close()
	return url
}

func statusOf(t *testing.T, url string) int {
	t.Helper()
	req, err := post(context.Background(), url)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return resp.StatusCode
}

func TestProxyEjectsBackendOnConnectError(t *testing.T) {
	r, proxy := twoBackendProxy(t, refusedURL(t), false)
	if got := statusOf(t, proxy.URL); got != http.StatusBadGateway {
		t.Fatalf("first request to dead backend: %d", got)
	}
	if r.Ejections() != 1 || r.DialRetries() != 0 {
		t.Fatalf("ejections %d retries %d", r.Ejections(), r.DialRetries())
	}
	for range 3 {
		if got := statusOf(t, proxy.URL); got != 200 {
			t.Fatalf("request after ejection: %d", got)
		}
	}
	waitEmpty(t, r)
}

func TestProxyRetriesDialErrorOnAnotherBackend(t *testing.T) {
	r, proxy := twoBackendProxy(t, refusedURL(t), true)
	if got := statusOf(t, proxy.URL); got != 200 {
		t.Fatalf("dial error was not retried: %d", got)
	}
	if r.Ejections() != 1 || r.DialRetries() != 1 {
		t.Fatalf("ejections %d retries %d", r.Ejections(), r.DialRetries())
	}
	waitEmpty(t, r)
}

func TestProxyDoesNotRetryAfterBackendAccepted(t *testing.T) {
	// The backend accepts and then drops the connection: the POST may have run, so no retry.
	var mu sync.Mutex
	hits := 0
	drop := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		mu.Unlock()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			conn.Close()
		}
	}))
	t.Cleanup(drop.Close)
	r, proxy := twoBackendProxy(t, drop.URL, true)
	if got := statusOf(t, proxy.URL); got != http.StatusBadGateway {
		t.Fatalf("accepted-then-dropped request: %d", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if r.DialRetries() != 0 || r.Ejections() != 1 || hits != 1 {
		t.Fatalf("retries %d ejections %d hits %d", r.DialRetries(), r.Ejections(), hits)
	}
	waitEmpty(t, r)
}

func TestProxyDoesNotEjectOnClientCancel(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
		select {
		case <-req.Context().Done():
		case <-time.After(3 * time.Second):
		}
	}))
	t.Cleanup(slow.Close)
	r, proxy := twoBackendProxy(t, slow.URL, true)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	req, err := post(ctx, proxy.URL)
	if err != nil {
		t.Fatal(err)
	}
	if resp, err := http.DefaultClient.Do(req); err == nil {
		resp.Body.Close()
	}
	waitEmpty(t, r)
	if r.Ejections() != 0 {
		t.Fatalf("client cancel ejected the backend: %d", r.Ejections())
	}
}
