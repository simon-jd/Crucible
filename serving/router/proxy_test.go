package router

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
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

func newHTTPFixture(t *testing.T, upstream http.Handler) (*Router, *httptest.Server) {
	t.Helper()
	backend := httptest.NewServer(upstream)
	t.Cleanup(backend.Close)
	r, err := New(Config{QueueThreshold: 4, MaxEntries: 4, MaxPrefixBytes: 4096, FreshFor: time.Minute}, []Backend{{ID: "a", URL: backend.URL}})
	if err != nil {
		t.Fatal(err)
	}
	if err = r.Update("a", Snapshot{ObservedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	tr := &http.Transport{ResponseHeaderTimeout: time.Second}
	t.Cleanup(tr.CloseIdleConnections)
	h, err := NewProxy(r, ProxyConfig{Scope: Scope{Model: "m", Revision: "r", Adapter: "base"}, MaxConcurrent: 2, MaxBodyBytes: 256, RequestTimeout: 2 * time.Second, Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	proxy := httptest.NewServer(h)
	t.Cleanup(proxy.Close)
	return r, proxy
}
func post(ctx context.Context, url string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", url+"/generate", strings.NewReader(`{"text":"hello","stream":true}`))
	if err == nil {
		req.Header.Set("X-Tenant", "t")
	}
	return req, err
}
func waitEmpty(t *testing.T, r *Router) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if r.Inflight() == 0 {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("reservation leaked")
}
func TestStreamingCancellationAndAdmission(t *testing.T) {
	cancelled := make(chan struct{}, 2)
	r, server := newHTTPFixture(t, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Header.Get("X-Tenant") != "" {
			t.Error("tenant control header forwarded")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		<-req.Context().Done()
		cancelled <- struct{}{}
	}))
	var cancel []context.CancelFunc
	var responses []*http.Response
	for i := 0; i < 2; i++ {
		ctx, c := context.WithCancel(context.Background())
		cancel = append(cancel, c)
		defer c()
		req, _ := post(ctx, server.URL)
		resp, err := server.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		responses = append(responses, resp)
		line, err := bufio.NewReader(resp.Body).ReadString('\n')
		if err != nil || line != "data: first\n" {
			t.Fatalf("first token was buffered: %q %v", line, err)
		}
	}
	if r.Inflight() != 2 {
		t.Fatal(r.Inflight())
	}
	req, _ := post(context.Background(), server.URL)
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatal(resp.StatusCode)
	}
	for i, c := range cancel {
		c()
		responses[i].Body.Close()
	}
	for range 2 {
		select {
		case <-cancelled:
		case <-time.After(time.Second):
			t.Fatal("upstream cancellation not propagated")
		}
	}
	waitEmpty(t, r)
}
func TestCompletionFailureAndBodyBounds(t *testing.T) {
	r, s := newHTTPFixture(t, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) { w.WriteHeader(429); fmt.Fprint(w, "busy") }))
	req, _ := post(context.Background(), s.URL)
	resp, err := s.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 429 || string(body) != "busy" {
		t.Fatal(resp.StatusCode, string(body))
	}
	waitEmpty(t, r)
	req, _ = http.NewRequest("POST", s.URL+"/generate", strings.NewReader(strings.Repeat("x", 257)))
	resp, err = s.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 413 {
		t.Fatal(resp.StatusCode)
	}
	waitEmpty(t, r)
}
func TestTransportFailureReleasesWithoutRetry(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	r, s := newHTTPFixture(t, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		mu.Lock()
		calls++
		mu.Unlock()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		conn.Close()
	}))
	req, _ := post(context.Background(), s.URL)
	resp, err := s.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 502 {
		t.Fatal(resp.StatusCode)
	}
	waitEmpty(t, r)
	mu.Lock()
	defer mu.Unlock()
	if calls != 1 {
		t.Fatal("retried POST", calls)
	}
}
func TestConcurrentSnapshotAndReservations(t *testing.T) {
	r, now, scope := fixture(t)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 200 {
			if err := r.Update("a", Snapshot{ObservedAt: *now, Queue: 1, Prefixes: []Prefix{{Scope: scope, Text: "hello"}}}); err != nil {
				t.Error(err)
			}
		}
	}()
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				lease, err := r.Acquire(scope, "hello")
				if err != nil {
					t.Error(err)
					return
				}
				lease.Release()
				lease.Release()
			}
		}()
	}
	wg.Wait()
	if r.Inflight() != 0 {
		t.Fatal(r.Inflight())
	}
}
func TestSnapshotHTTPContract(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		json.NewEncoder(w).Encode(Snapshot{ObservedAt: time.Now(), Queue: 3})
	}))
	defer s.Close()
	snapshot, err := FetchSnapshot(context.Background(), s.Client(), s.URL)
	if err != nil || snapshot.Queue != 3 {
		t.Fatal(snapshot, err)
	}
}

func TestRunnableRouterLifecycle(t *testing.T) {
	binary := os.Getenv("TASK14_ROUTER_BINARY")
	if binary == "" {
		t.Skip("set TASK14_ROUTER_BINARY to exercise built CLI (set in final gate)")
	}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, `{"text":"fixture"}`) }))
	defer backend.Close()
	adapter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(Snapshot{ObservedAt: time.Now()})
	}))
	defer adapter.Close()
	config := map[string]any{"Scope": Scope{Model: "m", Revision: "r", Adapter: "base"}, "Backends": []map[string]string{{"ID": "local", "URL": backend.URL, "MetricsURL": adapter.URL}}}
	data, _ := json.Marshal(config)
	path := filepath.Join(t.TempDir(), "router.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(binary, "-config", path, "-listen", "127.0.0.1:0")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err = cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	defer func() {
		if !waited {
			cmd.Process.Kill()
			cmd.Wait()
		}
	}()
	lines := make(chan string, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		if scanner.Scan() {
			lines <- scanner.Text()
		} else {
			lines <- ""
		}
	}()
	var line string
	select {
	case line = <-lines:
	case <-time.After(5 * time.Second):
		t.Fatal("CLI startup timeout")
	}
	if !strings.HasPrefix(line, "listening ") {
		t.Fatal(line)
	}
	endpoint := "http://" + strings.TrimPrefix(line, "listening ")
	client := &http.Client{Timeout: 2 * time.Second}
	defer client.CloseIdleConnections()
	req, _ := post(context.Background(), endpoint)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, err = io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil || resp.StatusCode != 200 || string(data) != `{"text":"fixture"}` {
		t.Fatal(resp.StatusCode, string(data), err)
	}
	if err = cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err = <-done:
		waited = true
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(7 * time.Second):
		t.Fatal("CLI did not terminate")
	}
	if _, err = client.Do(req); err == nil {
		t.Fatal("listener survived shutdown")
	}
}

// A small send buffer makes a non-reading real TCP peer stall deterministically.
type smallWriteListener struct{ net.Listener }

func (l smallWriteListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err == nil {
		if tcp, ok := c.(*net.TCPConn); ok {
			if err = tcp.SetWriteBuffer(4096); err != nil {
				c.Close()
				return nil, err
			}
		}
	}
	return c, err
}
func deadlineFixture(t *testing.T, budget time.Duration, base context.Context, upstream http.Handler) (*Router, *httptest.Server, <-chan struct{}) {
	t.Helper()
	backend := httptest.NewServer(upstream)
	t.Cleanup(backend.Close)
	r, err := New(Config{QueueThreshold: 4, MaxEntries: 4, MaxPrefixBytes: 4096, FreshFor: time.Minute}, []Backend{{ID: "a", URL: backend.URL}})
	if err != nil {
		t.Fatal(err)
	}
	if err = r.Update("a", Snapshot{ObservedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	tr := &http.Transport{ResponseHeaderTimeout: time.Second}
	t.Cleanup(tr.CloseIdleConnections)
	h, err := NewProxy(r, ProxyConfig{Scope: Scope{Model: "m", Revision: "r", Adapter: "base"}, MaxConcurrent: 1, MaxBodyBytes: 256, RequestTimeout: budget, Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{}, 8)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer func() { done <- struct{}{} }()
		h.ServeHTTP(w, req)
	}))
	server.Listener = smallWriteListener{server.Listener}
	server.Config.BaseContext = func(net.Listener) context.Context { return base }
	server.Start()
	t.Cleanup(server.Close)
	return r, server, done
}
func stalledClient(t *testing.T, server *httptest.Server, body string, length int) net.Conn {
	t.Helper()
	c, err := net.Dial("tcp", server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	if err = c.(*net.TCPConn).SetReadBuffer(1024); err != nil {
		t.Fatal(err)
	}
	if _, err = fmt.Fprintf(c, "POST /generate HTTP/1.1\r\nHost: fixture\r\nX-Tenant: t\r\nContent-Length: %d\r\n\r\n%s", length, body); err != nil {
		t.Fatal(err)
	}
	return c
}
func TestSlowConsumerDeadlineAndCancellation(t *testing.T) {
	for _, cancelEarly := range []bool{false, true} {
		t.Run(map[bool]string{false: "deadline", true: "parent-cancel"}[cancelEarly], func(t *testing.T) {
			base, cancel := context.WithCancel(context.Background())
			defer cancel()
			started := make(chan struct{})
			ended := make(chan struct{})
			budget := 200 * time.Millisecond
			if cancelEarly {
				budget = 5 * time.Second
			}
			r, server, done := deadlineFixture(t, budget, base, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(ended)
				close(started)
				chunk := make([]byte, 32<<10)
				for {
					if _, err := w.Write(chunk); err != nil {
						return
					}
					w.(http.Flusher).Flush()
				}
			}))
			body := `{"text":"hello"}`
			client := stalledClient(t, server, body, len(body))
			defer client.Close()
			select {
			case <-started:
			case <-time.After(time.Second):
				t.Fatal("upstream never started")
			}
			if cancelEarly {
				time.Sleep(30 * time.Millisecond)
				cancel()
			}
			// The client has not read or closed the connection at this assertion.
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("stalled downstream retained handler beyond budget")
			}
			if r.Inflight() != 0 {
				t.Fatal("lease retained after handler timeout", r.Inflight())
			}
			select {
			case <-ended:
			case <-time.After(time.Second):
				t.Fatal("upstream did not stop")
			}
		})
	}
}
func TestRequestBodyLifetimeBudget(t *testing.T) {
	r, server, done := deadlineFixture(t, 150*time.Millisecond, context.Background(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("incomplete body reached upstream") }))
	client := stalledClient(t, server, `{`, 100)
	defer client.Close()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("body read outlived request budget")
	}
	if r.Inflight() != 0 {
		t.Fatal("incomplete body acquired lease")
	}
}
func TestDeadlineKeepaliveReuse(t *testing.T) {
	r, server, done := deadlineFixture(t, 150*time.Millisecond, context.Background(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ok") }))
	client, err := net.Dial("tcp", server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	client.SetDeadline(time.Now().Add(3 * time.Second))
	reader := bufio.NewReader(client)
	for i := 0; i < 2; i++ {
		body := `{"text":"hello"}`
		fmt.Fprintf(client, "POST /generate HTTP/1.1\r\nHost: fixture\r\nX-Tenant: t\r\nContent-Length: %d\r\n\r\n%s", len(body), body)
		response, err := http.ReadResponse(reader, nil)
		if err != nil {
			t.Fatal("same keepalive connection failed", err)
		}
		data, err := io.ReadAll(response.Body)
		response.Body.Close()
		if err != nil || response.StatusCode != 200 || string(data) != "ok" {
			t.Fatal(response.StatusCode, string(data), err)
		}
		<-done
		if i == 0 {
			time.Sleep(250 * time.Millisecond)
		}
	}
	if r.Inflight() != 0 {
		t.Fatal("lease leak")
	}
}
func TestDeadlineCapabilityRequired(t *testing.T) {
	r, _, _ := fixture(t)
	h, err := NewProxy(r, ProxyConfig{Scope: Scope{Model: "m", Revision: "r", Adapter: "base"}, MaxConcurrent: 1, MaxBodyBytes: 256, RequestTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	// Recorder lacks socket deadline support. Fail closed before reading/forwarding.
	defer func() {
		if got := recover(); got != http.ErrAbortHandler {
			t.Errorf("want explicit abort for unsupported deadlines, got %v", got)
		}
		if r.Inflight() != 0 {
			t.Error("unsupported writer acquired a lease")
		}
	}()
	req, _ := post(context.Background(), "http://fixture")
	h.ServeHTTP(httptest.NewRecorder(), req)
}
