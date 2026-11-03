package replay

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestEmissionFramingAndPartialWrites(t *testing.T) {
	good := "POST /generate HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}"
	cases := []struct {
		name, request string
		n             int
		err           error
		ok            bool
	}{
		{"complete", good, len(good), nil, true},
		{"partial", good, len(good) - 1, io.ErrShortWrite, false},
		{"short-without-error", good, len(good) - 1, nil, false},
		{"failed", good, 0, io.ErrClosedPipe, false},
		{"missing-body", strings.TrimSuffix(good, "{}"), len(good) - 2, nil, false},
		{"transfer-encoding", "POST / HTTP/1.1\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}", 0, nil, false},
		{"wrong-length", strings.Replace(good, "Length: 2", "Length: 3", 1), len(good), nil, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			e := newEmission(time.Now(), 2)
			n := c.n
			if c.name == "transfer-encoding" {
				n = len(c.request)
			}
			e.observe([]byte(c.request), n, c.err, time.Now())
			if (e.snapshot() >= 0) != c.ok {
				t.Fatal(e.snapshot())
			}
		})
	}
	e := newEmission(time.Now(), 2)
	for _, part := range []string{good[:10], good[10 : len(good)-1], good[len(good)-1:]} {
		e.observe([]byte(part), len(part), nil, time.Now())
	}
	if e.snapshot() < 0 {
		t.Fatal("split writes lost")
	}
	e = newEmission(time.Now(), 2)
	large := "POST / HTTP/1.1\r\nX: " + strings.Repeat("x", 64<<10)
	e.observe([]byte(large), len(large), nil, time.Now())
	if e.snapshot() != -1 || len(e.header) > 64<<10 {
		t.Fatal("unbounded headers")
	}
}
func TestObserverIdentityRelease(t *testing.T) {
	c := &emissionConn{}
	first := newEmission(time.Now(), 2)
	next := newEmission(time.Now(), 2)
	c.owner.Store(first)
	c.owner.Store(next)
	c.release(first)
	if c.owner.Load() != next {
		t.Fatal("stale release detached next request")
	}
	c.release(next)
	if c.owner.Load() != nil {
		t.Fatal("owner retained")
	}
}

type failingWriteConn struct {
	net.Conn
	fail atomic.Bool
}

func (c *failingWriteConn) Write(p []byte) (int, error) {
	if c.fail.Load() {
		return 0, io.ErrUnexpectedEOF
	}
	return c.Conn.Write(p)
}
func TestStaleReusedPOSTIsNotRetried(t *testing.T) {
	var calls, dials atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	var conn *failingWriteConn
	transport := newTransport(DefaultConfig(server.URL, 1))
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		dials.Add(1)
		raw, e := (&net.Dialer{}).DialContext(ctx, network, address)
		if e != nil {
			return nil, e
		}
		conn = &failingWriteConn{Conn: raw}
		return &emissionConn{Conn: conn}, nil
	}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport}
	start := time.Now()
	first := emptyRecord(0)
	send(context.Background(), client, server.URL, []byte(`{"input_ids":[1]}`), start.Add(time.Second), start, &first, 1)
	if first.Status != "ok" {
		t.Fatal(first)
	}
	// Mark the actual pooled TCP connection stale at its next transport write.
	conn.fail.Store(true)
	second := emptyRecord(0)
	send(context.Background(), client, server.URL, []byte(`{"input_ids":[1]}`), start.Add(time.Second), start, &second, 1)
	if second.Status != "transport_error" || second.SentMs != -1 || dials.Load() != 1 || calls.Load() != 1 {
		t.Fatal(second, dials.Load(), calls.Load())
	}
}
func TestTLSVerificationAndHTTP1Policy(t *testing.T) {
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("untrusted TLS reached handler") }))
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.StartTLS()
	defer server.Close()
	c := DefaultConfig(server.URL, 1)
	r, e := Run(context.Background(), []Req{{0, 1, 1}}, c)
	if e != nil {
		t.Fatal(e)
	}
	if r.Records[0].SentMs != -1 || r.Records[0].Status != "transport_error" || !strings.Contains(r.Records[0].Error, "certificate") {
		t.Fatal(r)
	}
	tr := newTransport(c)
	defer tr.CloseIdleConnections()
	if tr.ForceAttemptHTTP2 || tr.TLSNextProto == nil || len(tr.TLSNextProto) != 0 || tr.Proxy != nil {
		t.Fatal("unexpected protocol/proxy policy")
	}
	// The verified TLS path must never relax certificate validation.
	_, e = tr.DialTLSContext(context.Background(), "tcp", strings.TrimPrefix(server.URL, "https://"))
	var verify *tls.CertificateVerificationError
	if !errors.As(e, &verify) {
		t.Fatalf("expected TLS verification error, got %v", e)
	}
}

type shortWriteConn struct{ net.Conn }

func (c *shortWriteConn) Write(p []byte) (int, error) {
	n, e := c.Conn.Write(p[:len(p)/2])
	if e != nil {
		return n, e
	}
	return n, io.ErrShortWrite
}
func TestRealHTTPPartialEmission(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("partial header reached handler") }))
	defer server.Close()
	transport := newTransport(DefaultConfig(server.URL, 1))
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, e := (&net.Dialer{}).DialContext(ctx, network, address)
		if e != nil {
			return nil, e
		}
		return &emissionConn{Conn: &shortWriteConn{Conn: conn}}, nil
	}
	defer transport.CloseIdleConnections()
	start := time.Now()
	r := emptyRecord(0)
	send(context.Background(), &http.Client{Transport: transport}, server.URL, []byte(`{}`), start.Add(time.Second), start, &r, 1)
	if r.SentMs != -1 || r.Status != "transport_error" {
		t.Fatal(r)
	}
}

type waitWriteReturnConn struct {
	net.Conn
	release <-chan struct{}
}

func (c *waitWriteReturnConn) Write(p []byte) (int, error) {
	n, e := c.Conn.Write(p)
	<-c.release
	return n, e
}
func TestEarlyResponseDoesNotProveWriteReturn(t *testing.T) {
	received := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n")
		w.(http.Flusher).Flush()
		close(received)
	}))
	defer server.Close()
	observed := make(chan *emissionConn, 1)
	transport := newTransport(DefaultConfig(server.URL, 1))
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, e := (&net.Dialer{}).DialContext(ctx, network, address)
		if e != nil {
			return nil, e
		}
		wrapped := &emissionConn{Conn: &waitWriteReturnConn{Conn: conn, release: release}}
		observed <- wrapped
		return wrapped, nil
	}
	defer transport.CloseIdleConnections()
	start := time.Now()
	r := emptyRecord(0)
	done := make(chan struct{})
	go func() {
		defer close(done)
		send(context.Background(), &http.Client{Transport: transport}, server.URL, []byte(`{}`), start.Add(time.Second), start, &r, 1)
	}()
	conn := <-observed
	select {
	case <-received:
	case <-time.After(time.Second):
		t.Fatal("no early response")
	}
	// The actual socket peer responded, but Write has not returned to the observer.
	if owner := conn.owner.Load(); owner != nil && owner.snapshot() != -1 {
		t.Fatal("early response falsely proved complete write")
	}
	// The conservative send path may already reject missing emission. Release the
	// writer, then join both send and transport cleanup before finishing this test.
	release <- struct{}{}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("send did not finish")
	}
}

// Under CPU contention net/http's writeLoop can lose the CPU after the request
// bytes reach the socket, while the server answers and client.Do returns. The
// record must wait for that Write's return instead of reporting a missing
// emission (TestReplayTimingAccuracy on a loaded machine, 2026-10-28).
func TestResponseBeforeWriteReturnStillRecordsEmission(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n")
		w.(http.Flusher).Flush()
		// Hold the writer until the response has had time to reach client.Do.
		time.AfterFunc(50*time.Millisecond, func() { close(release) })
	}))
	defer server.Close()
	transport := newTransport(DefaultConfig(server.URL, 1))
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, e := (&net.Dialer{}).DialContext(ctx, network, address)
		if e != nil {
			return nil, e
		}
		return &emissionConn{Conn: &waitWriteReturnConn{Conn: conn, release: release}}, nil
	}
	defer transport.CloseIdleConnections()
	start := time.Now()
	r := emptyRecord(0)
	send(context.Background(), &http.Client{Transport: transport}, server.URL, []byte(`{}`), start.Add(time.Second), start, &r, 1)
	// SentMs is when Write returned (after release), never earlier.
	if r.Status != "ok" || r.SentMs < 50 || r.OutputTokens != 1 {
		t.Fatalf("emission lost when the response beat the write return: %+v", r)
	}
}

func TestEmissionWaitEndsWithContextOrFailure(t *testing.T) {
	e := newEmission(time.Now(), 2)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if got := e.wait(ctx); got != -1 {
		t.Fatalf("unfinished emission reported %v", got)
	}
	e = newEmission(time.Now(), 2)
	e.fail()
	if got := e.wait(context.Background()); got != -1 {
		t.Fatalf("failed emission reported %v", got)
	}
	e = newEmission(time.Now(), 2)
	good := "POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}"
	e.observe([]byte(good), len(good), nil, time.Now())
	if got := e.wait(context.Background()); got < 0 {
		t.Fatalf("complete emission reported %v", got)
	}
}
