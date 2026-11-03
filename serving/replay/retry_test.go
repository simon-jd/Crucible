package replay

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const firstChunk = "data: {\"output_ids\":[11],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":null}}\n\n"
const lastChunk = "data: {\"output_ids\":[12,13],\"meta_info\":{\"completion_tokens\":3,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n"

// faultyServer answers the first `bad` requests with fault(w) and the rest normally.
func faultyServer(bad int32, fault func(http.ResponseWriter)) (*httptest.Server, *atomic.Int32) {
	var calls atomic.Int32
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		if calls.Add(1) <= bad {
			fault(w)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, firstChunk)
		w.(http.Flusher).Flush()
		time.Sleep(5 * time.Millisecond)
		fmt.Fprint(w, lastChunk)
	}))
	return s, &calls
}

// cutStream sends the headers and one token, then drops the connection, like a
// replica killed while streaming.
func cutStream(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprint(w, firstChunk)
	w.(http.Flusher).Flush()
	time.Sleep(20 * time.Millisecond)
	conn, _, err := w.(http.Hijacker).Hijack()
	if err == nil {
		conn.Close()
	}
}

func runOne(t *testing.T, url string, retries int) Record {
	t.Helper()
	c := DefaultConfig(url, 1)
	c.ClientRetries = retries
	res, err := Run(context.Background(), []Req{{0, 8, 3}}, c)
	if err != nil {
		t.Fatal(err)
	}
	return res.Records[0]
}

func TestClientRetryOffByDefaultKeepsCutStreamFailed(t *testing.T) {
	s, calls := faultyServer(1, cutStream)
	defer s.Close()
	r := runOne(t, s.URL, 0)
	if r.Status != "stream_error" || !strings.HasPrefix(r.Error, "stream read:") || calls.Load() != 1 || r.Attempts != nil {
		t.Fatalf("no retry expected: %+v calls=%d", r, calls.Load())
	}
}

func TestClientRetryResendsCutStreamAndKeepsBothAttempts(t *testing.T) {
	s, calls := faultyServer(1, cutStream)
	defer s.Close()
	r := runOne(t, s.URL, 1)
	if r.Status != "ok" || r.OutputTokens != 3 || calls.Load() != 2 || len(r.Attempts) != 2 {
		t.Fatalf("retry did not complete: %+v calls=%d", r, calls.Load())
	}
	first, last := r.Attempts[0], r.Attempts[1]
	if first.Status != "stream_error" || !strings.HasPrefix(first.Error, "stream read:") || last.Status != "ok" {
		t.Fatalf("attempt chain: %+v", r.Attempts)
	}
	// SentMs stays the first emission (sender validity); TTFT counts the lost time.
	if r.SentMs != first.SentMs || first.SentMs < 0 || last.SentMs < first.CompletedMs {
		t.Fatalf("emission times: %+v", r)
	}
	if r.TTFTms < first.CompletedMs-first.SentMs {
		t.Fatalf("TTFT %v hides the failed attempt (%v ms)", r.TTFTms, first.CompletedMs-first.SentMs)
	}
}

func TestClientRetryOnGatewayError(t *testing.T) {
	s, calls := faultyServer(1, func(w http.ResponseWriter) { w.WriteHeader(http.StatusBadGateway) })
	defer s.Close()
	r := runOne(t, s.URL, 1)
	if r.Status != "ok" || calls.Load() != 2 || len(r.Attempts) != 2 || r.Attempts[0].Error != "HTTP 502" {
		t.Fatalf("502 not retried: %+v calls=%d", r, calls.Load())
	}
}

func TestClientRetryIsBounded(t *testing.T) {
	s, calls := faultyServer(100, cutStream)
	defer s.Close()
	r := runOne(t, s.URL, 2)
	if r.Status != "stream_error" || calls.Load() != 3 || len(r.Attempts) != 3 {
		t.Fatalf("expected 3 failed attempts: %+v calls=%d", r, calls.Load())
	}
}

func TestClientRetrySkipsNonFaultErrors(t *testing.T) {
	malformed := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":7,\"finish_reason\":null}}\n\n")
	}
	notFound := func(w http.ResponseWriter) { w.WriteHeader(http.StatusNotFound) }
	for name, fault := range map[string]func(http.ResponseWriter){"malformed": malformed, "404": notFound} {
		s, calls := faultyServer(1, fault)
		r := runOne(t, s.URL, 3)
		s.Close()
		if r.Status == "ok" || calls.Load() != 1 || r.Attempts != nil {
			t.Fatalf("%s must not be retried: %+v calls=%d", name, r, calls.Load())
		}
	}
}

func TestClientRetryBoundsValidated(t *testing.T) {
	for _, n := range []int{-1, 4} {
		c := DefaultConfig("http://127.0.0.1:1/generate", 1)
		c.ClientRetries = n
		if _, err := Run(context.Background(), []Req{{0, 1, 1}}, c); err == nil {
			t.Fatalf("ClientRetries %d accepted", n)
		}
	}
}

func TestAccountRecordsCountsRetriedChains(t *testing.T) {
	cut := AttemptRecord{Status: "stream_error", Error: "stream read: unexpected EOF"}
	records := []Record{
		{Status: "ok"},
		{Status: "http_error", Error: "HTTP 500"},
		{Status: "ok", Attempts: []AttemptRecord{cut, {Status: "ok"}}},
		{Status: "stream_error", Error: cut.Error, Attempts: []AttemptRecord{cut, cut}},
	}
	s, err := AccountRecords(records)
	if err != nil {
		t.Fatal(err)
	}
	if s.Planned != 4 || s.Completed != 1 || s.Failed != 1 || s.Retried != 2 || s.RetriedCompleted != 1 {
		t.Fatalf("summary: %+v", s)
	}
	if s.FailureStatuses["http_error"] != 1 || s.FailureStatuses["stream_error"] != 1 || s.RetriedAttemptStatuses["stream_error"] != 2 {
		t.Fatalf("status counts: %+v", s)
	}
	bad := []Record{{Status: "ok", Attempts: []AttemptRecord{{Status: "ok"}, {Status: "ok"}}}}
	if _, err := AccountRecords(bad); err == nil {
		t.Fatal("retry after success accepted")
	}
}
