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
	"sync"
	"testing"
	"time"
)

func streamingServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[11],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":null}}\n\n")
		w.(http.Flusher).Flush()
		time.Sleep(15 * time.Millisecond)
		fmt.Fprint(w, "data: {\"output_ids\":[12,13],\"meta_info\":{\"completion_tokens\":3,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n")
	}))
}

func TestReplayTimingAccuracy(t *testing.T) {
	s := streamingServer()
	defer s.Close()
	trace := make([]Req, 400)
	for i := range trace {
		trace[i] = Req{ArrivalMs: float64(i) * 5, InTokens: 8, OutTokens: 3}
	}
	records := Replay(trace, s.URL, 1)
	late := 0
	for _, r := range records {
		if r.SentMs < 0 || r.SentMs-r.ScheduledMs > 5 {
			late++
		}
		if r.Status != "ok" {
			t.Errorf("request: %+v", r)
		}
	}
	t.Logf("200 rps: late=%d/%d", late, len(records))
	if len(records) != 400 || late*100 > len(records) {
		t.Fatalf("sender-invalid: %d/%d", late, len(records))
	}
}
func TestRecordsTTFTandTPOT(t *testing.T) {
	s := streamingServer()
	defer s.Close()
	r := Replay([]Req{{0, 8, 3}}, s.URL, 1)[0]
	if r.Status != "ok" || r.TTFTms < 0 || r.TPOTms < 5 || r.OutputTokens != 3 {
		t.Fatalf("bad output metrics: %+v", r)
	}
}
func TestInvalidatesLaggingRun(t *testing.T) {
	records := make([]Record, 100)
	for i := range records {
		records[i] = Record{SentMs: 5, Status: "ok"}
	}
	if !SenderValid(records) {
		t.Fatal("exactly 5ms must pass")
	}
	records[0].SentMs = 5.001
	if !SenderValid(records) {
		t.Fatal("exactly 1% must pass")
	}
	records[1].SentMs = -1
	if SenderValid(records) {
		t.Fatal("missing emission plus late request must invalidate")
	}
}

func TestOpenLoopAdmissionAndCancellation(t *testing.T) {
	entered := make(chan struct{}, 8)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		entered <- struct{}{}
		<-r.Context().Done()
	}))
	defer s.Close()
	c := DefaultConfig(s.URL, 1)
	c.MaxInflight = 1
	c.Timeout = 60 * time.Millisecond
	result, e := Run(context.Background(), []Req{{0, 2, 1}, {5, 2, 1}, {10, 2, 1}}, c)
	if e != nil || result.SenderValid || result.Records[1].Status != "admission_rejected" || result.Records[2].Status != "admission_rejected" || result.Records[0].SentMs < 0 {
		t.Fatalf("%+v %v", result, e)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, e = Run(ctx, []Req{{0, 2, 1}, {100, 2, 1}}, c)
	if e != nil || result.SenderValid || result.DurationMs > 50 {
		t.Fatalf("%+v %v", result, e)
	}
	for _, r := range result.Records {
		if r.SentMs != -1 || r.Status != "cancelled" {
			t.Fatal(r)
		}
	}
}
func TestHTTPFailuresAndPayload(t *testing.T) {
	for _, code := range []int{302, 429, 500} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			calls := 0
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				w.Header().Set("Location", "/retry")
				w.WriteHeader(code)
			}))
			defer s.Close()
			r := Replay([]Req{{0, 3, 2}}, s.URL, 1)[0]
			if r.Status != "http_error" || calls != 1 || r.SentMs < 0 {
				t.Fatal(r, calls)
			}
		})
	}
	var body struct {
		IDs    []int          `json:"input_ids"`
		Stream bool           `json:"stream"`
		Params map[string]any `json:"sampling_params"`
	}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[9,10],\"meta_info\":{\"completion_tokens\":2,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n")
	}))
	defer s.Close()
	r := Replay([]Req{{0, 17, 2}}, s.URL, 1)[0]
	if r.Status != "ok" || !r.Coalesced || len(body.IDs) != 17 || !body.Stream || body.Params["ignore_eos"] != true || body.Params["max_new_tokens"] != float64(2) {
		t.Fatal(r, body)
	}
}
func TestInputBounds(t *testing.T) {
	c := DefaultConfig("http://127.0.0.1:1", 1)
	for _, tr := range [][]Req{nil, {{math.NaN(), 1, 1}}, {{-1, 1, 1}}, {{2, 1, 1}, {1, 1, 1}}, {{0, -1, 1}}, {{0, 1, -1}}, {{1e300, 1, 1}}} {
		if _, e := Run(context.Background(), tr, c); e == nil {
			t.Fatal("accepted", tr)
		}
	}
	for _, mult := range []float64{0, -1, math.NaN(), math.Inf(1), 1e-300} {
		c.Multiplier = mult
		if _, e := Run(context.Background(), []Req{{1, 1, 1}}, c); e == nil {
			t.Fatal("accepted", mult)
		}
	}
}

type delayedWriteConn struct {
	net.Conn
	once sync.Once
}

func (c *delayedWriteConn) Write(b []byte) (int, error) {
	c.once.Do(func() { time.Sleep(20 * time.Millisecond) })
	return c.Conn.Write(b)
}
func TestTransportEmissionIncludesWriteWait(t *testing.T) {
	s := streamingServer()
	defer s.Close()
	transport := &http.Transport{DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, e := (&net.Dialer{}).DialContext(ctx, network, address)
		if e != nil {
			return nil, e
		}
		return &emissionConn{Conn: &delayedWriteConn{Conn: conn}}, nil
	}}
	defer transport.CloseIdleConnections()
	start := time.Now()
	r := emptyRecord(0)
	send(context.Background(), &http.Client{Transport: transport}, s.URL, []byte(`{"input_ids":[1],"stream":true}`), start.Add(time.Second), start, &r, 3)
	if r.Status != "ok" || r.SentMs < 20 || r.TTFTms < 0 || r.TTFTms >= r.CompletedMs-r.SentMs || SenderValid([]Record{r}) {
		t.Fatalf("write wait not measured: %+v", r)
	}
}
func TestNeverEmittedCountsLate(t *testing.T) {
	transport := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) { <-ctx.Done(); return nil, ctx.Err() }}
	defer transport.CloseIdleConnections()
	start := time.Now()
	r := emptyRecord(0)
	send(context.Background(), &http.Client{Transport: transport}, "http://127.0.0.1:1", []byte(`{}`), start.Add(10*time.Millisecond), start, &r, 1)
	if r.SentMs != -1 || r.Status != "transport_error" || SenderValid([]Record{r}) {
		t.Fatal(r)
	}
}
