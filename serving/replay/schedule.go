package replay

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/http/httptrace"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func Run(ctx context.Context, trace []Req, c Config) (Result, error) {
	return run(ctx, trace, c, nil)
}

// RunWithDiagnostics shares the original scheduler; capture is opt-in and
// independent of SenderValid. The recorder must be fresh for this run.
func RunWithDiagnostics(ctx context.Context, trace []Req, c Config, diagnostics *Diagnostics) (Result, error) {
	return run(ctx, trace, c, diagnostics)
}

func run(ctx context.Context, trace []Req, c Config, diagnostics *Diagnostics) (Result, error) {
	bodies, err := prepare(trace, c)
	if err != nil {
		return Result{}, err
	}
	parser := parseStream
	if c.API == "openai" || c.API == "openai-text" {
		parser = parseOpenAIStream
	}
	transport := newTransport(c)
	defer transport.CloseIdleConnections()
	var rt http.RoundTripper = transport
	if c.Tenant != "" {
		rt = tenantTransport{transport, c.Tenant}
	}
	client := &http.Client{Transport: rt, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	result := Result{Records: make([]Record, len(trace))}
	slots := make(chan struct{}, c.MaxInflight)
	var wg sync.WaitGroup
	start := time.Now()
	if err := diagnostics.begin(start, len(trace)); err != nil {
		return Result{}, err
	}
	for i, r := range trace {
		phase := diagnostics.request(i)
		scheduled := r.ArrivalMs / c.Multiplier
		result.Records[i] = emptyRecord(scheduled)
		due := start.Add(time.Duration(scheduled * float64(time.Millisecond)))
		waitUntil(ctx, due, phase)
		if ctx.Err() != nil {
			if phase != nil {
				phase.at("cancelled", time.Now())
			}
			result.Records[i].Status = "cancelled"
			result.Records[i].Error = ctx.Err().Error()
			result.Records[i].CompletedMs = ms(time.Since(start))
			continue
		}
		select {
		case slots <- struct{}{}:
			wg.Add(1)
			go func(i int, due time.Time) {
				if phase != nil {
					phase.at("send_entry", time.Now())
				}
				defer wg.Done()
				defer func() { <-slots }()
				sendWithRetries(ctx, client, c.URL, bodies[i], due.Add(c.Timeout), start, &result.Records[i], expectedTokens(c, trace[i]), phase, parser, c.ClientRetries)
			}(i, due)
		default:
			if phase != nil {
				phase.at("admission_rejected", time.Now())
			}
			result.Records[i].Status = "admission_rejected"
			result.Records[i].Error = "inflight bound reached"
			result.Records[i].CompletedMs = ms(time.Since(start))
		}
	}
	wg.Wait()
	result.DurationMs = ms(time.Since(start))
	result.SenderValid = SenderValid(result.Records)
	result.Aborted = ctx.Err() != nil
	return result, nil
}
func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
func send(parent context.Context, client *http.Client, url string, body []byte, deadline, start time.Time, r *Record, expected int) {
	sendWithRetries(parent, client, url, body, deadline, start, r, expected, nil, parseStream, 0)
}

type streamParser func(io.Reader, time.Time, float64, *Record, int) error

// retryable is a fault on the path to the server, not a wrong answer: a stream
// cut mid-response, a gateway with no live backend, or a transport error.
func retryable(r Record) bool {
	switch r.Status {
	case "stream_error":
		return strings.HasPrefix(r.Error, "stream read:")
	case "http_error":
		return r.Error == "HTTP 502" || r.Error == "HTTP 503"
	case "transport_error":
		return r.Error != "missing emission timestamp"
	}
	return false
}

// sendWithRetries sends once, then resends a retryable failure up to retries
// times inside the same deadline. Diagnostics cover the first attempt only.
func sendWithRetries(parent context.Context, client *http.Client, url string, body []byte, deadline, start time.Time, r *Record, expected int, phase *phaseRequest, parser streamParser, retries int) {
	ctx, cancel := context.WithDeadline(parent, deadline)
	defer cancel()
	sendWithDiagnostics(ctx, client, url, body, start, r, expected, phase, parser, -1)
	first := r.SentMs
	var attempts []AttemptRecord
	for n := 0; n < retries && first >= 0 && ctx.Err() == nil && retryable(*r); n++ {
		attempts = append(attempts, AttemptRecord{r.SentMs, r.CompletedMs, r.Status, r.Error})
		*r = emptyRecord(r.ScheduledMs)
		sendWithDiagnostics(ctx, client, url, body, start, r, expected, nil, parser, first)
	}
	if attempts != nil {
		r.Attempts = append(attempts, AttemptRecord{r.SentMs, r.CompletedMs, r.Status, r.Error})
		r.SentMs = first
	}
}

// sendWithDiagnostics runs one attempt under ctx. origin >= 0 (a resend) is the
// first attempt's emission; TTFT is measured from it instead of this attempt's.
func sendWithDiagnostics(ctx context.Context, client *http.Client, url string, body []byte, start time.Time, r *Record, expected int, phase *phaseRequest, parser streamParser, origin float64) {
	observer := newEmission(start, len(body))
	observer.diagnostic = phase
	var owned atomic.Pointer[emissionConn]
	trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) {
		var observed time.Time
		if phase != nil {
			observed = time.Now()
		}
		conn, ok := info.Conn.(*emissionConn)
		if !ok {
			observer.fail()
			return
		}
		if phase != nil {
			phase.capture(phaseEvent{Phase: "got_conn", AtNS: observed.Sub(start).Nanoseconds(), Connection: phase.parent.connectionID(conn), Reused: info.Reused, WasIdle: info.WasIdle, IdleNS: info.IdleTime.Nanoseconds()})
		}
		if !owned.CompareAndSwap(nil, conn) {
			observer.fail()
			return
		}
		conn.owner.Store(observer)
	}}
	if phase != nil {
		trace.GetConn = func(hostPort string) {
			phase.capture(phaseEvent{Phase: "get_conn", AtNS: time.Since(start).Nanoseconds(), Address: shortPhaseString(hostPort)})
		}
		trace.ConnectStart = func(network, addr string) {
			phase.capture(phaseEvent{Phase: "connect_start", AtNS: time.Since(start).Nanoseconds(), Network: shortPhaseString(network), Address: shortPhaseString(addr)})
		}
		trace.ConnectDone = func(network, addr string, err error) {
			phase.capture(phaseEvent{Phase: "connect_done", AtNS: time.Since(start).Nanoseconds(), Network: shortPhaseString(network), Address: shortPhaseString(addr), err: err})
		}
	}
	ctx = httptrace.WithClientTrace(ctx, trace)
	defer func() {
		if conn := owned.Load(); conn != nil {
			conn.release(observer)
		}
		r.SentMs = observer.snapshot()
		r.CompletedMs = ms(time.Since(start))
	}()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		r.Status = "request_error"
		r.Error = err.Error()
		return
	}
	// Prevent transport retry even if a peer supplies idempotency headers.
	req.GetBody = nil
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		r.Status = "transport_error"
		r.Error = err.Error()
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		r.Status = "http_error"
		r.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return
	}
	mediaType, _, mediaErr := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mediaErr != nil || mediaType != "text/event-stream" {
		r.Status = "stream_error"
		r.Error = "expected text/event-stream"
		return
	}
	emitted := observer.wait(ctx)
	if emitted < 0 {
		r.Status = "transport_error"
		r.Error = "missing emission timestamp"
		return
	}
	if origin >= 0 {
		emitted = origin
	}
	err = parser(resp.Body, start, emitted, r, expected)
	if err != nil {
		r.Status = "stream_error"
		r.Error = err.Error()
		return
	}
	r.Status = "ok"
}

// tenantTransport adds X-Tenant without changing the observed connection path.
type tenantTransport struct {
	base   http.RoundTripper
	tenant string
}

func (t tenantTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set("X-Tenant", t.tenant)
	return t.base.RoundTrip(req)
}
