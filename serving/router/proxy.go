package router

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"
)

type ProxyConfig struct {
	Scope          Scope // Tenant is supplied by a trusted ingress using X-Tenant.
	MaxConcurrent  int
	MaxBodyBytes   int64
	RequestTimeout time.Duration
	Transport      http.RoundTripper
	// RetryDialErrors resends a request once, to another backend, when connecting
	// to the chosen one failed. Then the backend never received the POST, so the
	// resend cannot run it twice. Any other transport error is never retried.
	RetryDialErrors bool
	// OnTransportError, if set, observes every backend transport failure that
	// ejected a backend (not client cancellations), and whether it was resent.
	OnTransportError func(backend string, err error, retried bool)
}

func isDialError(err error) bool {
	var op *net.OpError
	return errors.As(err, &op) && op.Op == "dial"
}

// NewProxy accepts the common SGLang/OpenAI string-prompt JSON shape. Chat/token
// arrays require a tokenizer-aware future adapter and are rejected, not guessed.
func NewProxy(r *Router, c ProxyConfig) (http.Handler, error) {
	scope := c.Scope
	scope.Tenant = "validation"
	if !validScope(scope) || c.MaxConcurrent < 1 || c.MaxConcurrent > 1024 || c.MaxBodyBytes < 1 || c.MaxBodyBytes > 16<<20 || c.RequestTimeout <= 0 {
		return nil, errors.New("invalid proxy bounds")
	}
	if c.Transport == nil {
		c.Transport = &http.Transport{MaxIdleConns: 64, MaxIdleConnsPerHost: 16, IdleConnTimeout: 30 * time.Second, ResponseHeaderTimeout: 10 * time.Second}
	}
	admission := make(chan struct{}, c.MaxConcurrent)
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method != "POST" || (req.URL.Path != "/generate" && req.URL.Path != "/v1/completions") {
			http.Error(w, "unsupported route", http.StatusNotFound)
			return
		}
		// The same budget covers request-body reads, upstream work, and writes
		// to a downstream peer that stays connected but stops consuming data.
		ctx, cancel := context.WithTimeout(req.Context(), c.RequestTimeout)
		defer cancel()
		req = req.WithContext(ctx)
		deadline, _ := ctx.Deadline()
		control := http.NewResponseController(w)
		// A wrapper must preserve these capabilities (typically via Unwrap).
		// Fail closed before body reads/admission/leases if a bound is unavailable.
		if err := control.SetReadDeadline(deadline); err != nil {
			panic(http.ErrAbortHandler)
		}
		if err := control.SetWriteDeadline(deadline); err != nil {
			panic(http.ErrAbortHandler)
		}
		interrupted := make(chan struct{})
		stopInterrupt := context.AfterFunc(ctx, func() {
			defer close(interrupted)
			// Cancellation may precede the timeout, e.g. during server shutdown.
			// Socket deadlines interrupt reads/writes that context alone cannot.
			_ = control.SetReadDeadline(time.Now())
			_ = control.SetWriteDeadline(time.Now())
		})
		defer func() {
			if !stopInterrupt() {
				<-interrupted // No callback may touch a later keepalive request.
			}
			if ctx.Err() != nil || !time.Now().Before(deadline) {
				panic(http.ErrAbortHandler) // Incomplete exchanges cannot be reused.
			}
			// Retain both deadlines through net/http's post-handler body drain
			// and response flush. net/http resets them before keepalive reuse;
			// clearing here would make that final I/O unbounded.
		}()
		select {
		case admission <- struct{}{}:
			defer func() { <-admission }()
		default:
			http.Error(w, "router full", http.StatusServiceUnavailable)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, req.Body, c.MaxBodyBytes))
		req.Body.Close()
		if err != nil {
			http.Error(w, "request too large or unreadable", http.StatusRequestEntityTooLarge)
			return
		}
		var input struct {
			Model  string  `json:"model"`
			Prompt *string `json:"prompt"`
			Text   *string `json:"text"`
		}
		if json.Unmarshal(body, &input) != nil {
			http.Error(w, "invalid JSON", 400)
			return
		}
		prompt := input.Prompt
		if prompt == nil {
			prompt = input.Text
		}
		if (input.Prompt != nil && input.Text != nil) || prompt == nil || (input.Model != "" && input.Model != c.Scope.Model) {
			http.Error(w, "string prompt and configured model required", 400)
			return
		}
		scope := c.Scope
		scope.Tenant = req.Header.Get("X-Tenant")
		if !validScope(scope) {
			http.Error(w, "trusted tenant required", 400)
			return
		}
		var forward func(attempt int)
		forward = func(attempt int) {
			lease, err := r.Acquire(scope, *prompt)
			if err != nil {
				http.Error(w, err.Error(), 503)
				return
			}
			defer lease.Release()
			req.Body = io.NopCloser(bytes.NewReader(body))
			req.ContentLength = int64(len(body))
			req.GetBody = nil
			target, _ := url.Parse(lease.Backend.URL)
			proxy := httputil.NewSingleHostReverseProxy(target)
			director := proxy.Director
			proxy.Director = func(out *http.Request) {
				director(out)
				out.Host = target.Host
				out.Header.Del("X-Tenant")
				out.Header.Del("X-Cache-Hit")
				out.Header.Del("X-Backend-URL")
			}
			proxy.Transport = c.Transport
			proxy.FlushInterval = -1
			proxy.ErrorHandler = func(w http.ResponseWriter, out *http.Request, err error) {
				// A cancelled or timed-out client says nothing about the backend.
				if out.Context().Err() == nil {
					r.Eject(lease.Backend.ID)
					retry := c.RetryDialErrors && attempt == 0 && isDialError(err)
					if c.OnTransportError != nil {
						c.OnTransportError(lease.Backend.ID, err, retry)
					}
					if retry {
						r.countDialRetry()
						lease.Release()
						forward(1)
						return
					}
				}
				http.Error(w, "backend unavailable", http.StatusBadGateway)
			}
			proxy.ServeHTTP(w, req) // release after stream completion/error.
		}
		forward(0)
	}), nil
}

// FetchSnapshot is an explicit adapter contract, not a native SGLang endpoint.
func FetchSnapshot(ctx context.Context, client *http.Client, endpoint string) (Snapshot, error) {
	var s Snapshot
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return s, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return s, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return s, errors.New("metrics adapter status not 200")
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20+1))
	if err != nil {
		return s, err
	}
	if len(body) > 2<<20 {
		return s, errors.New("metrics snapshot exceeds 2MiB")
	}
	var fields map[string]json.RawMessage
	if err = json.Unmarshal(body, &fields); err != nil {
		return s, err
	}
	queue, ok := fields["Queue"]
	if !ok || string(queue) == "null" {
		return s, errors.New("missing Queue metric")
	}
	err = json.Unmarshal(body, &s)
	return s, err
}
