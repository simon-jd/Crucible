package replay

import (
	"bytes"
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// emission observes only this client's Content-Length framed, HTTP/1 POST.
// It never treats WroteRequest as proof that net/http's buffer was flushed.
type emission struct {
	mu                 sync.Mutex
	start              time.Time
	expected, accepted int
	header             []byte
	headerDone, failed bool
	sent               float64
	diagnostic         *phaseRequest
	// settled closes once the outcome is known: the full request returned from
	// Write (sent >= 0) or the observation failed.
	settled    chan struct{}
	settleOnce sync.Once
}

func newEmission(start time.Time, expected int) *emission {
	return &emission{start: start, expected: expected, sent: -1, settled: make(chan struct{})}
}
func (e *emission) fail() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.failed = true
	e.sent = -1
	e.settle()
}

// settle is called with mu held.
func (e *emission) settle() {
	if e.failed || e.sent >= 0 {
		e.settleOnce.Do(func() { close(e.settled) })
	}
}
func (e *emission) snapshot() float64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.failed {
		return -1
	}
	return e.sent
}

// wait returns the emission time once the request's last Write has returned.
// net/http's writeLoop may be descheduled between the socket write and that
// return while the response already reached client.Do, so the caller must
// not read snapshot() alone. An unfinished emission at ctx's end stays -1.
func (e *emission) wait(ctx context.Context) float64 {
	select {
	case <-e.settled:
	case <-ctx.Done():
	}
	return e.snapshot()
}
func (e *emission) observe(p []byte, n int, err error, finished time.Time) {
	e.mu.Lock()
	defer e.mu.Unlock()
	defer e.settle()
	if e.failed {
		return
	}
	if err != nil || n != len(p) || n < 0 {
		e.failed = true
		e.sent = -1
		return
	}
	if e.sent >= 0 {
		e.failed = true
		e.sent = -1
		return
	}
	for !e.headerDone && len(p) > 0 {
		if len(e.header) >= 64<<10 {
			e.failed = true
			return
		}
		e.header = append(e.header, p[0])
		p = p[1:]
		if bytes.HasSuffix(e.header, []byte("\r\n\r\n")) {
			lines := strings.Split(string(e.header), "\r\n")
			if !strings.HasPrefix(lines[0], "POST ") || !strings.HasSuffix(lines[0], " HTTP/1.1") {
				e.failed = true
				return
			}
			count := 0
			for _, line := range lines[1:] {
				if line == "" {
					continue
				}
				key, value, ok := strings.Cut(line, ":")
				if !ok {
					e.failed = true
					return
				}
				if strings.EqualFold(key, "Transfer-Encoding") {
					e.failed = true
					return
				}
				if strings.EqualFold(key, "Content-Length") {
					length, err := strconv.Atoi(strings.TrimSpace(value))
					if err != nil || length != e.expected || length <= 0 {
						e.failed = true
						return
					}
					count++
				}
			}
			if count != 1 {
				e.failed = true
				return
			}
			e.headerDone = true
			e.header = nil
		}
	}
	if e.headerDone {
		e.accepted += len(p)
		if e.accepted > e.expected {
			e.failed = true
			return
		}
		if e.accepted == e.expected {
			e.sent = ms(finished.Sub(e.start))
		}
	}
}

type emissionConn struct {
	net.Conn
	owner        atomic.Pointer[emission]
	diagnosticID atomic.Uint64
}

func (c *emissionConn) Write(p []byte) (int, error) {
	owner := c.owner.Load()
	var entered time.Time
	if owner != nil && owner.diagnostic != nil {
		entered = time.Now()
	}
	n, err := c.Conn.Write(p)
	finished := time.Now()
	if owner != nil {
		if owner.diagnostic != nil {
			// Store the pair after the actual Write returns; recorder locking is
			// excluded from its measured interval. An unfinished pair is missing,
			// never fabricated, if Run ends before this return is observed.
			writeSequence := owner.diagnostic.writeEntry(c.diagnosticID.Load(), len(p), entered)
			owner.diagnostic.writeReturn(c.diagnosticID.Load(), writeSequence, len(p), n, err, finished)
		}
		owner.observe(p, n, err, finished)
	}
	return n, err
}
func (c *emissionConn) release(owner *emission) { c.owner.CompareAndSwap(owner, nil) }
func newTransport(c Config) *http.Transport {
	dialer := &net.Dialer{Timeout: c.Timeout}
	return &http.Transport{
		Proxy: nil, ForceAttemptHTTP2: false, TLSNextProto: map[string]func(string, *tls.Conn) http.RoundTripper{},
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			conn, err := dialer.DialContext(ctx, network, address)
			if err != nil {
				return nil, err
			}
			return &emissionConn{Conn: conn}, nil
		},
		DialTLSContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			d := tls.Dialer{NetDialer: dialer, Config: &tls.Config{MinVersion: tls.VersionTLS12, NextProtos: []string{"http/1.1"}}}
			conn, err := d.DialContext(ctx, network, address)
			if err != nil {
				return nil, err
			}
			return &emissionConn{Conn: conn}, nil
		},
		MaxConnsPerHost: c.MaxInflight, MaxIdleConnsPerHost: c.MaxInflight, MaxIdleConns: c.MaxInflight, IdleConnTimeout: time.Second, DisableCompression: true,
	}
}
