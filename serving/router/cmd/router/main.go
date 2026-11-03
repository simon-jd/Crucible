package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"crucible/serving/router"
)

type backendConfig struct{ ID, URL, MetricsURL string }
type config struct {
	Scope    router.Scope
	Backends []backendConfig
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	filename := flag.String("config", "", "JSON scope and explicit backend/adapter URLs")
	listen := flag.String("listen", "127.0.0.1:8080", "listen address; trusted ingress required if exposed")
	retryDial := flag.Bool("retry-dial", false, "resend once to another backend when connecting to the chosen one fails")
	flag.Parse()
	file, err := os.Open(*filename)
	if err != nil {
		return err
	}
	data, err := io.ReadAll(io.LimitReader(file, (1<<20)+1))
	file.Close()
	if err != nil {
		return err
	}
	if len(data) > 1<<20 {
		return errors.New("config exceeds 1MiB")
	}
	var cfg config
	if err = json.Unmarshal(data, &cfg); err != nil {
		return err
	}
	backends := []router.Backend{}
	for _, b := range cfg.Backends {
		if b.MetricsURL == "" {
			return errors.New("each backend requires an explicit metrics adapter URL")
		}
		backends = append(backends, router.Backend{ID: b.ID, URL: b.URL})
	}
	r, err := router.New(router.Config{QueueThreshold: 4, MaxEntries: 1024, MaxPrefixBytes: 4096, FreshFor: 15 * time.Second}, backends)
	if err != nil {
		return err
	}
	transport := &http.Transport{MaxIdleConns: 64, MaxIdleConnsPerHost: 16, IdleConnTimeout: 30 * time.Second, ResponseHeaderTimeout: 10 * time.Second}
	defer transport.CloseIdleConnections()
	handler, err := router.NewProxy(r, router.ProxyConfig{Scope: cfg.Scope, MaxConcurrent: 64, MaxBodyBytes: 1 << 20, RequestTimeout: 5 * time.Minute, Transport: transport,
		RetryDialErrors: *retryDial,
		OnTransportError: func(id string, err error, retried bool) {
			fmt.Fprintf(os.Stderr, "%s eject %s retried=%t: %v\n", time.Now().UTC().Format(time.RFC3339Nano), id, retried, err)
		}})
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	client := &http.Client{Timeout: 2 * time.Second, Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	poll := func() {
		var wg sync.WaitGroup
		for _, b := range cfg.Backends {
			wg.Add(1)
			go func(b backendConfig) {
				defer wg.Done()
				snapshot, err := router.FetchSnapshot(ctx, client, b.MetricsURL)
				if err == nil {
					err = r.Update(b.ID, snapshot)
				}
				if err != nil {
					fmt.Fprintf(os.Stderr, "adapter %s: %v\n", b.ID, err)
				}
			}(b)
		}
		wg.Wait()
	}
	poll()
	pollDone := make(chan struct{})
	go func() {
		defer close(pollDone)
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				poll()
			}
		}
	}()
	server := &http.Server{Addr: *listen, Handler: handler, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, IdleTimeout: 30 * time.Second, BaseContext: func(net.Listener) context.Context { return ctx }}
	listener, listenErr := net.Listen("tcp", *listen)
	if listenErr != nil {
		stop()
		<-pollDone
		return listenErr
	}
	fmt.Println("listening", listener.Addr().String())
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	select {
	case err = <-done:
		stop()
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err = server.Shutdown(shutdown)
		if err != nil {
			_ = server.Close()
		}
		<-done
	}
	stop()
	<-pollDone
	fmt.Fprintf(os.Stderr, "router stats ejections=%d dial_retries=%d\n", r.Ejections(), r.DialRetries())
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
