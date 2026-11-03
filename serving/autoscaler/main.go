package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	simulate := flag.Bool("simulate-burst", false, "run matched synthetic clock 1x/4x policy experiment; no network")
	rounds := flag.Int("rounds", 3, "synthetic matched rounds")
	metrics := flag.String("metrics-url", "", "explicit aggregate metrics adapter URL")
	apply := flag.Bool("apply", false, "opt in to conditional Kubernetes scale writes")
	api := flag.String("api-server", "", "explicit Kubernetes API server; no context discovery")
	ns := flag.String("namespace", "", "target namespace")
	name := flag.String("deployment", "", "target deployment (or statefulset with -resource)")
	resource := flag.String("resource", "deployments", "scale target resource: deployments or statefulsets")
	token := flag.String("token-file", "", "service account bearer token path")
	ca := flag.String("ca-file", "", "PEM CA file")
	once := flag.Bool("once", false, "one bounded poll and exit")
	interval := flag.Duration("interval", 5*time.Second, "poll interval")
	minN := flag.Int("min", 1, "minimum desired replicas")
	maxN := flag.Int("max", 8, "maximum desired replicas")
	qHigh := flag.Float64("queue-high", 4, "scale up waiting requests per replica")
	qLow := flag.Float64("queue-low", 1, "scale down waiting requests per replica")
	tHigh := flag.Float64("ttft-high", 0.8, "scale up p95 seconds")
	tLow := flag.Float64("ttft-low", 0.2, "scale down p95 seconds")
	flag.Parse()
	if *simulate {
		result, err := SimulateBurst(*rounds)
		if err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(result)
	}
	if *metrics == "" || *interval <= 0 {
		return fmt.Errorf("metrics-url and positive interval required")
	}
	p, err := NewPolicy(Config{Min: *minN, Max: *maxN, QueueLow: *qLow, QueueHigh: *qHigh, TTFTLow: *tLow, TTFTHigh: *tHigh, FreshFor: 15 * time.Second}, nil)
	if err != nil {
		return err
	}
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12}
	if *ca != "" {
		pem, err := os.ReadFile(*ca)
		if err != nil {
			return err
		}
		pool, err := x509.SystemCertPool()
		if err != nil {
			pool = x509.NewCertPool()
		}
		if !pool.AppendCertsFromPEM(pem) {
			return fmt.Errorf("invalid CA")
		}
		tlsConfig.RootCAs = pool
	}
	transport := &http.Transport{TLSClientConfig: tlsConfig, ResponseHeaderTimeout: 5 * time.Second, MaxIdleConns: 4, IdleConnTimeout: 30 * time.Second}
	defer transport.CloseIdleConnections()
	client := &http.Client{Timeout: 10 * time.Second, Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	c := Controller{Policy: p, HTTP: client, MetricsURL: *metrics, Apply: *apply}
	if *apply {
		if *token == "" {
			return fmt.Errorf("apply requires explicit token-file")
		}
		b, err := os.ReadFile(*token)
		if err != nil {
			return err
		}
		c.Scale = &ScaleClient{HTTP: client, BaseURL: *api, Namespace: *ns, Name: *name, Token: strings.TrimSpace(string(b)), Resource: *resource}
		if c.Scale.Token == "" {
			return fmt.Errorf("empty token")
		}
		if _, err = c.Scale.endpoint(); err != nil {
			return err
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()
	encoder := json.NewEncoder(os.Stdout)
	for {
		d, err := c.Tick(ctx)
		if err != nil {
			if *once {
				return err
			}
			fmt.Fprintln(os.Stderr, err)
		} else if err = encoder.Encode(d); err != nil {
			return err
		}
		if *once {
			return nil
		}
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
		}
	}
}
