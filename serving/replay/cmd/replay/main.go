package main

import (
	"context"
	"crucible/serving/replay"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("replay", flag.ContinueOnError)
	tracePath := flags.String("trace", "", "bounded source CSV")
	format := flags.String("format", "", "azure or burstgpt")
	endpoint := flags.String("url", "", "native /generate or /v1/completions URL")
	mult := flags.Float64("mult", 1, "arrival-rate multiplier")
	idsPath := flags.String("token-ids", "", "verified tokenizer allowlist JSON")
	timeout := flags.Duration("timeout", 30*time.Second, "absolute request lifetime from schedule")
	inflight := flags.Int("inflight", 64, "hard admission bound")
	seed := flags.Uint64("seed", 1, "synthetic input seed")
	phasePath := flags.String("phase-sidecar", "", "optional fresh bounded diagnostic sidecar")
	api := flags.String("api", "sglang", "sglang (native /generate), openai (vLLM /v1/completions, token-ID prompt) or openai-text (string prompt from token_texts)")
	model := flags.String("model", "", "served model name, required for -api openai")
	account := flags.Bool("account", false, "add request accounting (completed/failed/retried) to the output")
	tenant := flags.String("tenant", "", "optional X-Tenant header (trusted client of the Task 14 router)")
	clientRetries := flags.Int("client-retries", 0, "resend a cut stream, HTTP 502/503 or transport error up to N (0-3) times; resends are accounted as retried")
	if e := flags.Parse(args); e != nil {
		return e
	}
	f, e := os.Open(*tracePath)
	if e != nil {
		return e
	}
	trace, info, e := replay.ParseTrace(f, *format)
	f.Close()
	if e != nil {
		return e
	}
	tokenFile, e := os.Open(*idsPath)
	if e != nil {
		return e
	}
	data, e := io.ReadAll(io.LimitReader(tokenFile, (1<<20)+1))
	tokenFile.Close()
	if e != nil {
		return e
	}
	if len(data) > 1<<20 {
		return fmt.Errorf("allowlist too large")
	}
	var tokens struct {
		TokenIDs                         []int    `json:"token_ids"`
		TokenTexts                       []string `json:"token_texts"`
		Model, Revision, TokenizerSHA256 string
		FileSHA256                       map[string]string `json:"file_sha256"`
	}
	if e = json.Unmarshal(data, &tokens); e != nil {
		return e
	}
	_, revErr := hex.DecodeString(tokens.Revision)
	_, hashErr := hex.DecodeString(tokens.TokenizerSHA256)
	if revErr != nil || hashErr != nil || tokens.Model == "" || len(tokens.Revision) != 40 || len(tokens.TokenizerSHA256) != 64 {
		return fmt.Errorf("allowlist requires pinned model/revision/tokenizer hash")
	}
	cfg := replay.DefaultConfig(*endpoint, *mult)
	cfg.TokenIDs = tokens.TokenIDs
	cfg.TokenTexts = tokens.TokenTexts
	cfg.Timeout = *timeout
	cfg.MaxInflight = *inflight
	cfg.Seed = *seed
	cfg.API = *api
	cfg.Model = *model
	cfg.Tenant = *tenant
	cfg.ClientRetries = *clientRetries
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var diagnostics *replay.Diagnostics
	if *phasePath != "" {
		diagnostics, e = replay.NewDiagnostics(len(trace))
		if e != nil {
			return e
		}
	}
	result, e := replay.RunWithDiagnostics(ctx, trace, cfg, diagnostics)
	if e != nil {
		return e
	}
	// Output names the allowlist by file hash and count instead of copying IDs.
	provenance := struct {
		Model, Revision, TokenizerSHA256 string
		FileSHA256                       map[string]string `json:"file_sha256"`
		AllowlistSHA256                  string
		TokenIDCount                     int
	}{tokens.Model, tokens.Revision, tokens.TokenizerSHA256, tokens.FileSHA256, fmt.Sprintf("%x", sha256.Sum256(data)), len(tokens.TokenIDs)}
	var accounting *replay.RecordSummary
	if *account {
		summary, accErr := replay.AccountRecords(result.Records)
		if accErr != nil {
			return accErr
		}
		accounting = &summary
	}
	e = json.NewEncoder(output).Encode(struct {
		Trace           replay.TraceInfo
		Config          replay.Config
		TokenProvenance any
		Run             replay.Result
		Accounting      *replay.RecordSummary `json:",omitempty"`
	}{info, cfg, provenance, result, accounting})
	if e != nil {
		return e
	}
	var runError error
	if ctx.Err() != nil {
		runError = ctx.Err()
	} else if !result.SenderValid {
		runError = fmt.Errorf("sender-invalid run; no accepted performance")
	} else {
		for _, r := range result.Records {
			if r.Status != "ok" {
				runError = fmt.Errorf("run contains failed requests")
				break
			}
		}
	}
	if diagnostics != nil {
		return errors.Join(runError, diagnostics.WriteSidecar(*phasePath))
	}
	return runError
}
