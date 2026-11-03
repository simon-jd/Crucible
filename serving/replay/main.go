// Package replay implements bounded open-loop native SGLang trace replay.
package replay

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"strings"
	"time"
)

type Req struct {
	ArrivalMs           float64
	InTokens, OutTokens int
}
type Config struct {
	URL         string
	Multiplier  float64
	MaxInflight int
	Timeout     time.Duration
	TokenIDs    []int `json:"-"` // identified in output by the allowlist hash
	// TokenTexts, for API "openai-text", is parallel to TokenIDs: each ID's
	// single-token text. The prompt is their concatenation (a string), so a
	// string-only router can forward it.
	TokenTexts []string `json:"-"`
	Seed       uint64
	// API selects the wire format: "" or "sglang" is native SGLang /generate;
	// "openai" is vLLM's /v1/completions, which also needs Model;
	// "openai-text" is the same with a string prompt built from TokenTexts.
	API, Model string
	// Tenant, if set, is sent as X-Tenant (the Task 14 router's trusted tenant).
	Tenant string `json:",omitempty"`
	// ClientRetries (0-3) resends a request whose stream was cut, or that got
	// HTTP 502/503 or a transport error, within its original deadline. Each
	// resend is re-executed work and is accounted as retried. 0 never resends.
	ClientRetries int `json:",omitempty"`
}

func DefaultConfig(url string, mult float64) Config {
	// IDs are fixture defaults only. Real runs must supply tokenizer-verified IDs.
	return Config{URL: url, Multiplier: mult, MaxInflight: 64, Timeout: 30 * time.Second, TokenIDs: []int{100, 101, 102, 103}, Seed: 1}
}

// expectedTokens is the exact output length requested from the server. vLLM
// rejects max_tokens=0, so OpenAI mode asks for one token on zero-output rows.
func expectedTokens(c Config, r Req) int {
	if (c.API == "openai" || c.API == "openai-text") && r.OutTokens == 0 {
		return 1
	}
	return r.OutTokens
}

func requestBody(c Config, ids []int, r Req) ([]byte, error) {
	var prompt any = ids
	switch c.API {
	case "", "sglang":
		return json.Marshal(struct {
			InputIDs []int          `json:"input_ids"`
			Stream   bool           `json:"stream"`
			Sampling map[string]any `json:"sampling_params"`
		}{ids, true, map[string]any{"max_new_tokens": r.OutTokens, "ignore_eos": true, "temperature": 0}})
	case "openai", "openai-text":
		if c.API == "openai-text" {
			text, err := promptText(c, ids)
			if err != nil {
				return nil, err
			}
			prompt = text
		}
		if c.Model == "" {
			return nil, fmt.Errorf("openai API requires a model name")
		}
		n := expectedTokens(c, r)
		return json.Marshal(map[string]any{
			"model": c.Model, "prompt": prompt, "max_tokens": n, "min_tokens": n, "ignore_eos": true,
			"temperature": 0, "stream": true, "return_token_ids": true,
			"stream_options": map[string]any{"include_usage": true},
		})
	}
	return nil, fmt.Errorf("unknown API %q", c.API)
}
func promptText(c Config, ids []int) (string, error) {
	if len(c.TokenTexts) != len(c.TokenIDs) {
		return "", fmt.Errorf("token texts must be parallel to token IDs")
	}
	text := map[int]string{}
	for i, id := range c.TokenIDs {
		if c.TokenTexts[i] == "" {
			return "", fmt.Errorf("empty token text")
		}
		text[id] = c.TokenTexts[i]
	}
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(text[id])
	}
	return b.String(), nil
}

func Replay(trace []Req, url string, mult float64) []Record {
	run, err := Run(context.Background(), trace, DefaultConfig(url, mult))
	if err != nil {
		if len(trace) > MaxRequests {
			r := emptyRecord(-1)
			r.Status = "invalid_input"
			r.Error = err.Error()
			return []Record{r}
		}
		r := make([]Record, len(trace))
		for i := range r {
			r[i] = emptyRecord(trace[i].ArrivalMs)
			r[i].Status = "invalid_input"
			r[i].Error = err.Error()
		}
		return r
	}
	return run.Records
}

const MaxRequests = 10000
const MaxTokens = 32768
const MaxPayloadBytes = 16 << 20

func prepare(trace []Req, c Config) ([][]byte, error) {
	u, e := url.Parse(c.URL)
	if e != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("invalid endpoint")
	}
	if len(trace) == 0 || len(trace) > MaxRequests || math.IsNaN(c.Multiplier) || math.IsInf(c.Multiplier, 0) || c.Multiplier <= 0 || c.MaxInflight < 1 || c.MaxInflight > 256 || c.ClientRetries < 0 || c.ClientRetries > 3 || c.Timeout <= 0 || c.Timeout > 10*time.Minute || len(c.TokenIDs) == 0 || len(c.TokenIDs) > 65536 {
		return nil, fmt.Errorf("invalid replay bounds")
	}
	for _, id := range c.TokenIDs {
		if id < 0 || id > 10000000 {
			return nil, fmt.Errorf("invalid token ID")
		}
	}
	bodies := make([][]byte, len(trace))
	total := 0
	state := c.Seed
	for i, r := range trace {
		if math.IsNaN(r.ArrivalMs) || math.IsInf(r.ArrivalMs, 0) || r.ArrivalMs < 0 || (i > 0 && r.ArrivalMs < trace[i-1].ArrivalMs) || r.ArrivalMs/c.Multiplier > float64(time.Hour/time.Millisecond) || r.InTokens < 0 || r.InTokens > MaxTokens || r.OutTokens < 0 || r.OutTokens > MaxTokens || r.InTokens+r.OutTokens > MaxTokens {
			return nil, fmt.Errorf("invalid request %d", i)
		}
		ids := make([]int, r.InTokens)
		for j := range ids {
			state = state*6364136223846793005 + 1442695040888963407
			ids[j] = c.TokenIDs[(state>>32)%uint64(len(c.TokenIDs))]
		}
		bodies[i], e = requestBody(c, ids, r)
		if e != nil {
			return nil, e
		}
		total += len(bodies[i])
		if total > MaxPayloadBytes {
			return nil, fmt.Errorf("payload budget exceeded")
		}
	}
	return bodies, nil
}
