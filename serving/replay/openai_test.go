package replay

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// vLLM 0.10.2 /v1/completions with stream, return_token_ids and include_usage:
// one chunk per engine step with delta token_ids, finish_reason on the last
// token chunk, then a choices=[] usage chunk, then [DONE].
func openAIChunk(ids string, finish string) string {
	reason := "null"
	if finish != "" {
		reason = `"` + finish + `"`
	}
	return `data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":"x","logprobs":null,"finish_reason":` + reason + `,"stop_reason":null,"prompt_token_ids":null,"token_ids":[` + ids + `]}],"usage":null}` + "\n\n"
}

func openAIUsage(n int) string {
	return fmt.Sprintf(`data: {"id":"cmpl-1","object":"text_completion","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":%d,"total_tokens":%d}}`+"\n\n", n, n+8)
}

func TestOpenAIStreamSemantics(t *testing.T) {
	done := "data: [DONE]\n\n"
	cases := []struct {
		name, body string
		count      int
		ok         bool
	}{
		{"single", openAIChunk("5", "length") + openAIUsage(1) + done, 1, true},
		{"three", openAIChunk("5", "") + openAIChunk("6", "") + openAIChunk("7", "length") + openAIUsage(3) + done, 3, true},
		{"no-usage", openAIChunk("5", "length") + done, 1, false},
		{"usage-mismatch", openAIChunk("5", "length") + openAIUsage(2) + done, 1, false},
		{"early-stop", openAIChunk("5", "stop") + openAIUsage(1) + done, 2, false},
		{"abort", openAIChunk("5", "abort") + openAIUsage(1) + done, 1, false},
		{"missing-token-ids", `data: {"choices":[{"index":0,"text":"x","finish_reason":"length"}],"usage":null}` + "\n\n" + openAIUsage(1) + done, 1, false},
		{"after-finish", openAIChunk("5", "length") + openAIChunk("6", "") + openAIUsage(2) + done, 2, false},
		{"error", `data: {"error":{"message":"bad","code":400}}` + "\n\n", 1, false},
		{"truncated", openAIChunk("5", ""), 1, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := emptyRecord(0)
			e := parseOpenAIStream(strings.NewReader(c.body), time.Now(), 0, &r, c.count)
			if (e == nil) != c.ok {
				t.Fatal(e)
			}
			if c.ok && r.OutputTokens != c.count {
				t.Fatal(r)
			}
		})
	}
}

func TestOpenAIReplayBodyAndMetrics(t *testing.T) {
	var body map[string]any
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			w.WriteHeader(404)
			return
		}
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, openAIChunk("11", ""))
		w.(http.Flusher).Flush()
		time.Sleep(15 * time.Millisecond)
		fmt.Fprint(w, openAIChunk("12,13", "length")+openAIUsage(3)+"data: [DONE]\n\n")
	}))
	defer s.Close()
	c := DefaultConfig(s.URL+"/v1/completions", 1)
	c.API = "openai"
	c.Model = "Qwen/Qwen3-8B-AWQ"
	result, e := Run(t.Context(), []Req{{0, 17, 3}}, c)
	if e != nil {
		t.Fatal(e)
	}
	r := result.Records[0]
	if r.Status != "ok" || r.OutputTokens != 3 || !r.Coalesced || r.TTFTms < 0 || r.TPOTms < 5 {
		t.Fatalf("bad metrics: %+v", r)
	}
	prompt, _ := body["prompt"].([]any)
	opts, _ := body["stream_options"].(map[string]any)
	if len(prompt) != 17 || body["model"] != "Qwen/Qwen3-8B-AWQ" || body["stream"] != true || body["ignore_eos"] != true ||
		body["max_tokens"] != float64(3) || body["min_tokens"] != float64(3) || body["temperature"] != float64(0) ||
		body["return_token_ids"] != true || opts["include_usage"] != true {
		t.Fatal(body)
	}
}

// serving/eval/fp8_bundle.py re-derives these prompt IDs to hash the actual
// prompts; its test pins the same values.
func TestSyntheticPromptIDsArePinned(t *testing.T) {
	c := DefaultConfig("http://127.0.0.1:1/v1/completions", 1)
	c.API, c.Model = "openai", "m"
	bodies, e := prepare([]Req{{0, 5, 1}, {1, 3, 1}}, c)
	if e != nil {
		t.Fatal(e)
	}
	want := [][]any{{100.0, 103.0, 101.0, 101.0, 100.0}, {102.0, 100.0, 101.0}}
	for i, b := range bodies {
		var body map[string]any
		json.Unmarshal(b, &body)
		if fmt.Sprint(body["prompt"]) != fmt.Sprint(want[i]) {
			t.Fatal(i, body["prompt"])
		}
	}
}

func TestOpenAIZeroOutputRowsRequestOneToken(t *testing.T) {
	// vLLM rejects max_tokens=0, so a zero-output trace row becomes a one-token
	// (prefill-dominated) request; it can never count toward >=2-token goodput.
	c := DefaultConfig("http://127.0.0.1:1/v1/completions", 1)
	c.API, c.Model = "openai", "m"
	bodies, e := prepare([]Req{{0, 4, 0}}, c)
	if e != nil {
		t.Fatal(e)
	}
	var body map[string]any
	json.Unmarshal(bodies[0], &body)
	if body["max_tokens"] != float64(1) || body["min_tokens"] != float64(1) || expectedTokens(c, Req{0, 4, 0}) != 1 {
		t.Fatal(body)
	}
	c.API = "bogus"
	if _, e := prepare([]Req{{0, 4, 1}}, c); e == nil {
		t.Fatal("accepted unknown API")
	}
	c.API, c.Model = "openai", ""
	if _, e := prepare([]Req{{0, 4, 1}}, c); e == nil {
		t.Fatal("openai mode requires a model name")
	}
}

// openai-text sends the same seeded IDs as their single-token texts, so a
// string-only router can forward the request and prompt length stays exact.
func TestOpenAITextPromptJoinsTokenTexts(t *testing.T) {
	c := DefaultConfig("http://127.0.0.1:1/v1/completions", 1)
	c.API, c.Model = "openai-text", "m"
	c.TokenTexts = []string{" a", " b", " c", " d"}
	bodies, e := prepare([]Req{{0, 5, 1}, {1, 3, 1}}, c)
	if e != nil {
		t.Fatal(e)
	}
	// Same draws as TestSyntheticPromptIDsArePinned: 100,103,101,101,100 / 102,100,101.
	want := []string{" a d b b a", " c a b"}
	for i, b := range bodies {
		var body map[string]any
		json.Unmarshal(b, &body)
		if body["prompt"] != want[i] || body["return_token_ids"] != true || body["min_tokens"] != float64(1) {
			t.Fatal(i, body)
		}
	}
	if expectedTokens(c, Req{0, 4, 0}) != 1 {
		t.Fatal("openai-text must also request one token on zero-output rows")
	}
	c.TokenTexts = c.TokenTexts[:3]
	if _, e := prepare([]Req{{0, 4, 1}}, c); e == nil {
		t.Fatal("accepted texts not parallel to token IDs")
	}
	c.TokenTexts = []string{" a", "", " c", " d"}
	if _, e := prepare([]Req{{0, 4, 1}}, c); e == nil {
		t.Fatal("accepted an empty token text")
	}
}

func TestOpenAITextUsesOpenAIStreamParser(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Config.Tenant is sent as X-Tenant, which the Task 14 router requires.
		if r.Header.Get("X-Tenant") != "bench" {
			w.WriteHeader(400)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, openAIChunk("11", "")+openAIChunk("12", "length")+openAIUsage(2)+"data: [DONE]\n\n")
	}))
	defer s.Close()
	c := DefaultConfig(s.URL+"/v1/completions", 1)
	c.API, c.Model, c.TokenTexts = "openai-text", "m", []string{" a", " b", " c", " d"}
	c.Tenant = "bench"
	result, e := Run(t.Context(), []Req{{0, 4, 2}}, c)
	if e != nil || result.Records[0].Status != "ok" || result.Records[0].OutputTokens != 2 {
		t.Fatal(e, result.Records[0])
	}
}
