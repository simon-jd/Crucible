package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDiagnosticCLIOptionalAndOriginalJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	dir := t.TempDir()
	trace := filepath.Join(dir, "trace.csv")
	ids := filepath.Join(dir, "ids.json")
	if err := os.WriteFile(trace, []byte("Timestamp,Request tokens,Response tokens\n0,2,1\n"), 0600); err != nil {
		t.Fatal(err)
	}
	tokens, _ := json.Marshal(map[string]any{"Model": "fixture", "Revision": strings.Repeat("a", 40), "TokenizerSHA256": strings.Repeat("b", 64), "token_ids": []int{1, 2}})
	if err := os.WriteFile(ids, tokens, 0600); err != nil {
		t.Fatal(err)
	}
	args := []string{"-trace", trace, "-format", "burstgpt", "-token-ids", ids, "-url", server.URL}
	for _, enabled := range []bool{false, true} {
		name := "disabled"
		if enabled {
			name = "enabled"
		}
		t.Run(name, func(t *testing.T) {
			argv := append([]string(nil), args...)
			path := filepath.Join(dir, "sidecar.json")
			if enabled {
				argv = append(argv, "-phase-sidecar", path)
			}
			var out bytes.Buffer
			err := run(argv, &out)
			if err != nil && err.Error() != "sender-invalid run; no accepted performance" {
				t.Fatal(err)
			}
			var raw map[string]json.RawMessage
			if e := json.Unmarshal(out.Bytes(), &raw); e != nil {
				t.Fatal(e)
			}
			if len(raw) != 4 || raw["Trace"] == nil || raw["Config"] == nil || raw["TokenProvenance"] == nil || raw["Run"] == nil {
				t.Fatal("ordinary JSON changed", raw)
			}
			var config map[string]json.RawMessage
			json.Unmarshal(raw["Config"], &config)
			if len(config) != 7 || config["Seed"] == nil || config["TokenIDs"] != nil || string(config["API"]) != `"sglang"` {
				t.Fatal("Config serialization changed", config)
			}
			if !enabled {
				if _, e := os.Stat(path); !os.IsNotExist(e) {
					t.Fatal("disabled capture wrote file", e)
				}
				return
			}
			data, e := os.ReadFile(path)
			if e != nil {
				t.Fatal(e)
			}
			var capture struct {
				CaptureValid bool  `json:"capture_valid"`
				Requests     []any `json:"requests"`
			}
			if e = json.Unmarshal(data, &capture); e != nil || !capture.CaptureValid || len(capture.Requests) != 1 {
				t.Fatal(e, capture, string(data))
			}
			// Existing sidecar must fail, while preserving complete original raw JSON.
			out.Reset()
			e = run(argv, &out)
			if e == nil || !strings.Contains(e.Error(), "exist") {
				t.Fatal("sidecar collision accepted", e)
			}
			if e = json.Unmarshal(out.Bytes(), &raw); e != nil {
				t.Fatal("sidecar error discarded raw run", e)
			}
			after, _ := os.ReadFile(path)
			if !bytes.Equal(data, after) {
				t.Fatal("sidecar clobbered")
			}
		})
	}
}
