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
	"sync/atomic"
	"testing"
)

func TestCLIClientRetriesAccountsRetried(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		if calls.Add(1) == 1 { // cut the first stream after its headers
			w.(http.Flusher).Flush()
			if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
				conn.Close()
			}
			return
		}
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
	var out bytes.Buffer
	err := run([]string{"-trace", trace, "-format", "burstgpt", "-token-ids", ids, "-url", server.URL, "-account", "-client-retries", "1"}, &out)
	if err != nil {
		t.Fatal(err, out.String())
	}
	var got struct {
		Config     struct{ ClientRetries int }
		Accounting struct{ Planned, Completed, Failed, Retried, RetriedCompleted int }
	}
	if e := json.Unmarshal(out.Bytes(), &got); e != nil {
		t.Fatal(e)
	}
	a := got.Accounting
	if got.Config.ClientRetries != 1 || a.Planned != 1 || a.Retried != 1 || a.RetriedCompleted != 1 || a.Failed != 0 || calls.Load() != 2 {
		t.Fatalf("retry not accounted: %+v calls=%d", got, calls.Load())
	}
}
