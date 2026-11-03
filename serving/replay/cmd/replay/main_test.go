package main

import (
	"bytes"
	"context"
	"crucible/serving/replay"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestCLIHelper(t *testing.T) {
	if os.Getenv("CRUCIBLE_REPLAY_TEST_HELPER") != "1" {
		t.Skip("subprocess entrypoint")
	}
	for i, arg := range os.Args {
		if arg == "--" {
			if e := run(os.Args[i+1:], os.Stdout); e != nil {
				os.Stderr.WriteString(e.Error())
				os.Exit(1)
			}
			os.Exit(0)
		}
	}
	os.Exit(2)
}
func TestCLISignalCancellation(t *testing.T) {
	entered := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		entered <- struct{}{}
		<-r.Context().Done()
	}))
	defer server.Close()
	dir := t.TempDir()
	trace := filepath.Join(dir, "trace.csv")
	ids := filepath.Join(dir, "ids.json")
	if e := os.WriteFile(trace, []byte("Timestamp,Request tokens,Response tokens\n0,2,2\n10,2,2\n"), 0600); e != nil {
		t.Fatal(e)
	}
	data, _ := json.Marshal(map[string]any{"Model": "fixture", "Revision": strings.Repeat("a", 40), "TokenizerSHA256": strings.Repeat("b", 64), "token_ids": []int{1, 2}})
	if e := os.WriteFile(ids, data, 0600); e != nil {
		t.Fatal(e)
	}
	binary, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "-test.run=^TestCLIHelper$", "--", "-trace", trace, "-format", "burstgpt", "-token-ids", ids, "-url", server.URL)
	cmd.Env = append(os.Environ(), "CRUCIBLE_REPLAY_TEST_HELPER=1")
	var out, stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if e = cmd.Start(); e != nil {
		t.Fatal(e)
	}
	select {
	case <-entered:
	case <-ctx.Done():
		cmd.Wait()
		t.Fatal("request never arrived", stderr.String())
	}
	if e = cmd.Process.Signal(syscall.SIGTERM); e != nil {
		t.Fatal(e)
	}
	e = cmd.Wait()
	if e == nil || ctx.Err() != nil || !strings.Contains(stderr.String(), "context canceled") {
		t.Fatal(e, ctx.Err(), stderr.String())
	}
	var result struct{ Run replay.Result }
	if e = json.Unmarshal(out.Bytes(), &result); e != nil {
		t.Fatal(e, out.String())
	}
	if result.Run.SenderValid || len(result.Run.Records) != 2 || result.Run.Records[1].SentMs != -1 {
		t.Fatal(result)
	}
	// Provenance names the allowlist by file hash; it does not copy every ID.
	var provenance struct{ TokenProvenance map[string]any }
	json.Unmarshal(out.Bytes(), &provenance)
	p := provenance.TokenProvenance
	if p["token_ids"] != nil || p["TokenIDCount"] != float64(2) || p["AllowlistSHA256"] != fmt.Sprintf("%x", sha256.Sum256(data)) || p["Model"] != "fixture" {
		t.Fatal(p)
	}
}
