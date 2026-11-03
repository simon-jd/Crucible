package replay

import (
	"strings"
	"testing"
	"time"
)

func TestStreamSemantics(t *testing.T) {
	cases := []struct {
		name, body string
		count      int
		ok         bool
	}{
		{"zero", "data: {\"output_ids\":[],\"meta_info\":{\"completion_tokens\":0,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n", 0, true},
		{"single", "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":{\"type\":\"length\"}}}\n\ndata: [DONE]\n\n", 1, true},
		{"done-only", "data: [DONE]\n\n", 0, false},
		{"truncated", "data: {\"output_ids\":[1]}\n\n", 1, false},
		{"bad-json", "data: {\n\n", 0, false},
		{"error", "data: {\"error\":{\"message\":\"bad\"}}\n\n", 0, false},
		{"mismatch", "data: {\"output_ids\":[1],\"meta_info\":{\"completion_tokens\":2}}\n\n", 2, false},
		{"oversized", "data: " + strings.Repeat("x", maxEvent) + "\n\n", 1, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := emptyRecord(0)
			e := parseStream(strings.NewReader(c.body), time.Now(), 0, &r, c.count)
			if (e == nil) != c.ok {
				t.Fatal(e)
			}
			if c.ok && r.TPOTms != -1 {
				t.Fatal("undefined TPOT", r)
			}
		})
	}
}
