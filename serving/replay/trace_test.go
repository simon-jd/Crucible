package replay

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestRealTracePrefixes(t *testing.T) {
	var manifest []struct {
		Name, SHA256 string
		Rows         int
	}
	b, e := os.ReadFile("testdata/provenance.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &manifest); e != nil {
		t.Fatal(e)
	}
	for _, m := range manifest {
		t.Run(m.Name, func(t *testing.T) {
			b, e := os.ReadFile("testdata/" + m.Name + ".csv")
			if e != nil {
				t.Fatal(e)
			}
			if fmt.Sprintf("%x", sha256.Sum256(b)) != m.SHA256 {
				t.Fatal("provenance hash")
			}
			r, info, e := ParseTrace(strings.NewReader(string(b)), m.Name)
			if e != nil {
				t.Fatal(e)
			}
			if len(r) != 100 || info.Rows != m.Rows || r[0].ArrivalMs != 0 {
				t.Fatal(info)
			}
			t.Logf("rows=%d zero-output=%d sha=%s", info.Rows, info.ZeroOutputs, info.SHA256)
		})
	}
}
func TestTraceValidation(t *testing.T) {
	header := "Timestamp,Request tokens,Response tokens\n"
	for _, rows := range []string{"", "NaN,1,2\n", "1,1,-1\n", "2,1,1\n1,1,1\n", "1,-1,0\n", "1,1,1,extra\n"} {
		if _, _, e := ParseTrace(strings.NewReader(header+rows), "burstgpt"); e == nil {
			t.Errorf("accepted %q", rows)
		}
	}
	r, info, e := ParseTrace(strings.NewReader(header+"5,2,0\n5,2,1\n6,2,2\n"), "burstgpt")
	if e != nil || len(r) != 3 || info.ZeroOutputs != 1 || r[1].ArrivalMs != 0 || r[2].ArrivalMs != 1000 {
		t.Fatalf("%+v %+v %v", r, info, e)
	}
}
func TestSLOAndGoodput(t *testing.T) {
	r := Result{SenderValid: true, DurationMs: 2000, Records: []Record{{Status: "ok", SentMs: 1, OutputTokens: 2, TTFTms: 10, TPOTms: 2}, {Status: "stream_error", SentMs: 1}}}
	s, e := Calibrate(r, "calibration")
	if e != nil || s.Samples != 1 {
		t.Fatal(s, e)
	}
	g, n, e := Goodput(r, s, "evaluation")
	if e != nil || g != .5 || n != 1 {
		t.Fatal(g, n, e)
	}
	if _, _, e = Goodput(r, s, "calibration"); e == nil {
		t.Fatal("lineage overlap")
	}
	r.Aborted = true
	if _, _, e = Goodput(r, s, "evaluation"); e == nil {
		t.Fatal("aborted run accepted")
	}
	r.Aborted = false
	r.Records[0].SentMs = 6
	if _, _, e = Goodput(r, s, "evaluation"); e == nil {
		t.Fatal("invalid sender accepted")
	}
}
