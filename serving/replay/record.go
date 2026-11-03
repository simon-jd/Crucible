package replay

import (
	"fmt"
	"math"
	"sort"
)

type Record struct {
	ScheduledMs, SentMs, TTFTms, TPOTms float64
	CompletedMs                         float64
	OutputTokens                        int
	Coalesced                           bool
	Status, Error                       string
	// Attempts is set only when the request was resent (Config.ClientRetries):
	// every attempt in order, the last one being the outcome above. SentMs is
	// then the first attempt's emission and TTFTms counts from it.
	Attempts []AttemptRecord `json:",omitempty"`
}

// AttemptRecord is one send of a resent request.
type AttemptRecord struct {
	SentMs, CompletedMs float64
	Status, Error       string
}
type Result struct {
	Records     []Record
	SenderValid bool
	Aborted     bool
	DurationMs  float64
}

func emptyRecord(s float64) Record { return Record{ScheduledMs: s, SentMs: -1, TTFTms: -1, TPOTms: -1} }
func SenderValid(records []Record) bool {
	if len(records) == 0 {
		return false
	}
	late := 0
	for _, r := range records {
		if r.ScheduledMs < 0 || r.SentMs < 0 || math.IsNaN(r.SentMs) || math.IsInf(r.SentMs, 0) || math.IsNaN(r.ScheduledMs) || math.IsInf(r.ScheduledMs, 0) || r.SentMs-r.ScheduledMs > 5 {
			late++
		}
	}
	return late*100 <= len(records)
}

type SLO struct {
	TTFTms, TPOTms float64
	CalibrationID  string
	Samples        int
}

// Calibrate uses nearest-rank p99 from complete >=2-token requests. Failures
// and undefined metrics are reported by the caller, never imputed as zero.
func Calibrate(r Result, id string) (SLO, error) {
	if r.Aborted || !r.SenderValid || !SenderValid(r.Records) || id == "" {
		return SLO{}, fmt.Errorf("invalid calibration lineage or sender")
	}
	var a, b []float64
	for _, x := range r.Records {
		if x.Status == "ok" && x.OutputTokens >= 2 && finiteMetric(x.TTFTms) && finiteMetric(x.TPOTms) {
			a = append(a, x.TTFTms)
			b = append(b, x.TPOTms)
		}
	}
	if len(a) == 0 {
		return SLO{}, fmt.Errorf("no defined calibration samples")
	}
	sort.Float64s(a)
	sort.Float64s(b)
	k := int(math.Ceil(.99*float64(len(a)))) - 1
	return SLO{a[k], b[k], id, len(a)}, nil
}
func finiteMetric(x float64) bool { return x >= 0 && !math.IsNaN(x) && !math.IsInf(x, 0) }
func Goodput(r Result, s SLO, evaluationID string) (float64, int, error) {
	if r.Aborted || !r.SenderValid || !SenderValid(r.Records) || !finiteMetric(r.DurationMs) || r.DurationMs == 0 || !finiteMetric(s.TTFTms) || !finiteMetric(s.TPOTms) || s.CalibrationID == "" || evaluationID == "" || evaluationID == s.CalibrationID || s.Samples < 1 {
		return 0, 0, fmt.Errorf("invalid run or SLO lineage")
	}
	good := 0
	for _, x := range r.Records {
		if x.Status == "ok" && x.OutputTokens >= 2 && finiteMetric(x.TTFTms) && finiteMetric(x.TPOTms) && x.TTFTms <= s.TTFTms && x.TPOTms <= s.TPOTms {
			good++
		}
	}
	return float64(good) / (r.DurationMs / 1000), good, nil
}
