package replay

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"strconv"
	"time"
)

type TraceInfo struct {
	Format, SHA256    string
	Rows, ZeroOutputs int
	Origin            string
}

// ParseTrace consumes a bounded selected CSV. It preserves duplicate arrivals and
// zero outputs. Selection, source URL/revision and original byte hashes belong in
// the adjacent provenance manifest; this hash identifies the exact replay input.
func ParseTrace(reader io.Reader, format string) ([]Req, TraceInfo, error) {
	info := TraceInfo{Format: format}
	data, err := io.ReadAll(io.LimitReader(reader, (4<<20)+1))
	if err != nil {
		return nil, info, err
	}
	if len(data) > 4<<20 {
		return nil, info, fmt.Errorf("trace exceeds 4MiB")
	}
	info.SHA256 = fmt.Sprintf("%x", sha256.Sum256(data))
	cr := csv.NewReader(bytes.NewReader(data))
	header, err := cr.Read()
	if err != nil {
		return nil, info, err
	}
	indexes := map[string]int{}
	for i, k := range header {
		if _, ok := indexes[k]; ok {
			return nil, info, fmt.Errorf("duplicate column")
		}
		indexes[k] = i
	}
	names := []string{"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
	if format == "burstgpt" {
		names = []string{"Timestamp", "Request tokens", "Response tokens"}
	} else if format != "azure" {
		return nil, info, fmt.Errorf("unknown trace format")
	}
	for _, k := range names {
		if _, ok := indexes[k]; !ok {
			return nil, info, fmt.Errorf("missing %s", k)
		}
	}
	var result []Req
	var origin time.Time
	var originSeconds float64
	for {
		row, e := cr.Read()
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, info, e
		}
		if len(result) >= MaxRequests {
			return nil, info, fmt.Errorf("too many trace rows")
		}
		stamp := row[indexes[names[0]]]
		var arrival float64
		if format == "azure" {
			ts, e := time.Parse("2006-01-02 15:04:05.999999999", stamp)
			if e != nil {
				return nil, info, e
			}
			if len(result) == 0 {
				origin = ts
			}
			arrival = ms(ts.Sub(origin))
		} else {
			s, e := strconv.ParseFloat(stamp, 64)
			if e != nil || math.IsNaN(s) || math.IsInf(s, 0) || s < 0 {
				return nil, info, fmt.Errorf("invalid timestamp")
			}
			if len(result) == 0 {
				originSeconds = s
			}
			arrival = (s - originSeconds) * 1000
		}
		if len(result) == 0 {
			info.Origin = stamp
		}
		in, e := strconv.Atoi(row[indexes[names[1]]])
		if e != nil {
			return nil, info, e
		}
		out, e := strconv.Atoi(row[indexes[names[2]]])
		if e != nil {
			return nil, info, e
		}
		if in < 0 || out < 0 || in > MaxTokens || out > MaxTokens || in+out > MaxTokens || arrival < 0 || !finiteMetric(arrival) || (len(result) > 0 && arrival < result[len(result)-1].ArrivalMs) {
			return nil, info, fmt.Errorf("invalid or unsorted trace row %d", len(result)+1)
		}
		result = append(result, Req{arrival, in, out})
		if out == 0 {
			info.ZeroOutputs++
		}
	}
	if len(result) == 0 {
		return nil, info, fmt.Errorf("empty trace")
	}
	info.Rows = len(result)
	return result, info, nil
}
