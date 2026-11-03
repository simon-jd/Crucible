package replay

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

const maxEvent = 1 << 20

var errDone = fmt.Errorf("done")

// sseEvents calls handle for each bounded SSE data event. handle returns
// errDone to finish successfully; a stream ending without that is truncated.
func sseEvents(reader io.Reader, handle func(event string) error) error {
	scan := bufio.NewScanner(io.LimitReader(reader, 64<<20))
	scan.Buffer(make([]byte, 4096), maxEvent)
	var data strings.Builder
	events := 0
	for scan.Scan() {
		line := scan.Text()
		if line != "" {
			if strings.HasPrefix(line, "data:") {
				part := strings.TrimPrefix(line, "data:")
				if data.Len()+len(part) > maxEvent {
					return fmt.Errorf("event too large")
				}
				data.WriteString(part)
				data.WriteByte('\n')
			}
			continue
		}
		if data.Len() == 0 {
			continue
		}
		event := strings.TrimSpace(data.String())
		data.Reset()
		events++
		if events > 65536 {
			return fmt.Errorf("too many events")
		}
		if err := handle(event); err == errDone {
			return nil
		} else if err != nil {
			return err
		}
	}
	if err := scan.Err(); err != nil {
		return fmt.Errorf("stream read: %w", err)
	}
	return fmt.Errorf("truncated stream")
}

// tokenClock turns delta token arrivals into TTFT/TPOT on the record.
type tokenClock struct{ first, last float64 }

func (c *tokenClock) add(ids []int, start time.Time, sent float64, r *Record, expected int) error {
	if len(ids) == 0 {
		return nil
	}
	if r.OutputTokens+len(ids) > expected {
		return fmt.Errorf("too many output tokens")
	}
	for _, id := range ids {
		if id < 0 {
			return fmt.Errorf("negative output ID")
		}
	}
	now := ms(time.Since(start))
	if r.OutputTokens == 0 {
		c.first = now
		r.TTFTms = now - sent
	}
	c.last = now
	r.OutputTokens += len(ids)
	if len(ids) > 1 {
		r.Coalesced = true
	}
	if r.OutputTokens > 1 {
		r.TPOTms = (c.last - c.first) / float64(r.OutputTokens-1)
	}
	return nil
}

// Native v0.5.5 with normal tokenizer emits delta output_ids and cumulative
// completion_tokens. skip-tokenizer-init / cumulative token streams unsupported.
func parseStream(reader io.Reader, start time.Time, sent float64, r *Record, expected int) error {
	terminal := false
	var clock tokenClock
	return sseEvents(reader, func(event string) error {
		if event == "[DONE]" {
			if !terminal {
				return fmt.Errorf("DONE before finish")
			}
			if r.OutputTokens != expected {
				return fmt.Errorf("output count %d expected %d", r.OutputTokens, expected)
			}
			return errDone
		}
		if terminal {
			return fmt.Errorf("event after finish")
		}
		var out struct {
			IDs   []int           `json:"output_ids"`
			Error json.RawMessage `json:"error"`
			Meta  struct {
				Count  *int `json:"completion_tokens"`
				Finish *struct {
					Type string `json:"type"`
				} `json:"finish_reason"`
			} `json:"meta_info"`
		}
		if err := json.Unmarshal([]byte(event), &out); err != nil {
			return fmt.Errorf("malformed event: %w", err)
		}
		if len(out.Error) > 0 && string(out.Error) != "null" {
			return fmt.Errorf("server stream error")
		}
		if err := clock.add(out.IDs, start, sent, r, expected); err != nil {
			return err
		}
		if out.Meta.Count != nil && *out.Meta.Count != r.OutputTokens {
			return fmt.Errorf("token count mismatch")
		}
		if out.Meta.Finish != nil {
			if out.Meta.Finish.Type != "length" && out.Meta.Finish.Type != "stop" {
				return fmt.Errorf("unsuccessful finish: %s", out.Meta.Finish.Type)
			}
			if out.Meta.Count == nil {
				return fmt.Errorf("finish missing count")
			}
			terminal = true
		}
		return nil
	})
}

// parseOpenAIStream reads vLLM /v1/completions with return_token_ids and
// include_usage: delta token_ids per chunk, finish_reason on the last token
// chunk, a choices=[] usage chunk whose count must match, then [DONE].
func parseOpenAIStream(reader io.Reader, start time.Time, sent float64, r *Record, expected int) error {
	terminal, usage := false, false
	var clock tokenClock
	return sseEvents(reader, func(event string) error {
		if event == "[DONE]" {
			if !terminal || !usage {
				return fmt.Errorf("DONE before finish and usage")
			}
			if r.OutputTokens != expected {
				return fmt.Errorf("output count %d expected %d", r.OutputTokens, expected)
			}
			return errDone
		}
		if usage {
			return fmt.Errorf("event after usage")
		}
		var out struct {
			Choices []struct {
				IDs    *[]int  `json:"token_ids"`
				Finish *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				Count *int `json:"completion_tokens"`
			} `json:"usage"`
			Error json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal([]byte(event), &out); err != nil {
			return fmt.Errorf("malformed event: %w", err)
		}
		if len(out.Error) > 0 && string(out.Error) != "null" {
			return fmt.Errorf("server stream error")
		}
		if len(out.Choices) > 1 {
			return fmt.Errorf("multiple choices")
		}
		if len(out.Choices) == 1 {
			if terminal {
				return fmt.Errorf("event after finish")
			}
			choice := out.Choices[0]
			if choice.IDs == nil {
				return fmt.Errorf("chunk missing token_ids")
			}
			if err := clock.add(*choice.IDs, start, sent, r, expected); err != nil {
				return err
			}
			if choice.Finish != nil {
				if *choice.Finish != "length" && *choice.Finish != "stop" {
					return fmt.Errorf("unsuccessful finish: %s", *choice.Finish)
				}
				terminal = true
			}
			return nil
		}
		if out.Usage == nil || out.Usage.Count == nil {
			return fmt.Errorf("event without choices or usage")
		}
		if !terminal {
			return fmt.Errorf("usage before finish")
		}
		if *out.Usage.Count != r.OutputTokens {
			return fmt.Errorf("token count mismatch")
		}
		usage = true
		return nil
	})
}
