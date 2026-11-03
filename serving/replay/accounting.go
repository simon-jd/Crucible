package replay

import (
	"fmt"
	"sort"
	"strings"
)

// Attempt is one terminal observation. Final=false explicitly requires a retry
// successor; it is valid only for failed attempts. No retries are scheduled here.
type Attempt struct {
	RequestID string
	Number    int
	Status    string
	Error     string
	Final     bool
}

// RequestAccounting preserves a complete terminal chain and its final outcome.
// Category=retried does not imply that FinalStatus=completed.
type RequestAccounting struct {
	RequestID    string
	Category     string
	FinalStatus  string
	AttemptCount int
	Attempts     []Attempt
}

// Accounting partitions planned logical requests into mutually exclusive counts.
type Accounting struct {
	Planned   int
	Completed int
	Failed    int
	Retried   int
	Requests  []RequestAccounting
}

// AccountRequests requires exactly one complete chain for every planned ID.
// Invalid input returns an error and zero Accounting, never partial valid counts.
// Results are sorted independent copies; caller-owned slices remain untouched.
func AccountRequests(planned []string, attempts []Attempt) (Accounting, error) {
	if len(planned) == 0 {
		return Accounting{}, fmt.Errorf("planned request IDs must not be empty")
	}
	chains := make(map[string][]Attempt, len(planned))
	for _, id := range planned {
		if strings.TrimSpace(id) == "" {
			return Accounting{}, fmt.Errorf("blank planned request ID")
		}
		if _, exists := chains[id]; exists {
			return Accounting{}, fmt.Errorf("duplicate planned request %q", id)
		}
		chains[id] = nil
	}
	for _, attempt := range attempts {
		if _, exists := chains[attempt.RequestID]; !exists {
			return Accounting{}, fmt.Errorf("unknown attempt request %q", attempt.RequestID)
		}
		if attempt.Number < 0 || (attempt.Status != "completed" && attempt.Status != "failed") {
			return Accounting{}, fmt.Errorf("invalid terminal attempt for %q", attempt.RequestID)
		}
		chains[attempt.RequestID] = append(chains[attempt.RequestID], attempt)
	}
	ids := append([]string(nil), planned...)
	sort.Strings(ids)
	result := Accounting{Planned: len(ids), Requests: make([]RequestAccounting, 0, len(ids))}
	for _, id := range ids {
		chain := chains[id]
		if len(chain) == 0 {
			return Accounting{}, fmt.Errorf("missing terminal chain for %q", id)
		}
		sort.Slice(chain, func(i, j int) bool { return chain[i].Number < chain[j].Number })
		for i, attempt := range chain {
			if attempt.Number != i {
				return Accounting{}, fmt.Errorf("duplicate or noncontiguous attempt for %q at %d", id, i)
			}
			last := i == len(chain)-1
			if last && !attempt.Final {
				return Accounting{}, fmt.Errorf("open retry handoff for %q: successor missing", id)
			}
			if !last && (attempt.Final || attempt.Status != "failed") {
				return Accounting{}, fmt.Errorf("conflicting closure or retry after success for %q", id)
			}
		}
		finalStatus := chain[len(chain)-1].Status
		category := finalStatus
		if len(chain) > 1 {
			category = "retried"
		}
		switch category {
		case "completed":
			result.Completed++
		case "failed":
			result.Failed++
		case "retried":
			result.Retried++
		}
		result.Requests = append(result.Requests, RequestAccounting{
			RequestID: id, Category: category, FinalStatus: finalStatus,
			AttemptCount: len(chain), Attempts: chain,
		})
	}
	if result.Completed+result.Failed+result.Retried != result.Planned {
		return Accounting{}, fmt.Errorf("logical-request accounting count mismatch")
	}
	return result, nil
}

// RecordSummary is AccountRequests applied to one replay run, without the
// per-request chains. FailureStatuses counts final failures by record status;
// RetriedAttemptStatuses counts the failed attempts that were resent.
type RecordSummary struct {
	Planned, Completed, Failed, Retried int
	RetriedCompleted                    int // retried requests whose last attempt completed
	FailureStatuses                     map[string]int
	RetriedAttemptStatuses              map[string]int `json:",omitempty"`
}

// AccountRecords turns every scheduled record into its attempt chain: one
// final attempt, or Record.Attempts when the client resent it. Status "ok" is
// completed, anything else failed.
func AccountRecords(records []Record) (RecordSummary, error) {
	planned := make([]string, len(records))
	var attempts []Attempt
	statuses, retriedStatuses := map[string]int{}, map[string]int{}
	attempt := func(id string, n int, status, detail string, final bool) Attempt {
		a := Attempt{RequestID: id, Number: n, Status: "completed", Final: final}
		if status != "ok" {
			if status == "" {
				status = "unset"
			}
			a.Status, a.Error = "failed", status+": "+detail
		}
		return a
	}
	for i, r := range records {
		planned[i] = fmt.Sprintf("r%06d", i)
		chain := r.Attempts
		if len(chain) == 0 {
			chain = []AttemptRecord{{Status: r.Status, Error: r.Error}}
		} else if last := chain[len(chain)-1]; last.Status != r.Status || last.Error != r.Error {
			return RecordSummary{}, fmt.Errorf("record %d: last attempt differs from its outcome", i)
		}
		for n, a := range chain {
			final := n == len(chain)-1
			attempts = append(attempts, attempt(planned[i], n, a.Status, a.Error, final))
			if !final && a.Status != "ok" {
				retriedStatuses[a.Status]++
			}
		}
		if r.Status != "ok" {
			status := r.Status
			if status == "" {
				status = "unset"
			}
			statuses[status]++
		}
	}
	acc, err := AccountRequests(planned, attempts)
	if err != nil {
		return RecordSummary{}, err
	}
	summary := RecordSummary{Planned: acc.Planned, Completed: acc.Completed, Failed: acc.Failed, Retried: acc.Retried, FailureStatuses: statuses}
	for _, req := range acc.Requests {
		if req.Category == "retried" && req.FinalStatus == "completed" {
			summary.RetriedCompleted++
		}
	}
	if len(retriedStatuses) > 0 {
		summary.RetriedAttemptStatuses = retriedStatuses
	}
	return summary, nil
}
