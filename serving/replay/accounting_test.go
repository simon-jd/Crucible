package replay

import (
	"reflect"
	"testing"
)

// Synthetic terminal observations, not an injected serving fault.
func TestAccountingNoSilentLoss(t *testing.T) {
	if _, err := AccountRequests([]string{"synthetic-a", "synthetic-lost"}, []Attempt{
		{RequestID: "synthetic-a", Number: 0, Status: "completed", Final: true},
	}); err == nil {
		t.Fatal("missing planned request silently disappeared")
	}
}

func TestAccountingRetryFinalFailureAndDeterminism(t *testing.T) {
	planned := []string{"synthetic-z", "synthetic-a", "synthetic-b"}
	attempts := []Attempt{
		{RequestID: "synthetic-z", Number: 1, Status: "failed", Error: "synthetic retry terminal failure", Final: true},
		{RequestID: "synthetic-a", Number: 0, Status: "completed", Final: true},
		{RequestID: "synthetic-z", Number: 0, Status: "failed", Error: "synthetic handoff", Final: false},
		{RequestID: "synthetic-b", Number: 0, Status: "failed", Error: "synthetic single failure", Final: true},
	}
	beforePlanned := append([]string(nil), planned...)
	beforeAttempts := append([]Attempt(nil), attempts...)
	got, err := AccountRequests(planned, attempts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Planned != 3 || got.Completed != 1 || got.Failed != 1 || got.Retried != 1 {
		t.Fatalf("incorrect exclusive counts: %+v", got)
	}
	if got.Requests[2].RequestID != "synthetic-z" || got.Requests[2].Category != "retried" || got.Requests[2].FinalStatus != "failed" || got.Requests[2].AttemptCount != 2 {
		t.Fatalf("final failed retry disappeared: %+v", got.Requests[2])
	}
	if got.Requests[2].Attempts[1].Error != "synthetic retry terminal failure" {
		t.Fatal("raw failure lost")
	}
	for i := range planned {
		if planned[i] != beforePlanned[i] {
			t.Fatal("planned IDs mutated")
		}
	}
	for i := range attempts {
		if attempts[i] != beforeAttempts[i] {
			t.Fatal("attempt inputs mutated")
		}
	}
	reversed := append([]Attempt(nil), attempts...)
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	other, err := AccountRequests([]string{"synthetic-b", "synthetic-a", "synthetic-z"}, reversed)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, other) {
		t.Fatal("result depends on input order")
	}
	got.Requests[2].Attempts[0].Error = "changed output"
	if attempts[2].Error != beforeAttempts[2].Error {
		t.Fatal("result aliases input")
	}
}

func TestAccountingRetryFinalSuccess(t *testing.T) {
	got, err := AccountRequests([]string{"synthetic-a"}, []Attempt{
		{RequestID: "synthetic-a", Number: 0, Status: "failed", Error: "synthetic failed prefix"},
		{RequestID: "synthetic-a", Number: 1, Status: "completed", Final: true},
	})
	if err != nil || got.Retried != 1 || got.Requests[0].FinalStatus != "completed" {
		t.Fatalf("%+v %v", got, err)
	}
}

func TestAccountingRejectsInvalidAndOpenChains(t *testing.T) {
	final := Attempt{RequestID: "synthetic-a", Number: 0, Status: "failed", Error: "synthetic", Final: true}
	cases := []struct {
		name     string
		planned  []string
		attempts []Attempt
	}{
		{"empty", nil, nil},
		{"blank", []string{" "}, nil},
		{"duplicate_plan", []string{"synthetic-a", "synthetic-a"}, []Attempt{final}},
		{"unknown", []string{"synthetic-other"}, []Attempt{final}},
		{"missing_all", []string{"synthetic-a"}, nil},
		{"duplicate_attempt", []string{"synthetic-a"}, []Attempt{final, final}},
		{"negative", []string{"synthetic-a"}, []Attempt{{RequestID: "synthetic-a", Number: -1, Status: "failed", Final: true}}},
		{"sparse", []string{"synthetic-a"}, []Attempt{{RequestID: "synthetic-a", Number: 1, Status: "failed", Final: true}}},
		{"invalid_status", []string{"synthetic-a"}, []Attempt{{RequestID: "synthetic-a", Status: "retried", Final: true}}},
		{"blank_attempt_id", []string{"synthetic-a"}, []Attempt{{RequestID: "", Status: "failed", Final: true}}},
		{"missing_retry_successor", []string{"synthetic-a"}, []Attempt{{RequestID: "synthetic-a", Status: "failed", Final: false}}},
		{"success_before_retry", []string{"synthetic-a"}, []Attempt{{RequestID: "synthetic-a", Status: "completed"}, {RequestID: "synthetic-a", Number: 1, Status: "completed", Final: true}}},
		{"conflicting_final_closure", []string{"synthetic-a"}, []Attempt{final, {RequestID: "synthetic-a", Number: 1, Status: "completed", Final: true}}},
		{"open_retry_tail", []string{"synthetic-a"}, []Attempt{{RequestID: "synthetic-a", Status: "failed"}, {RequestID: "synthetic-a", Number: 1, Status: "failed"}}},
		{"missing_middle", []string{"synthetic-a"}, []Attempt{{RequestID: "synthetic-a", Status: "failed"}, {RequestID: "synthetic-a", Number: 2, Status: "completed", Final: true}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := AccountRequests(tc.planned, tc.attempts)
			if err == nil {
				t.Fatalf("invalid chain accepted: %+v", got)
			}
			if !reflect.DeepEqual(got, Accounting{}) {
				t.Fatal("error returned valid-looking partial counts")
			}
		})
	}
}

// The replayer makes one attempt per scheduled request and never retries, so
// every record becomes one final attempt; any non-ok status is a failure.
func TestAccountRecordsEveryScheduledRequest(t *testing.T) {
	ok := emptyRecord(0)
	ok.Status = "ok"
	refused := emptyRecord(1)
	refused.Status, refused.Error = "http_error", "HTTP 502"
	rejected := emptyRecord(2)
	rejected.Status = "admission_rejected"
	never := emptyRecord(3) // unset status: still counted, as a failure
	summary, err := AccountRecords([]Record{ok, refused, rejected, never})
	if err != nil {
		t.Fatal(err)
	}
	if summary.Planned != 4 || summary.Completed != 1 || summary.Failed != 3 || summary.Retried != 0 {
		t.Fatalf("%+v", summary)
	}
	want := map[string]int{"http_error": 1, "admission_rejected": 1, "unset": 1}
	if !reflect.DeepEqual(summary.FailureStatuses, want) {
		t.Fatal(summary.FailureStatuses)
	}
	if _, err := AccountRecords(nil); err == nil {
		t.Fatal("empty run accepted")
	}
}
