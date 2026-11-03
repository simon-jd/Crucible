package main

import (
	"strings"
	"testing"
)

func TestScaleEndpointPerResource(t *testing.T) {
	for resource, want := range map[string]string{
		"":             "/apis/apps/v1/namespaces/ns/deployments/model/scale",
		"deployments":  "/apis/apps/v1/namespaces/ns/deployments/model/scale",
		"statefulsets": "/apis/apps/v1/namespaces/ns/statefulsets/model/scale",
	} {
		c := ScaleClient{BaseURL: "https://10.96.0.1", Namespace: "ns", Name: "model", Resource: resource}
		got, err := c.endpoint()
		if err != nil || !strings.HasSuffix(got, want) {
			t.Fatalf("%q: %q %v", resource, got, err)
		}
	}
}

func TestScaleEndpointRejectsOtherResources(t *testing.T) {
	c := ScaleClient{BaseURL: "https://10.96.0.1", Namespace: "ns", Name: "model", Resource: "replicasets/../pods"}
	if _, err := c.endpoint(); err == nil {
		t.Fatal("unexpected resource accepted")
	}
}
