from serving.k8s import adapter_k8s
from serving.localscale import metrics


def test_replica_urls_follow_statefulset_dns():
    urls = adapter_k8s.replica_urls("fake", "fake", "crucible", 3, 8000)
    assert urls == {
        "fake-0": "http://fake-0.fake.crucible.svc.cluster.local:8000",
        "fake-1": "http://fake-1.fake.crucible.svc.cluster.local:8000",
        "fake-2": "http://fake-2.fake.crucible.svc.cluster.local:8000",
    }


def test_poll_marks_scraped_replicas_serving_and_others_not():
    urls = adapter_k8s.replica_urls("fake", "fake", "crucible", 2, 8000)
    state = adapter_k8s.new_state("m", urls)
    text = 'vllm:num_requests_waiting{engine="0",model_name="m"} 3\n'

    def scrape(url):
        if "fake-0" in url:
            return text
        raise OSError("refused")

    adapter_k8s.poll_once(state, scrape, desired=lambda: 1, now=100.0)
    assert state.serving == {"fake-0": True, "fake-1": False}
    assert state.desired == 1
    code, snap = state.snapshot("fake-0")
    assert code == 200 and snap["Queue"] == 3
    assert state.snapshot("fake-1")[0] == 503
    assert state.aggregate(now=100.0)["queue_per_replica"] == 3


def test_failed_scrape_forgets_a_replica_that_was_serving():
    urls = adapter_k8s.replica_urls("fake", "fake", "crucible", 1, 8000)
    state = adapter_k8s.new_state("m", urls)
    adapter_k8s.poll_once(state, lambda url: "", desired=lambda: 1, now=1.0)
    assert state.serving["fake-0"]

    def refuse(url):
        raise OSError

    adapter_k8s.poll_once(state, refuse, desired=lambda: 1, now=2.0)
    assert not state.serving["fake-0"] and state.snapshot("fake-0")[0] == 503


def test_desired_read_failure_keeps_the_last_value():
    urls = adapter_k8s.replica_urls("fake", "fake", "crucible", 1, 8000)
    state = adapter_k8s.new_state("m", urls)
    adapter_k8s.poll_once(state, lambda url: "", desired=lambda: 2, now=1.0)

    def broken():
        raise OSError

    adapter_k8s.poll_once(state, lambda url: "", desired=broken, now=2.0)
    assert state.desired == 2


def test_scale_replicas_reads_the_scale_subresource():
    body = b'{"kind": "Scale", "apiVersion": "autoscaling/v1", "spec": {"replicas": 2}, "status": {"replicas": 1}}'
    assert adapter_k8s.parse_scale(body) == 2
    assert metrics.parse("").waiting == 0
