"""Two-process async GRPO orchestration on CPU (gloo): schedule, weight versions, staleness, merge."""

import json
import multiprocessing as mp
import socket
import time

import pytest

from rl.local import async_grpo as ag

torch = pytest.importorskip("torch")

P, G = 2, 3  # prompts per step, samples per group


def test_schedule_matches_grpo_run_k1():
    # grpo_run --staleness 1: B0 and B1 from version 0, then B_g from version g-1.
    assert [ag.generation_version(g) for g in range(5)] == [0, 0, 1, 2, 3]
    steps = 4
    received = [v for g in range(steps) for v in ag.receives_after(g, steps)]
    assert received == [1, 2, 3, 4]  # one version per learner step; the last one is for the eval
    assert ag.receives_after(0, 1) == [1]


def test_group_codec_round_trip():
    from rl.rollout.sampler import Prompt, Sample, SamplingConfig

    prompt = Prompt("p1", "question?")
    samples = (Sample("p1", "a", (1, 2), (-0.5, -0.25), 3), Sample("p1", "b", (7,), (-1.0,), 3))
    row = ag.encode_group(
        prompt, samples, (1.0, 0.0), "g4", "g4/0", SamplingConfig(seed=17, max_tokens=8)
    )
    (group,) = ag.decode_groups(json.loads(json.dumps([row])))
    assert group.prompt == prompt and group.samples == samples and group.rewards == (1.0, 0.0)
    assert group.sampling == SamplingConfig(seed=17, max_tokens=8) and group.group_id == "g4/0"


class FakeLearner:
    """version t -> t+1 per step; weight w == version, so the rollout can check what it holds."""

    def __init__(self):
        self.version = 0
        self.w = torch.zeros(4)

    def step(self, groups):
        time.sleep(0.05)
        ages = [self.version - s.policy_version for g in groups for s in g.samples]
        dropped = sum(a > 1 for a in ages)
        self.version += 1
        self.w = torch.full((4,), float(self.version))
        return {"loss": 0.0, "dropped_stale": dropped, "accepted_samples": len(ages) - dropped,
                "effective_tokens": 2 * len(ages), "policy_version": self.version}  # fmt: skip

    def export(self):
        return [("w", self.w.clone())]


class FakeRollout:
    def __init__(self):
        self.w = torch.zeros(4)
        self.held = []

    def generate(self, g, version):
        from rl.rollout.sampler import Prompt, Sample, SamplingConfig

        assert torch.equal(self.w, torch.full((4,), float(version))), (g, version, self.w)
        self.held.append(version)
        t0 = time.time()
        time.sleep(0.05)
        rows = []
        for i in range(P):
            prompt = Prompt(f"q{g}-{i}", "text")
            samples = tuple(Sample(prompt.prompt_id, "x", (1,), (-0.1,), version) for _ in range(G))
            rows.append(
                ag.encode_group(prompt, samples, (1.0,) * G, f"g{g}", f"g{g}/{i}", SamplingConfig())
            )
        meta = {"reward": 1.0, "shaped_reward": 1.0, "flags": {"truncated": 1}, "len": 1.0, "samples": P * G,
                "rollout_s": 0.05, "reward_s": 0.0, "gen_interval": [t0, time.time()]}  # fmt: skip
        return rows, meta

    def load(self, tensors):
        ((name, value),) = tensors
        assert name == "w"
        self.w = value.clone()


def _role(role, port, tmp, steps, bucket_mb):
    from datetime import timedelta

    import torch.distributed as dist

    from rl.sync.bucketed import make_weight_sync
    from rl.sync.pair_group import destroy_pair_group, init_pair_group

    rank = 0 if role == "learner" else 1
    dist.init_process_group("gloo", init_method=f"file://{tmp}/rdv-{role}", rank=0, world_size=1,
                            timeout=timedelta(seconds=60))  # fmt: skip
    group, kv = init_pair_group("gloo", "127.0.0.1", port, rank, timeout=60)
    device = torch.device("cpu")
    store, sync = make_weight_sync({"w": ((4,), torch.float32)}, keep=rank == 1, group=group,
                                   timeout=60, bucket_mb=bucket_mb)  # fmt: skip
    channel = ag.Channel(kv, timeout=60)
    try:
        if role == "learner":
            with open(f"{tmp}/learner/learner.jsonl", "w") as log:
                result = ag.run_learner(FakeLearner(), channel, sync, group, steps, log, device)
            result |= {"peak_allocated_bytes": 1}
            (tmp / "learner" / "learner-summary.json").write_text(json.dumps(result))
            channel.get("rollout/done")
        else:
            backend = FakeRollout()
            with open(f"{tmp}/rollout/rollout.jsonl", "w") as log:
                result = ag.run_rollout(backend, channel, sync, store, group, steps, log, device)
            assert backend.held == [ag.generation_version(g) for g in range(steps)]
            assert torch.equal(backend.w, torch.full((4,), float(steps)))
            summary = {"args": {"seed": "17", "eval_n": "4", "staleness": "1"}, "data_sha256": "d",
                       "after": {"gsm8k": {"pass_at_1": 0.5}}, "eval_after_s": 1.0, "peak_allocated_bytes": 2,
                       **result}  # fmt: skip
            (tmp / "rollout" / "rollout-summary.json").write_text(json.dumps(summary))
            channel.post("rollout/done", True)
    finally:
        destroy_pair_group(group)
        dist.destroy_process_group()


@pytest.mark.parametrize("bucket_mb", [0, 1])  # per-tensor WeightSync, BucketedSync
def test_two_process_pipeline_versions_staleness_and_merge(tmp_path, bucket_mb):
    steps = 5
    for d in ("learner", "rollout"):
        (tmp_path / d).mkdir()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    children = [
        context.Process(target=_role, args=(r, port, tmp_path, steps, bucket_mb))
        for r in ("learner", "rollout")
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(120)
        assert [child.exitcode for child in children] == [0, 0]
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join()
    rollout = json.loads((tmp_path / "rollout/rollout-summary.json").read_text())
    assert rollout["loaded_versions"] == [1, 2, 3, 4, 5] and rollout["final_version"] == steps
    learner = json.loads((tmp_path / "learner/learner-summary.json").read_text())
    assert learner["staleness_histogram"] == {"0": P * G, "1": (steps - 1) * P * G}

    before = tmp_path / "before.json"
    before.write_text(
        json.dumps(
            {"data_sha256": "d", "args": {"seed": "17"}, "before": {"gsm8k": {"pass_at_1": 0.4}}}
        )
    )
    summary = ag.merge(tmp_path / "learner", tmp_path / "rollout", tmp_path / "merged", before)
    pipe = summary["pipeline"]
    assert pipe["versions_loaded"] == steps and pipe["dropped_stale"] == 0
    assert summary["totals"] == {"generated": steps * P * G, "trained": steps * P * G, "dropped_stale": 0,
                                 "flags": {"truncated": steps}}  # fmt: skip
    assert 0 <= pipe["learner_gpu_idle_share"] <= 1 and 0 <= pipe["rollout_gpu_idle_share"] <= 1
    assert (
        pipe["learn_generate_overlap_s"] > 0
    )  # learner trains B_t while the rollout generates B_t+1
    assert pipe["sync_bytes_per_version"] == [16]
    learner_steps = [json.loads(x) for x in (tmp_path / "learner/learner.jsonl").read_text().splitlines()]
    if bucket_mb:  # one broadcast per bucket: 1 bucket + 10 control/status collectives
        assert {e["sync_detail"]["collectives"] for e in learner_steps} == {11}
        assert {e["sync_detail"]["buckets"] for e in learner_steps} == {1}
    else:
        assert all("sync_detail" not in e for e in learner_steps)
    assert (
        summary["before_source"] == str(before) and summary["before"]["gsm8k"]["pass_at_1"] == 0.4
    )
    steps_out = [json.loads(x) for x in (tmp_path / "merged/steps.jsonl").read_text().splitlines()]
    assert [s["trained_version"] for s in steps_out] == [0, 0, 1, 2, 3]
    assert [s["policy_version"] for s in steps_out] == [1, 2, 3, 4, 5]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"data_sha256": "d", "args": {"seed": "18"}, "before": {}}))
    with pytest.raises(ValueError, match="same data and seed"):
        ag.merge(tmp_path / "learner", tmp_path / "rollout", tmp_path / "merged2", bad)
    from rl.local.summarize import step_stats

    assert step_stats(steps_out)["steps"] == steps  # grpo_run-compatible step records


def test_interval_helpers():
    assert ag._clip_total([[0, 2], [1, 3], [5, 6]], 0, 10) == 4
    assert ag._clip_total([[0, 2]], 1, 10) == 1
    assert ag._intersect_total([[0, 2], [4, 6]], [[1, 5]], 0, 10) == 2
