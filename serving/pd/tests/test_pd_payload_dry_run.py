"""CPU dry run of the E4 payload's layout loop with fake vLLM servers and the real Go replayer."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = ROOT / "scripts/snowflake/payloads/e4_pd.py"


def load_payload():
    spec = importlib.util.spec_from_file_location("e4_pd_payload", PAYLOAD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Calibration replays the first calibration requests at 0.25x their source rate (about 75 s), so only once.
@pytest.mark.parametrize(("layout", "calibrate"), [("colo", True), ("dp2", False), ("pd-shm", False)])
def test_layout_loop_runs_calibration_and_evaluation_against_fake_servers(layout, calibrate, tmp_path, monkeypatch):
    pytest.importorskip("aiohttp")
    mod = load_payload()
    if not mod.REPLAY.exists() or not mod.IDS.exists():
        pytest.skip("needs .cache/e4/bin/replay and .cache/e4/qwen3-8b-awq-ids.json")
    monkeypatch.setattr(mod, "WORK", tmp_path)
    monkeypatch.setattr(mod, "SERVER_MODULE", "serving.pd.tests.fake_vllm")
    a = SimpleNamespace(decode_pool_gb=1, decode_buffer_bytes=1e8, shm_default=False, max_pd_inflight=8,
                        levels=[1, 2], rounds=2, smoke=6, calibrate=calibrate)
    s = mod.run_layout(layout, a, "/nonexistent-model", None, deadline=float("inf"))
    assert s["stop_reason"] is None, s["stop_reason"]
    assert len(s["calibration"]) == (3 if calibrate else 0) and len(s["evaluation"]) == 4
    assert [(t["round"], t["level"]) for t in s["evaluation"]] == [(0, 1), (0, 2), (1, 2), (1, 1)]
    for t in s["calibration"] + s["evaluation"]:
        assert t["sender_valid"] and t["ok"] == t["requests"] == 6 and t["idle_after"]
        assert "kv" not in t  # the fake servers have no KV probe
    out = tmp_path / "out" / layout
    saved = json.loads((out / "summary.json").read_text())
    assert saved["plan"]["proxy_mode"] == ("pd" if layout.startswith("pd") else "dp")
    recs = [json.loads(x) for x in (out / "proxy.jsonl").read_text().splitlines()]
    assert len(recs) == 4 + (7 if calibrate else 4) * 6 and all(r["status"] == 200 for r in recs)
    if layout == "dp2":
        assert {r["backend"] for r in recs} == {0, 1}
    if layout == "pd-shm":
        assert all("t_prefill_done" in r for r in recs)
    assert sorted(p.name for p in out.glob("server-*.log"))
