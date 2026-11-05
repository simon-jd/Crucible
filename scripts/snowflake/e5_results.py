"""Summarize E5 report.json files into the tables of docs/evidence/snowflake/e5-multinode.md; export redacted raw outputs.

  python scripts/snowflake/e5_results.py summary <report.json> [<report.json> ...] [--json out.json]
  python scripts/snowflake/e5_results.py export <run dir> <dest dir>

Uses the repository's definitions: scaling.analyze_nsys.ModelStructure / mfu (6N + 12LHDS per token, N without the
embedding) through scaling.experiment.scaling_metrics (efficiency = MFU / reference MFU, here against the 1-node run),
scaling.goodput.account_events (replayed from the logged events), cluster.health.parse_nccl.evaluate with
cluster/health/thresholds.yaml for the bandwidth gate, and cluster/drain's rule via the payload's verify_drain.

export copies report.json, driver.log and instance<i>.log, removing the PyPI proxy token and the account identifier
(SNOWFLAKE_ACCOUNT, if set, plus the generic patterns of the payload's redact), and fails if either remains.
"""
import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("e5_multinode", ROOT / "scripts/snowflake/payloads/e5_multinode.py")
e5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e5)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def fmt(s, digits=3):
    if not s:
        return "–"
    if s["n"] == 1:
        return f"{s['median']:.{digits}f}"
    return f"{s['median']:.{digits}f} [{s['min']:.{digits}f}–{s['max']:.{digits}f}]"


def phases(paths):
    """Phase results from driver report.json files (RESULT lines) or from the payload's per-phase <phase>.json files
    (the full results fetched from the stage with results.py)."""
    out = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        for r in data.get("results", []) if "results" in data else [data]:
            key = r.get("phase")
            if key:
                out.setdefault(key, []).append(r)
    return out


# ---------------------------------------------------------------- health

def health_table(res):
    rows = []
    for n in res["nodes"]:
        mm = n.get("matmul", {})
        disk = {Path(d["path"]).parts[1] if d.get("path") else "?": d.get("GBps") for d in n.get("disk_write", [])}
        ib = n.get("infiniband_ports") or {}
        rows.append({
            "host": n["host"], "gpu": (n.get("nvidia_smi") or "").split(", ")[1:2],
            "tools_present": sorted(k for k, v in n.get("tools", {}).items() if v),
            "tools_absent": sorted(k for k, v in n.get("tools", {}).items() if not v),
            "libdcgm": n.get("libs", {}).get("libdcgm"), "nccl_tests_on_disk": n.get("libs", {}).get("nccl_tests_on_disk"),
            "ib_sysfs": n.get("infiniband_sysfs"), "ib_ports": ib, "dev_infiniband": n.get("dev_infiniband"),
            "bf16_tflops": {k: statistics.median(v) for k, v in mm.get("tflops", {}).items()},
            "cupti": mm.get("cupti_probe"), "disk_GBps": disk, "eth0": n.get("eth0"),
            "smi_health": n.get("nvidia_smi_health"), "cpus": n.get("cpus")})
    lines = [("| Node | Present | Absent | IB sysfs (link layer, rate) | /dev/infiniband | bf16 matmul TFLOPS 4096 / 8192 "
              "| CUPTI kernel records | Disk write GB/s (/tmp, /dev/shm) |"),
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ib = "; ".join(f"{k}: {v.get('link_layer')}, {v.get('rate')}" for k, v in r["ib_ports"].items()) or "–"
        tf = r["bf16_tflops"]
        cupti = r["cupti"] or {}
        lines.append(f"| {r['host']} | {', '.join(r['tools_present']) or '–'} | {', '.join(r['tools_absent'])} | {ib} | "
                     f"{r['dev_infiniband'] or 'absent'} | {tf.get('bf16_4096', 0):.1f} / {tf.get('bf16_8192', 0):.1f} | "
                     f"{cupti.get('kernel_records', cupti.get('error', '–'))} | "
                     f"{r['disk_GBps'].get('tmp')} / {r['disk_GBps'].get('dev')} |")
    pip = next((n["pip_probe"] for n in res["nodes"] if "pip_probe" in n), {})
    pip_rows = {k: {"rc": v.get("rc"), "last": (v.get("out") or "").strip().splitlines()[-1:]} for k, v in pip.items()
                if isinstance(v, dict) and "rc" in v}
    return {"nodes": rows, "pip_probe": pip_rows}, "\n".join(lines)


# ---------------------------------------------------------------- collectives

def slowest_rows(run):
    """Per op and size, the slowest rank's measurement (every rank must finish an op for it to be done)."""
    out = []
    for rows in zip(*[r["bw"] for r in run["ranks"]]):
        slow = min(rows, key=lambda x: x["GBps"])
        out.append({**slow, "ms": max(x["ms"] for x in rows)})
    return out


def bandwidth_gate(runs, theoretical_busbw_gbs, thresholds_path=ROOT / "cluster/health/thresholds.yaml"):
    """cluster/health's burn-in rule (busbw >= min_busbw_fraction x site theory at >= min_size_bytes) per run.
    The 'theory' here is the pod interface's reported speed, not a measured or published peak."""
    from cluster.health.parse_nccl import Row, evaluate
    th = {**yaml.safe_load(Path(thresholds_path).read_text())["nccl"], "theoretical_busbw_gbs": theoretical_busbw_gbs}
    out = {}
    for run in runs:
        if "ranks" not in run:
            continue
        ok = all(r.get("check_ok") for r in run["ranks"])
        for op in ("all_reduce", "broadcast"):
            rows = [Row(size_bytes=x["bytes"], algbw_gbs=x["GBps"], busbw_gbs=x["busGBps"], time_us=x["ms"] * 1000,
                        wrong=0 if ok else None) for x in slowest_rows(run) if x["op"] == op]
            key = f"{run['world']}n/{run['variant']}/{op}"
            try:
                v = evaluate(rows, th)
                verdict = "pass" if not v else ",".join(sorted({x.code for x in v}))
            except ValueError as e:
                verdict = f"not evaluated: {e}"
            out.setdefault(key, []).append(verdict)
    return {"thresholds": th, "verdicts": out}


def collectives_table(res, theoretical_busbw_gbs=1.25):
    acc = {}
    for run in res.get("runs", []):
        if "ranks" not in run:
            continue
        for slow in slowest_rows(run):
            key = (run["world"], run["variant"], slow["op"], slow["bytes"])
            acc.setdefault(key, {"alg": [], "bus": []})
            acc[key]["alg"].append(slow["GBps"])
            acc[key]["bus"].append(slow["busGBps"])
    table = {f"{w}n/{v}/{op}/{b >> 20}MiB": {"algbw": stats(x["alg"]), "busbw": stats(x["bus"])}
             for (w, v, op, b), x in sorted(acc.items())}
    sizes = sorted({k[3] for k in acc})
    lines = ["| Nodes | Variant | Op | " + " | ".join(f"{b >> 20} MiB" for b in sizes) + " |",
             "|---|---|---|" + "---|" * len(sizes)]
    for w, v, op in sorted({k[:3] for k in acc}):
        cells = [fmt(stats(acc[(w, v, op, b)]["bus"])) if (w, v, op, b) in acc else "–" for b in sizes]
        lines.append(f"| {w} | {v} | {op} | " + " | ".join(cells) + " |")
    failed = [{k: r[k] for k in ("world", "variant", "pass")} for r in res.get("runs", []) if "ranks" not in r]
    transports = sorted({line for r in res.get("runs", []) for rk in r.get("ranks", []) for line in rk.get("nccl_log", [])
                         if "Using network" in line or ("NET/" in line and "Using" in line)})
    return {"busbw_algbw": table, "gate": bandwidth_gate(res.get("runs", []), theoretical_busbw_gbs),
            "failed_runs": failed, "transport_lines": transports}, "\n".join(lines)


# ---------------------------------------------------------------- scaling

def scaling_table(res):
    from scaling.analyze_nsys import ModelStructure
    from scaling.experiment import scaling_metrics
    peak = res.get("peak_flops", e5.A10_BF16_DENSE_PEAK)
    runs = [r for r in res["runs"] if all("global_tokens_per_s" in x for x in r["ranks"])]
    failed = [{"kind": r["kind"], "world": r["world"], "round": r["round"],
               "error": [x.get("log_tail", [])[-3:] for x in r["ranks"] if "global_tokens_per_s" not in x]}
              for r in res["runs"] if r not in runs]
    if not any(r["world"] == 1 for r in runs):
        return {"rows": [], "failed": failed}, "no successful 1-node reference run"
    st = next(x["structure"] for r in runs for x in r["ranks"] if "structure" in x)
    params = ModelStructure(st["active_parameters"], st["layers"], st["heads"], st["head_dim"])
    seq = runs[0]["plan"]["seq"]
    by = {}
    for r in runs:
        by.setdefault((r["kind"], r["world"]), []).append(r)
    ref_med = statistics.median(r["ranks"][0]["global_tokens_per_s"] for r in by[("weak", 1)])
    rows = []
    lines = [("| Scaling | Nodes | Global seq/step | Accum | Step s | Tokens/s/GPU | MFU | Scaling eff. | "
              "Compute s | Exposed comm s | Exposed comm share | All-reduce alone s |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (kind, world), rs in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tps = [r["ranks"][0]["global_tokens_per_s"] for r in rs]  # identical on every rank (slowest-rank window)
        effs = [scaling_metrics(t, params, seq, world, peak, ref_med, reference_gpus=1) for t in tps]
        splits = [r["ranks"][0]["split"] for r in rs]
        row = {"kind": kind, "world": world, "rounds": len(rs), "global_seq": rs[0]["plan"]["global_seq"],
               "accum": rs[0]["plan"]["accum"], "per_gpu_tokens_per_step": rs[0]["plan"]["per_rank_seq"] * seq,
               "step_s": stats([s["step_s"] for s in splits]), "tokens_per_s": stats(tps),
               "tokens_per_s_per_gpu": stats([m["tokens_per_s_per_gpu"] for m in effs]),
               "mfu": stats([m["mfu"] for m in effs]),
               "scaling_efficiency": stats([m["scaling_efficiency"] for m in effs]),
               "compute_s": stats([s.get("compute_s") for s in splits]),
               "exposed_comm_s": stats([s.get("exposed_comm_s") for s in splits]),
               "exposed_comm_share": stats([s.get("exposed_comm_share") for s in splits]),
               "comm_alone_s": stats([s.get("comm_alone_s") for s in splits]),
               "comm_alone_share": stats([s.get("comm_alone_share") for s in splits]),
               "comm_alone_bytes": rs[0]["ranks"][0].get("comm_alone", {}).get("bytes"),
               "final_loss": [r["ranks"][0]["losses"][-1] for r in rs],
               "peak_mem_gib": max(x.get("peak_mem_gib", 0) for r in rs for x in r["ranks"])}
        rows.append(row)
        lines.append(f"| {kind} | {world} | {row['global_seq']} | {row['accum']} | {fmt(row['step_s'], 2)} | "
                     f"{fmt(row['tokens_per_s_per_gpu'], 0)} | {fmt(row['mfu'])} | {fmt(row['scaling_efficiency'], 2)} | "
                     f"{fmt(row['compute_s'], 2)} | {fmt(row['exposed_comm_s'], 2)} | "
                     f"{fmt(row['exposed_comm_share'], 2)} | {fmt(row['comm_alone_s'], 2)} |")
    return {"structure": st, "peak_flops": peak, "reference_tokens_per_s": ref_med, "rows": rows,
            "failed": failed}, "\n".join(lines)


# ---------------------------------------------------------------- drain and fault

def drain_summary(res):
    out = {}
    for feed, log in res["logs"].items():
        v = e5.verify_drain(log, res["slow_node"], ratio=res["rules"]["step_time_ratio"],
                            consecutive=res["rules"]["consecutive_checks"],
                            min_nodes=res["rules"]["min_comparison_nodes"])
        indeterminate = sum(1 for e in log for a in e["actions"] if "indeterminate" in a["reason"])
        out[feed] = {**v, "samples": len(log), "indeterminate_alerts": indeterminate,
                     "executor_statuses": sorted({x["status"] for e in log for x in e["executor"]})}
    nodes = list(res["logs"]["local"][0]["times"])
    lines = ["| Sample | Step | Load on | " + " | ".join(nodes) + " | Alerts (local) | Alerts (sync) |",
             "|---|---|---|" + "---|" * (len(nodes) + 2)]
    for loc, syn in zip(res["logs"]["local"], res["logs"]["sync"]):
        med = statistics.median(loc["times"].values())
        cells = [f"{t:.3f} / {syn['times'][n]:.2f}" + (" **>1.2×**" if t > med * res["rules"]["step_time_ratio"] else "")
                 for n, t in loc["times"].items()]
        lines.append(f"| {loc['sample']} | {loc['step']} | {'yes' if loc['load_on'] else ''} | " + " | ".join(cells)
                     + f" | {', '.join(loc['alerts']) or '–'} | {', '.join(syn['alerts']) or '–'} |")
    return out, "\n".join(lines)


def fault_summary(res):
    from scaling.goodput import account_events
    acct = account_events(res["events"])
    ev = res["events"]
    t0 = ev[0]["at"]
    steps = [e for e in ev if e["kind"] == "step_complete"]
    saves = [e for e in ev if e["kind"] == "checkpoint_saved"]
    timings = next((r["timings"] for r in res.get("attempt1", []) if r.get("timings")), {})
    fail_at = next(e["at"] for e in ev if e["kind"] in ("failure", "operation_failed"))
    restore = next(e for e in ev if e["kind"] == "restore_start")
    out = {"accounting": acct, "logged_accounting_matches": acct == res.get("accounting"),
           "step_seconds_median": statistics.median(e["at"] - e["started_at"] for e in steps),
           "save_seconds": [round(e["at"] - e["started_at"], 2) for e in saves],
           "save_bytes_attempt1": timings.get("save_bytes"),
           "restore_seconds": acct["recovery_seconds"],
           "failure_to_restore_start_s": round(restore["at"] - fail_at, 2),
           "kill_to_failure_record_s": round(fail_at - res["killed"]["at"], 2),
           "first_event_to_first_step_s": round(steps[0]["started_at"] - t0, 2),
           "events": [(round(e["at"] - t0, 2), e["kind"], e.get("step"), e.get("checkpoint_id", "").rsplit("/", 1)[-1])
                      for e in ev if e["kind"] != "step_complete"]}
    losses = {}
    for p in res.get("progress", []):
        losses.setdefault(p["step"], {})[p["attempt"]] = p["loss"]
    out["replayed_step_losses"] = {s: v for s, v in losses.items() if len(v) > 1}
    return out


def fault_table(runs):
    data, lines = [], [("| Round | Checkpoint store | Goodput | Useful s | Wall s | Lost steps (s) | Save s | "
                        "Restore s | Failure → restore start s |"), "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        f = {"round": r.get("round"), "store": r.get("store", "disk"), **fault_summary(r)}
        data.append(f)
        acct = f["accounting"]
        lines.append(f"| {f['round']} | {f['store']} | {acct['goodput']:.3f} | {acct['useful_steps_time']:.1f} | "
                     f"{acct['wall_time']:.1f} | {acct['lost_steps']} ({acct['lost_steps_time']:.1f}) | "
                     f"{statistics.median(f['save_seconds']):.1f} | {acct['recovery_seconds'][0]:.1f} | "
                     f"{f['failure_to_restore_start_s']:.1f} |")
    return data, "\n".join(lines)


def summary(a):
    ph = phases(a.reports)
    out = {}
    for name, fn in (("health", health_table), ("collectives", collectives_table), ("scaling", scaling_table),
                     ("drain", drain_summary)):
        for res in ph.get(name, []):
            if "error" in res:
                print(f"## {name}: error {res['error'][:300]}\n")
                continue
            data, md = fn(res)
            out.setdefault(name, []).append(data)
            print(f"## {name}\n\n{md}\n")
    for res in ph.get("fault", []):
        runs = [r for r in res.get("runs", [res]) if "events" in r and r.get("killed")]
        failed = [{k: r.get(k) for k in ("round", "store", "error")} for r in res.get("runs", [res]) if r not in runs]
        data, md = fault_table(runs)
        out.setdefault("fault", []).append({"runs": data, "failed": failed})
        print(f"## fault\n\n{md}\n\nfailed: {failed}\n")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------- export

LEAKS = (re.compile(r"eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}"), re.compile(r"//[^/@\s'\"<]+:[^/@\s'\"<]+@"))


def export(src, dest, secrets=None):
    secrets = [s for s in (secrets if secrets is not None else [os.environ.get("SNOWFLAKE_ACCOUNT", "")]) if s]
    dest.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name in ("report.json", "driver.log", *sorted(p.name for p in src.glob("instance*.log"))):
        if not (src / name).exists():
            continue
        text = e5.redact((src / name).read_text(errors="replace"), secrets)
        bad = [p.pattern for p in LEAKS if p.search(text)] + [
            "account" for s in secrets if s.lower() in text.lower()]
        if bad:
            raise SystemExit(f"{name}: still contains {bad}; not exported")
        (dest / name).write_text(text)
        sizes[name] = len(text.encode())
    return sizes


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("reports", nargs="+")
    s.add_argument("--json", type=Path)
    x = sub.add_parser("export")
    x.add_argument("src", type=Path)
    x.add_argument("dest", type=Path)
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a)
    else:
        print(json.dumps(export(a.src, a.dest)))


if __name__ == "__main__":
    main()
