"""Sync microbenchmark bookkeeping (CPU only; the transfers themselves run on GPUs)."""

import pytest

from rl.sync.sync_bench import summarize, variant_names


def test_variants_cover_both_baselines_every_bucket_and_its_raw_floor():
    assert variant_names([16, 256]) == [
        "per_tensor_sum64", "per_tensor_sha256", "bucket_16M", "bucket_256M", "raw_16M", "raw_256M",
    ]  # fmt: skip


def test_summary_skips_warmup_and_takes_the_slower_rank_per_pass():
    records = [
        {"pass": 0, "variant": "bucket_64M", "seconds": 9.0, "receiver_seconds": 9.0},
        {"pass": 1, "variant": "bucket_64M", "seconds": 0.50, "receiver_seconds": 0.52,
         "detail": {"collectives": 26, "buckets": 15, "chunks": 60, "pack_s": 0.01, "transfer_s": 0.4,
                    "hash_wait_s": 0.02, "verify_commit_s": 0.05}},
        {"pass": 2, "variant": "bucket_64M", "seconds": 0.70, "receiver_seconds": 0.60,
         "detail": {"collectives": 26, "buckets": 15, "chunks": 60, "pack_s": 0.03, "transfer_s": 0.5,
                    "hash_wait_s": 0.02, "verify_commit_s": 0.07}},
        {"pass": 3, "variant": "bucket_64M", "seconds": 0.40, "receiver_seconds": 0.45,
         "detail": {"collectives": 26, "buckets": 15, "chunks": 60, "pack_s": 0.02, "transfer_s": 0.3,
                    "hash_wait_s": 0.02, "verify_commit_s": 0.06}},
        {"pass": 1, "variant": "raw_64M", "seconds": 0.30, "receiver_seconds": 0.31, "detail": {"collectives": 15}},
    ]  # fmt: skip
    rows = summarize(records, 1_000_000_000)
    bucket = rows["bucket_64M"]
    assert bucket["source_s"] == [0.50, 0.70, 0.40]
    assert bucket["median_s"] == pytest.approx(0.52)
    assert (bucket["min_s"], bucket["max_s"]) == (0.45, 0.70)
    assert bucket["GBps"] == pytest.approx(1 / 0.52)
    assert bucket["collectives"] == 26 and bucket["buckets"] == 15
    assert bucket["median_transfer_s"] == pytest.approx(0.4)
    assert rows["raw_64M"]["median_s"] == pytest.approx(0.31)
    assert rows["raw_64M"]["collectives"] == 15
