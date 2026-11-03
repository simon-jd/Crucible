"""T11 data for verl: .cache/t11/data.json -> train.parquet + val.parquet.

Train rows are written in exactly the order grpo_run consumes them: step s
(0-based) trains on data["train"][(8*s + i) % len(train)], i = 0..7, so for
100 steps x 8 prompts the parquet holds 800 rows, row k = train[k % len(train)].

Order. With data.shuffle=false verl 0.9.1 builds a SequentialSampler over the
dataset (create_rl_sampler), batches it by data.train_batch_size=8 with
drop_last, and does not filter (data.filter_overlong_prompts=false), so verl's
step s gets parquet rows 8s..8s+7 = grpo_run's step-s prompts, and the 800 rows
are exactly one epoch of 100 steps. Inside a step verl repeats each prompt n=8
times (interleaved); grpo_run also groups the 8 samples of a prompt together.

Prompt. The "prompt" column is grpo_run.prompt_messages(question); verl's
RLHFDataset applies the tokenizer chat template with add_generation_prompt=True,
as grpo_run does. reward_model.ground_truth is the reference answer passed to
the verifier; extra_info.index is verl's row index.

The val parquet exists only because verl always builds a val dataset from
data.val_files; validation itself is disabled in the launcher.

usage: .venv-verl/bin/python -m rl.verl.prepare_verl_data [--data D] [--out DIR]
"""

import argparse
import json
from pathlib import Path

from rl.local.grpo_run import prompt_messages

DATA_SHA256 = "bb491e5505e00da7d81364866d5d5db06bae484d51b713b68d140c15069ac705"


def verl_row(row, index, split):
    return {
        "data_source": row.get("source", "gsm8k"),
        "prompt": prompt_messages(row["question"]),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": row["answer"]},
        "extra_info": {"id": row["id"], "index": index, "split": split},
    }


def train_rows(data, steps=100, per_step=8):
    train = data["train"]
    return [verl_row(train[k % len(train)], k, "train") for k in range(steps * per_step)]


def val_rows(data, count=8):
    return [verl_row(r, k, "val") for k, r in enumerate(data["gsm8k_eval"][:count])]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path(".cache/t11/data.json"))
    p.add_argument("--out", type=Path, default=Path(".cache/t11/verl-data"))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--prompts-per-step", type=int, default=8)
    args = p.parse_args(argv)

    data = json.loads(args.data.read_text())
    if data["sha256"] != DATA_SHA256:
        raise SystemExit(f"{args.data}: sha256 {data['sha256']} != T11 base run {DATA_SHA256}")
    import pandas as pd  # .venv-verl only

    args.out.mkdir(parents=True, exist_ok=True)
    train = train_rows(data, args.steps, args.prompts_per_step)
    val = val_rows(data)
    pd.DataFrame(train).to_parquet(args.out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out / "val.parquet", index=False)
    manifest = {
        "data_sha256": data["sha256"],
        "steps": args.steps,
        "prompts_per_step": args.prompts_per_step,
        "train_rows": len(train),
        "val_rows": len(val),
        "first_train_ids": [r["extra_info"]["id"] for r in train[: args.prompts_per_step]],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
