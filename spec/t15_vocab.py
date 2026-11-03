"""One EAGLE-3 draft vocabulary mapping over all T15 training conversations (CPU, phase 7).

Offline features are generated batch by batch, and prepare_hidden_states.py would derive
a mapping from each batch alone. The run trains one draft across all batches, so its
mapping comes once from the whole regenerated set, with the same SpecForge functions
prepare_hidden_states.py uses (safe_conversations_generator, build_eagle3_dataset,
generate_vocab_mapping_file); training reads it via model.vocab_mapping_path.
Runs in .venv-specforge:
  python -m spec.t15_vocab REGEN.jsonl TARGET_DIR DRAFT_CONFIG.json MAX_LENGTH OUT.pt
"""

import hashlib
import json
import os
import sys
from pathlib import Path


def main(argv):
    regen, target, draft_config, max_length, out = argv
    from datasets import Dataset
    from specforge.data.preprocessing import build_eagle3_dataset, generate_vocab_mapping_file
    from specforge.utils import load_tokenizer, safe_conversations_generator

    cache = Path(".cache/t15/sf-cache")
    dataset = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": regen},
        cache_dir=str(cache / "hf_dataset"),
    )
    key = hashlib.md5(f"{regen}-{max_length}-qwen-{target}-vocab".encode()).hexdigest()
    processed = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=load_tokenizer(target),
        chat_template="qwen",
        max_length=int(max_length),
        cache_dir=str(cache / "processed_dataset"),
        cache_key=key,
        num_proc=8,
    )
    draft = json.loads(Path(draft_config).read_text())
    out = Path(out)
    path = generate_vocab_mapping_file(
        processed,
        target_vocab_size=draft["vocab_size"],
        draft_vocab_size=draft["draft_vocab_size"],
        cache_dir=str(out.parent),
        cache_key=out.stem,
    )
    print(json.dumps({"samples": len(processed), "mapping": path, "bytes": os.path.getsize(path)}))


if __name__ == "__main__":
    main(sys.argv[1:])
