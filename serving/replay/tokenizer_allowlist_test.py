import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "allowlist", Path(__file__).with_name("tokenizer_allowlist.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, vocab=None):
    revision = "a" * 40
    path = tmp_path / "models--fixture--model" / "snapshots" / revision
    path.mkdir(parents=True)
    docs = {
        "tokenizer.json": {
            "model": {"vocab": vocab or {"a": 0, "b": 1, "c": 2, "d": 3}},
            "added_tokens": [{"id": 3}],
        },
        "tokenizer_config.json": {},
        "config.json": {"vocab_size": 4, "eos_token_id": 2},
    }
    for name, doc in docs.items():
        (path / name).write_text(json.dumps(doc))
    return path, revision


def test_reproducible_exclusion(tmp_path):
    path, revision = fixture(tmp_path)
    result = module.build(path, "fixture/model", revision)
    assert result["token_ids"] == [0, 1]
    assert len(result["file_sha256"]) == 3
    assert module.build(path, "fixture/model", revision) == result
    with pytest.raises(ValueError, match="model path"):
        module.build(path, "other/model", revision)


@pytest.mark.parametrize("vocab", [{"a": True}, {"a": -1}, {"a": 4}, {"a": 1, "b": 1}])
def test_invalid_ids(tmp_path, vocab):
    path, revision = fixture(tmp_path, vocab)
    with pytest.raises(ValueError):
        module.build(path, "fixture/model", revision)


def test_revision_and_size_bound(tmp_path, monkeypatch):
    path, revision = fixture(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        module.build(path, "fixture/model", "main")
    monkeypatch.setattr(module, "LIMIT", 3)
    with pytest.raises(ValueError, match="bound"):
        module.build(path, "fixture/model", revision)


def test_explicit_special_names(tmp_path):
    path, revision = fixture(tmp_path)
    (path / "config.json").write_text(json.dumps({"vocab_size": 8, "eos_token_id": 2}))
    (path / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {str(i): i for i in range(8)}}, "added_tokens": [{"id": 3}]})
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"unk_token": {"content": "4"}, "added_tokens_decoder": {"5": {}}})
    )
    assert module.build(path, "fixture/model", revision)["token_ids"] == [0, 1, 6, 7]


def test_words_mode_keeps_space_prefixed_lowercase_words_with_texts(tmp_path):
    vocab = {"Ġthe": 0, "Ġof": 1, "Ġmodel": 2, "Ġdata": 3, "the": 4, "ĠThe": 5, "Ġx9": 6, "Ġwell": 7}
    path, revision = fixture(tmp_path, vocab)
    (path / "config.json").write_text(json.dumps({"vocab_size": 8, "eos_token_id": 7}))
    result = module.build(path, "fixture/model", revision, words=True)
    # "Ġof" is too short, "the"/"ĠThe"/"Ġx9" are not space+lowercase, 3 is added, 7 is EOS.
    assert result["token_ids"] == [0, 2]
    assert result["token_texts"] == [" the", " model"]
    assert "words" in result["policy"]
    assert "token_texts" not in module.build(path, "fixture/model", revision)
