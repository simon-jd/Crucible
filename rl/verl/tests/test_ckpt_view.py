import json

import pytest

from rl.verl import ckpt_view as cv


def test_view_links_trained_weights_and_base_metadata(tmp_path):
    ckpt, base, out = tmp_path / "ckpt", tmp_path / "base", tmp_path / "view"
    ckpt.mkdir()
    base.mkdir()
    # transformers 5 writes rope_parameters and drops rope_theta; 4.x then defaults to 1e4
    (ckpt / "config.json").write_text(json.dumps({"rope_parameters": {"rope_theta": 1e6}}))
    (ckpt / "model.safetensors").write_bytes(b"trained")
    (ckpt / "tokenizer.json").write_text("{}")
    (base / "config.json").write_text(json.dumps({"rope_theta": 1e6, "model_type": "qwen2"}))
    (base / "model.safetensors").write_bytes(b"base")
    (base / "tokenizer.json").write_text('{"base": 1}')
    cv.make_view(ckpt, base, out)
    assert json.loads((out / "config.json").read_text())["rope_theta"] == 1e6
    assert (out / "model.safetensors").resolve() == (ckpt / "model.safetensors").resolve()
    # transformers 5's tokenizer files do not load in 4.55; the tokenizer is not trained
    assert (out / "tokenizer.json").resolve() == (base / "tokenizer.json").resolve()


def test_view_refuses_a_different_architecture(tmp_path):
    ckpt, base = tmp_path / "ckpt", tmp_path / "base"
    ckpt.mkdir()
    base.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "qwen2", "num_hidden_layers": 24}))
    (base / "config.json").write_text(json.dumps({"model_type": "qwen2", "num_hidden_layers": 28}))
    with pytest.raises(ValueError):
        cv.make_view(ckpt, base, tmp_path / "view")
