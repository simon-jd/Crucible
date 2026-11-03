"""The public EAGLE-3 draft gets an SGLang-loadable view without changing its weights."""

import json

from spec.t15_public_draft import make_view


def test_make_view_renames_only_the_architecture_and_links_the_rest(tmp_path):
    src, out = tmp_path / "snap", tmp_path / "view"
    src.mkdir()
    cfg = {"architectures": ["Eagle3LlamaForCausalLM"], "draft_vocab_size": 32000}
    (src / "config.json").write_text(json.dumps(cfg))
    (src / "model.safetensors").write_bytes(b"w")
    report = make_view(src, out)
    assert json.loads((out / "config.json").read_text()) == {
        "architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 32000}
    assert (out / "model.safetensors").is_symlink()
    assert (out / "model.safetensors").resolve() == (src / "model.safetensors").resolve()
    assert report["architecture"] == ["Eagle3LlamaForCausalLM", "LlamaForCausalLMEagle3"]
