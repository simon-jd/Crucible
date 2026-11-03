"""Sealed FD loader and all-coordinate verifier. Actual vLLM/GPU UNVERIFIED.

Pure module import; vLLM/Torch imports occur only in owned consumers.
"""

import multiprocessing.context
import multiprocessing.reduction
import os

from rl.learner import handoff_stream as stream
from rl.sync.ram_weights import read_weights


def _detach(wrapper):
    return wrapper.detach()


class SpawnFD:
    """No eager duplication/resource_sharer; pinned POSIX spawn inherits same FD.

    Python3.12 popen_spawn_posix.duplicate_for_child only appends to pass_fds.
    It does not dup in parent. Parent therefore retains its original lease only.
    Ambiguous Process.start still retains the entire lease and Process handle.
    """

    def __init__(self, fd, receipt):
        self.fd, self.receipt = fd, receipt
        self.phase = "reserved"

    def __reduce__(self):
        self.phase = "reducing"
        popen = multiprocessing.context.get_spawning_popen()
        if popen is None or type(popen).__module__ != "multiprocessing.popen_spawn_posix":
            raise RuntimeError("FD reduction requires owned POSIX spawn; broker forbidden")
        self.receipt.validate(self.fd)
        wrapped = multiprocessing.reduction.DupFd(self.fd)
        self.phase = "inherited"
        return _detach, (wrapped,)


def iter_weights(fd, checkpoint):
    import torch

    records, start = read_weights(fd, checkpoint)
    for record in records:
        tensor = torch.empty(
            record["shape"], dtype=getattr(torch, record["dtype"][6:]), device="cpu"
        )
        stream._read_record(fd, tensor, start + record["offset"])
        for part in tensor.reshape(-1).split(16384):
            if not bool(torch.isfinite(part).all()):
                raise ValueError("nonfinite inference content")
        yield record["name"], tensor
        del tensor


def persistent_state(model):
    tensors = dict(model.named_parameters(remove_duplicate=False))
    for name, value in model.named_buffers(remove_duplicate=False):
        parent, _, child = name.rpartition(".")
        if child not in model.get_submodule(parent)._non_persistent_buffers_set:
            if name in tensors:
                raise ValueError("duplicate model state name")
            tensors[name] = value
    return tensors


def target_views(model, records):
    """Narrow unquantized TP1 Qwen2 mapping, shared with CPU tensor fixtures."""
    tensors = persistent_state(model)
    views, covered, padding = {}, set(), []
    for record in records:
        source, shape = record["name"], record["shape"]
        target, offset, extent = source, None, None
        for shard in ("q", "k", "v"):
            needle = f".self_attn.{shard}_proj."
            if needle in source:
                parent = source.split(needle)[0] + ".self_attn"
                module = model.get_submodule(parent)
                sizes = [module.q_size, module.kv_size, module.kv_size]
                index = ("q", "k", "v").index(shard)
                offset, extent = sum(sizes[:index]), sizes[index]
                target = source.replace(needle, ".self_attn.qkv_proj.")
        for shard, index in (("gate", 0), ("up", 1)):
            needle = f".mlp.{shard}_proj."
            if needle in source:
                target = source.replace(needle, ".mlp.gate_up_proj.")
                extent = model.config.intermediate_size
                offset = index * extent
        if target not in tensors:
            raise ValueError(f"unsupported destination: {source}")
        value = tensors[target].detach()
        if not value.is_contiguous() or str(value.dtype) != record["dtype"]:
            raise ValueError("target contiguous/dtype")
        if offset is not None:
            expected = (
                model.get_submodule(source.split(".self_attn.")[0] + ".self_attn").q_size
                + 2 * model.get_submodule(source.split(".self_attn.")[0] + ".self_attn").kv_size
                if ".self_attn." in source
                else 2 * extent
            )
            if value.shape[0] != expected:
                raise ValueError("packed target extent")
            value = value.narrow(0, offset, extent)
        elif source == "model.embed_tokens.weight" and list(value.shape) != shape:
            if (
                len(shape) != 2
                or value.ndim != 2
                or value.shape[1] != shape[1]
                or value.shape[0] < shape[0]
                or shape[0] != model.config.vocab_size
            ):
                raise ValueError("embedding padding shape")
            padding.append(value[shape[0] :])
            value = value[: shape[0]]
        if list(value.shape) != shape:
            raise ValueError("target shape")
        views[source] = value
        covered.add(target)
    # Require every packed shard, not merely the packed parameter's name.
    names = set(views)
    for target in covered:
        if ".qkv_proj." in target:
            required = {target.replace(".qkv_proj.", f".{s}_proj.") for s in ("q", "k", "v")}
            if not required <= names:
                raise ValueError("missing qkv shard")
        if ".gate_up_proj." in target and (
            not {target.replace(".gate_up_proj.", f".{s}_proj.") for s in ("gate", "up")} <= names
        ):
            raise ValueError("missing gate/up shard")
    if "lm_head.weight" in tensors and "model.embed_tokens.weight" in tensors:
        head, embedding = tensors["lm_head.weight"], tensors["model.embed_tokens.weight"]
        if not getattr(model.config, "tie_word_embeddings", False) or head is not embedding:
            raise ValueError("exact tied head required")
        covered.add("lm_head.weight")
    if covered != set(tensors):
        raise ValueError("uncovered parameters/persistent buffers")
    return views, padding


def verify_model(model, fd, checkpoint):
    import torch

    records, start = read_weights(fd, checkpoint)
    views, padding = target_views(model, records)
    devices = {str(t.device) for t in views.values()}
    if devices - {"cpu", "cuda:0"}:
        raise ValueError("unsupported verifier device")
    if "cuda:0" in devices:
        torch.cuda.synchronize(0)
    for record in records:
        flat = views[record["name"]].reshape(-1)
        position = start + record["offset"]
        for part in flat.split(16384):
            host = part.to(device="cpu")
            raw = stream._raw(host)
            expected = os.pread(fd, len(raw), position)
            if len(expected) != len(raw) or raw != expected:
                raise ValueError(f"loaded coordinate mismatch: {record['name']}")
            position += len(raw)
            del raw, host, expected
    for value in padding:
        for part in value.reshape(-1).split(16384):
            if not bool((part == 0).all()):
                raise ValueError("nonzero embedding padding")
    checkpoint.receipt.validate(fd)
    return checkpoint.ack()


def install_loader(fd, checkpoint):
    """Called only inside actual owned vLLM worker; no CPU test imports vLLM."""
    from vllm.model_executor.model_loader import BaseModelLoader, register_model_loader
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

    used = False

    @register_model_loader("crucible_ram_v1")
    class RamLoader(BaseModelLoader):
        def load_model(self, vllm_config, model_config):
            verify_configuration(vllm_config)
            return super().load_model(vllm_config, model_config)

        def download_model(self, model_config):
            raise RuntimeError("RAM loader forbids downloads")

        def load_weights(self, model, model_config):
            nonlocal used
            if used or type(model) is not Qwen2ForCausalLM or model_config.quantization is not None:
                raise ValueError("one-use unquantized Qwen2 loader")
            used = True
            iterator = iter_weights(fd, checkpoint)
            loaded = model.load_weights(iterator)
            if next(iterator, None) is not None or not loaded:
                raise ValueError("loader did not consume all weights")
            verify_model(model, fd, checkpoint)

    return "crucible_ram_v1"


def verify_runtime(llm):
    from vllm.v1.engine.core_client import InprocClient
    from vllm.v1.executor.abstract import UniProcExecutor

    engine = llm.llm_engine
    if (
        type(engine.engine_core) is not InprocClient
        or type(engine.model_executor) is not UniProcExecutor
    ):
        raise ValueError("RAM loader requires V1 in-process uni executor")
    verify_configuration(engine.vllm_config)


def verify_configuration(config):
    parallel = config.parallel_config
    if (
        parallel.tensor_parallel_size,
        parallel.pipeline_parallel_size,
        parallel.data_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("RAM loader requires world1")
    if parallel.distributed_executor_backend != "uni" or parallel.worker_extension_cls:
        raise ValueError("RAM loader requires unextended uni worker")
    if (
        config.quant_config is not None
        or config.lora_config is not None
        or config.speculative_config is not None
        or config.cache_config.cpu_offload_gb
    ):
        raise ValueError("unsupported RAM runtime configuration")
