"""Lazy vLLM 0.10.2 V1 backend, isolated in an owned spawn child.

V1 in-process core + uni executor avoids another persistent engine service.
The spawn process's exit/join is the CUDA-context ownership boundary. The
outer GPU guard additionally reaps all descendants before releasing flock.
"""

import multiprocessing as mp
import os
import socket
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS

from rl.rollout._rpc import FramedSocket, check_deadline


def _validate_owned_bf16(checkpoint):
    """The actual loader supports BF16 floating state; generic stores preserve any dtype."""
    if not (Path(checkpoint.model_path) / "manifest.json").exists():
        return False  # Legacy caller-owned Task7 BF16 contract.
    import torch

    from rl.sync.version import verify_checkpoint

    manifest = verify_checkpoint(checkpoint)
    for name, (_shape, dtype_name) in manifest["contract"]["schema"].items():
        dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype) or (
            (dtype.is_floating_point or dtype.is_complex) and dtype != torch.bfloat16
        ):
            raise ValueError(f"actual vLLM loader requires BF16 floating schema: {name}")
    return True


def _configure_worker_environment(runtime_root):
    """Constrain backend caches before importing CUDA consumers, including API callers."""
    import tempfile

    names = {
        "CUDA_CACHE_PATH": "cuda",
        "XDG_CACHE_HOME": "xdg",
        "VLLM_CACHE_ROOT": "vllm",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "torch",
        "TORCH_HOME": "torch-home",
        "CUPY_CACHE_DIR": "cupy",
        "NUMBA_CACHE_DIR": "numba",
        "TMPDIR": "tmp",
    }
    environment = {}
    for key, child in names.items():
        path = runtime_root / child
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    os.environ.update(environment)
    tempfile.tempdir = environment["TMPDIR"]
    return environment


def _serve(sock, checkpoint, config, diagnostic_logits=False, ram_fd=None, runtime_root=None):
    connection = FramedSocket(sock)
    llm = None
    cleanup = None
    try:
        from rl.sync.ram_weights import RamPolicyCheckpoint

        ram = isinstance(checkpoint, RamPolicyCheckpoint)
        if ram != (ram_fd is not None):
            raise ValueError("explicit RAM capability required")
        if ram:
            from rl.sync.ram_weights import verify_qwen_assets

            checkpoint.receipt.validate(ram_fd)
            verify_qwen_assets(checkpoint)
            root = Path(__file__).resolve().parents[2]
            runtime_root = Path(runtime_root)
            if root not in runtime_root.resolve().parents or runtime_root != runtime_root.resolve():
                raise ValueError("owned project RAM runtime root")
        cache_environment = _configure_worker_environment(
            runtime_root if ram else Path(__file__).resolve().parents[2] / ".cache/task7/runtime"
        )
        owned = False if ram else _validate_owned_bf16(checkpoint)
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if ram:
            os.environ.update(
                {
                    "HF_HOME": "/home/postedism/.cache/huggingface",
                    "HF_HUB_CACHE": "/home/postedism/.cache/huggingface/hub",
                    "HUGGINGFACE_HUB_CACHE": "/home/postedism/.cache/huggingface/hub",
                }
            )
            os.environ.pop("TRANSFORMERS_CACHE", None)
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup = cleanup_dist_env_and_memory
        import torch
        import vllm

        if vllm.__version__ != "0.10.2" or torch.__version__ != "2.8.0+cu128":
            raise RuntimeError("rollout requires pinned vLLM0.10.2/Torch2.8.0+cu128")
        if not Path(checkpoint.model_path).is_dir():
            raise ValueError("local checkpoint directory does not exist")
        extra = {}
        if ram:
            from rl.rollout.ram_loader import install_loader

            extra = {
                "load_format": install_loader(ram_fd, checkpoint),
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "data_parallel_size_local": 1,
                "worker_cls": "vllm.v1.worker.gpu_worker.Worker",
                "worker_extension_cls": "",
                "quantization": None,
                "speculative_config": None,
                "cpu_offload_gb": 0,
                "swap_space": 0,
                "enable_lora": False,
            }
        llm = LLM(
            model=checkpoint.model_path,
            tokenizer=checkpoint.model_path,
            revision=checkpoint.base_revision if ram else checkpoint.revision,
            tokenizer_revision=checkpoint.base_revision if ram else checkpoint.revision,
            trust_remote_code=False,
            dtype="bfloat16",
            seed=config.seed,
            gpu_memory_utilization=0.80,
            max_model_len=config.max_model_len,
            enforce_eager=True,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_num_seqs=16,
            generation_config="vllm",
            logprobs_mode="raw_logits" if diagnostic_logits else "raw_logprobs",
            max_logprobs=-1 if diagnostic_logits else 20,
            max_num_batched_tokens=config.max_model_len * 16,
            **extra,
        )
        if not llm.llm_engine.__class__.__module__.startswith("vllm.v1."):
            raise RuntimeError("expected V1 raw-logprob engine")
        if owned:
            from rl.sync.version import verify_checkpoint

            verify_checkpoint(checkpoint)
        loaded = asdict(checkpoint)
        if ram:
            from rl.rollout.ram_loader import verify_model, verify_runtime

            verify_runtime(llm)
            answers = llm.apply_model(lambda model: verify_model(model, ram_fd, checkpoint))
            if answers != [checkpoint.ack()]:
                raise ValueError("RAM all-coordinate ACK")
            verify_qwen_assets(checkpoint)
            loaded = answers[0]
        connection.send(["ready", [loaded, cache_environment]], time.monotonic() + 300)
        while True:
            request = connection.receive(time.monotonic() + 4 * 60 * 60)
            if request == ["close"]:
                break
            if (
                diagnostic_logits
                and isinstance(request, list)
                and len(request) == 2
                and request[0] == "full_logits"
            ):
                prompts = request[1]
                if (
                    not isinstance(prompts, list)
                    or not 1 <= len(prompts) <= 4
                    or any(not isinstance(p, str) for p in prompts)
                ):
                    raise ValueError("diagnostic prompts must be one to four strings")
                params = SamplingParams(
                    max_tokens=1, temperature=1, logprobs=-1, detokenize=False, seed=config.seed
                )
                result = llm.generate(prompts, params, use_tqdm=False)
                vocab = llm.llm_engine.model_config.get_vocab_size()
                vectors = []
                for output in result:
                    mapping = output.outputs[0].logprobs[0]
                    if set(mapping) != set(range(vocab)):
                        raise ValueError("diagnostic output omits model vocabulary coordinates")
                    vectors.append([float(mapping[i].logprob) for i in range(vocab)])
                connection.send(
                    [
                        "full_logits",
                        {
                            "logits": vectors,
                            "prompt_token_ids": [
                                list(output.prompt_token_ids) for output in result
                            ],
                            "vocab_size": vocab,
                        },
                    ],
                    time.monotonic() + 120,
                )
                continue
            if diagnostic_logits:
                raise ValueError("diagnostic worker cannot produce Sample logprobs")
            if (
                not isinstance(request, list)
                or len(request) != 3
                or request[0] != "generate"
                or type(request[2]) is not int
                or request[2] <= 0
            ):
                raise ValueError("invalid rollout request wire shape")
            _, prompts, group_size = request
            if not isinstance(prompts, list) or any(not isinstance(p, str) for p in prompts):
                raise ValueError("invalid rollout request wire shape")
            # Per-call seeds reset stochastic generators and are independent of request IDs.
            params = [
                SamplingParams(
                    n=group_size,
                    seed=config.seed + i,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_tokens=config.max_tokens,
                    logprobs=0,
                    detokenize=True,
                    skip_special_tokens=True,
                )
                for i, _ in enumerate(prompts)
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            outputs = [
                {
                    "outputs": [
                        {
                            "index": c.index,
                            "text": c.text,
                            "token_ids": list(c.token_ids),
                            "logprobs": [
                                {str(t): float(v.logprob) for t, v in m.items()} for m in c.logprobs
                            ],
                        }
                        for c in r.outputs
                    ]
                }
                for r in outputs
            ]
            connection.send(["outputs", outputs], time.monotonic() + 120)
    except BaseException:  # noqa: BLE001 - serialize child failure across the RPC boundary
        try:
            connection.send(["error", traceback.format_exc()], time.monotonic() + 1)
        except (OSError, TimeoutError, ValueError):
            pass  # A lost peer must never block local GPU cleanup.
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        finally:
            try:
                if cleanup is not None:
                    cleanup()
                    print("rollout distributed groups destroyed", flush=True)
            finally:
                try:
                    if ram_fd is not None:
                        from rl.learner.handoff_protocol import _close_identity

                        _close_identity(
                            ram_fd, (checkpoint.receipt.device, checkpoint.receipt.inode)
                        )
                finally:
                    connection.close()


class WorkerStartupError(RuntimeError):
    """Retain a partially started worker and distinguish confirmed cleanup."""

    def __init__(self, worker, cleanup_confirmed):
        super().__init__("vLLM worker startup failed")
        self.worker = worker
        self.cleanup_confirmed = cleanup_confirmed


class VLLMWorker:
    """Absolute-deadline RPC and transport-independent owned-process cleanup."""

    STARTUP_SECONDS = 300
    REQUEST_SECONDS = 120
    CLOSE_SEND_SECONDS = 0.1
    GRACE_SECONDS = 30
    TERM_SECONDS = 10
    KILL_SECONDS = 10

    def __init__(
        self,
        checkpoint,
        config,
        *,
        diagnostic_logits=False,
        ram_fd=None,
        runtime_root=None,
        _serve_target=None,
    ):
        deadline = time.monotonic() + self.STARTUP_SECONDS
        from rl.sync.ram_weights import RamPolicyCheckpoint

        ram = isinstance(checkpoint, RamPolicyCheckpoint)
        try:
            if ram:
                if ram_fd is None or runtime_root is None:
                    raise ValueError("RAM worker capability/runtime root required")
                checkpoint.receipt.validate(ram_fd)
            else:
                if ram_fd is not None or _serve_target is not None:
                    raise ValueError("RAM-only worker option")
                _validate_owned_bf16(checkpoint)
            check_deadline(deadline)
        except BaseException as validation_error:
            validation_error.cleanup_confirmed = True  # No worker or socket exists yet.
            raise
        context = mp.get_context("spawn")
        parent, child = socket.socketpair()
        self._connection = FramedSocket(parent)
        self._ram_capability = None
        arguments = (child, checkpoint, config, diagnostic_logits)
        if ram:
            from rl.rollout.ram_loader import SpawnFD

            self._ram_capability = SpawnFD(ram_fd, checkpoint.receipt)
            arguments += (self._ram_capability, runtime_root)
        self._process = context.Process(
            target=_serve if _serve_target is None else _serve_target, args=arguments
        )
        self._closed = False
        self._failed = False
        self._start_uncertain = ram
        try:
            self._process.start()
            self._start_uncertain = False
        except BaseException as startup_error:
            if ram:
                # start() may have forked before assigning _popen/pid. Neither
                # pid=None nor serialization failure proves no inherited holder.
                self._failed = True
                errors = [startup_error]
                for channel in (self._connection, child):
                    try:
                        channel.close()
                    except BaseException as error:  # noqa: BLE001 - retain ownership on interrupted cleanup
                        errors.append(error)
                raise WorkerStartupError(self, False) from BaseExceptionGroup(
                    "RAM spawn/cleanup uncertain", errors
                )
            self._connection.close()
            child.close()
            if self._process.pid is None:
                self._closed = True
                raise WorkerStartupError(self, True) from startup_error
            try:
                self.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve ownership on any failure
                raise WorkerStartupError(self, False) from BaseExceptionGroup(
                    "process start and cleanup failed", [startup_error, cleanup_error]
                )
            raise WorkerStartupError(self, True) from startup_error
        try:
            child.close()
            kind, loaded = self._receive(deadline=deadline)
            if kind != "ready" or loaded[0] != (checkpoint.ack() if ram else asdict(checkpoint)):
                raise RuntimeError("backend did not confirm loaded checkpoint")
            if ram:
                checkpoint.receipt.validate(ram_fd)
            self.cache_environment = loaded[1]
        except BaseException as startup_error:
            try:
                self.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve ownership on any failure
                raise WorkerStartupError(self, False) from BaseExceptionGroup(
                    "startup and cleanup failed", [startup_error, cleanup_error]
                )
            raise WorkerStartupError(self, True) from startup_error

    def full_logits(self, prompts):
        if self._closed or self._failed:
            raise RuntimeError("vLLM worker is closed or failed")
        deadline = time.monotonic() + self.REQUEST_SECONDS
        try:
            self._connection.send(["full_logits", [p.text for p in prompts]], deadline)
            kind, vectors = self._receive(deadline=deadline)
            if kind != "full_logits":
                raise RuntimeError("unexpected diagnostic response")
            self.last_logits_metadata = {
                key: vectors[key] for key in ("prompt_token_ids", "vocab_size")
            }
            return vectors["logits"]
        except BaseException:
            self._failed = True
            self.close()
            raise

    def _receive(self, timeout=120, *, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + timeout
        message = self._connection.receive(deadline)
        if not isinstance(message, list) or len(message) != 2:
            raise ValueError("invalid rollout response wire shape")
        kind, value = message
        if kind == "error":
            raise RuntimeError(value)
        return kind, value

    def generate(self, prompts, group_size):
        if self._closed or self._failed:
            raise RuntimeError("vLLM worker is closed or failed")
        deadline = time.monotonic() + self.REQUEST_SECONDS
        try:
            texts = []
            for prompt in prompts:
                check_deadline(deadline)
                texts.append(prompt.text)
            self._connection.send(["generate", texts, group_size], deadline)
            kind, result = self._receive(deadline=deadline)
            if kind != "outputs":
                raise RuntimeError("unexpected vLLM worker response")
            outputs = []
            for request in result:
                check_deadline(deadline)
                completions = []
                for completion in request["outputs"]:
                    check_deadline(deadline)
                    completions.append(
                        NS(
                            index=completion["index"],
                            text=completion["text"],
                            token_ids=completion["token_ids"],
                            logprobs=[
                                {int(t): NS(logprob=v) for t, v in mapping.items()}
                                for mapping in completion["logprobs"]
                            ],
                        )
                    )
                outputs.append(NS(outputs=completions))
            check_deadline(deadline)
            return outputs
        except BaseException:
            self._failed = True
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        if getattr(self, "_start_uncertain", False):
            raise RuntimeError("ambiguous spawn retains original Process and RAM capability")
        self._failed = True  # Retain owner handles and refuse work even if cleanup fails.
        try:
            if self._process.is_alive():
                try:
                    self._connection.send(["close"], time.monotonic() + self.CLOSE_SEND_SECONDS)
                except (OSError, TimeoutError, ValueError):
                    pass  # Graceful signalling is best-effort; cleanup never waits on transport.
        finally:
            try:
                self._connection.close()
            finally:
                self._join_owned()

    def _join_owned(self):
        self._process.join(timeout=self.GRACE_SECONDS)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=self.TERM_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=self.KILL_SECONDS)
        if self._process.is_alive():
            raise RuntimeError("owned vLLM worker did not exit; retain GPU guard")
        self._closed = True
        if self._process.exitcode != 0:
            raise RuntimeError(f"owned vLLM worker exitcode={self._process.exitcode}")
        print(f"rollout worker pid={self._process.pid} joined exitcode=0", flush=True)
