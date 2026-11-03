"""One immutable loaded policy and one explicitly owned, closeable rollout worker.

Create/use/close only inside scripts/with_gpu.sh for actual GPU execution.
Owned stores support exclusive close/load/rollback updates. Two 0.80 engines
are never staged simultaneously by this transaction. A version becomes usable
only after the worker successfully loads its corresponding checkpoint.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from rl.rollout.sampler import Prompt, Sample, SamplingConfig, samples_from_outputs


@dataclass(frozen=True)
class PolicyCheckpoint:
    model_path: str
    revision: str
    policy_version: int

    def __post_init__(self):
        if not isinstance(self.model_path, str) or not Path(self.model_path).is_absolute():
            raise ValueError("model_path must be an absolute local checkpoint path")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("revision must identify the loaded checkpoint")
        if type(self.policy_version) is not int or self.policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")


class RolloutEngine:
    def __init__(
        self,
        checkpoint: PolicyCheckpoint,
        config: SamplingConfig | None = None,
        *,
        _backend_factory=None,
        weight_store=None,
        _diagnostic_logits=False,
    ):
        self._lock = RLock()
        self._closed = True
        self._checkpoint = checkpoint
        self._config = config or SamplingConfig()
        if _backend_factory is None:
            from rl.rollout._worker import VLLMWorker

            _backend_factory = VLLMWorker
        self._weight_store = weight_store
        self._backend_factory = _backend_factory
        self._diagnostic_logits = _diagnostic_logits
        self._sync_failed = False
        self._pending_backend = None
        self._backend = None
        self._ownership_unknown = False
        if weight_store is not None:
            weight_store._reserve_engine(self, checkpoint)
        try:
            self._backend = self._start(checkpoint)
        except BaseException as startup_error:
            self._sync_failed = True
            confirmed = getattr(startup_error, "cleanup_confirmed", False)
            self._pending_backend = None if confirmed else getattr(startup_error, "worker", None)
            self._ownership_unknown = not confirmed and self._pending_backend is None
            # The original exception/traceback survives; caller can retry close on this owner.
            startup_error.recovery_engine = self
            if confirmed and weight_store is not None:
                weight_store._release_engine(self)
            raise
        self._closed = False

    def _start(self, checkpoint):
        if self._diagnostic_logits:
            return self._backend_factory(checkpoint, self._config, diagnostic_logits=True)
        return self._backend_factory(checkpoint, self._config)

    def load_weights(self, version: int):
        """Exclusive close/load/rollback. No active version changes before load ACK.

        Unconfirmed constructor/cleanup failure retains ownership and fails closed;
        it must never start a second worker for rollback in that case.
        """
        from rl.sync.version import verify_checkpoint

        with self._lock:
            if self._closed or self._sync_failed or self._weight_store is None:
                raise RuntimeError("atomic load requires a live owned-store engine")
            candidate = self._weight_store.checkpoint(version)
            if version == self.policy_version:
                return
            if version < self.policy_version:
                raise ValueError("active version must increase")
            previous = self._checkpoint
            try:
                self._backend.close()
            except BaseException:
                self._sync_failed = True
                raise
            self._closed = True
            self._backend = None
            new_backend = None
            try:
                new_backend = self._start(candidate)
                verify_checkpoint(candidate)
            except BaseException as load_error:
                cleanup_confirmed = getattr(load_error, "cleanup_confirmed", False)
                self._pending_backend = getattr(load_error, "worker", None)
                if new_backend is not None:
                    self._pending_backend = new_backend
                    try:
                        new_backend.close()
                        cleanup_confirmed = True
                    except BaseException as cleanup_error:  # noqa: BLE001 - preserve ownership on any failure
                        self._sync_failed = True
                        raise BaseExceptionGroup(
                            "candidate load and cleanup failed", [load_error, cleanup_error]
                        )
                if not cleanup_confirmed:
                    self._sync_failed = True
                    self._ownership_unknown = self._pending_backend is None
                    raise
                self._pending_backend = None
                try:
                    verify_checkpoint(previous)
                    self._backend = self._start(previous)
                    verify_checkpoint(previous)
                    self._closed = False
                except BaseException as rollback_error:  # noqa: BLE001 - fail closed on uncertain recovery
                    self._sync_failed = True
                    confirmed = getattr(rollback_error, "cleanup_confirmed", False)
                    self._pending_backend = (
                        None if confirmed else getattr(rollback_error, "worker", self._backend)
                    )
                    self._ownership_unknown = not confirmed and self._pending_backend is None
                    raise BaseExceptionGroup(
                        "candidate load and rollback failed", [load_error, rollback_error]
                    )
                raise
            self._backend = new_backend
            self._checkpoint = candidate
            self._weight_store._activate_engine(self, version)
            self._closed = False

    def full_logits(self, prompts):
        """Diagnostic last-position logits, separate from Sample raw logprobs."""
        with self._lock:
            if self._closed or self._sync_failed or not self._diagnostic_logits:
                raise RuntimeError("full logits require a live diagnostic engine")
            return self._backend.full_logits(prompts)

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def config(self):
        return self._config

    @property
    def policy_version(self):
        return self._checkpoint.policy_version

    def generate(self, prompts: list[Prompt], group_size: int, policy_version: int) -> list[Sample]:
        with self._lock:
            if self._closed or self._sync_failed:
                raise RuntimeError("rollout engine is closed or synchronization failed")
            if self._diagnostic_logits:
                raise RuntimeError("diagnostic logits cannot produce Sample.logprobs")
            if type(policy_version) is not int or policy_version != self.policy_version:
                raise ValueError(f"request does not match loaded policy {self.policy_version}")
            if type(group_size) is not int or group_size <= 0:
                raise ValueError("group_size must be a positive integer")
            if not isinstance(prompts, list) or any(not isinstance(p, Prompt) for p in prompts):
                raise ValueError("prompts must be a list of Prompt")
            if not prompts:
                return []
            try:
                outputs = self._backend.generate(prompts, group_size)
                return samples_from_outputs(prompts, outputs, group_size, self.policy_version)
            except BaseException:
                self.close()
                raise

    def close(self):
        with self._lock:
            if self._ownership_unknown:
                raise RuntimeError("worker ownership unresolved without a cleanup handle")
            if self._pending_backend is not None:
                self._pending_backend.close()
                self._pending_backend = None
            if not self._closed:
                # Only mark closed after confirmed worker shutdown. Failures stay visible.
                self._backend.close()
                self._closed = True
            if self._weight_store is not None:
                self._weight_store._release_engine(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
