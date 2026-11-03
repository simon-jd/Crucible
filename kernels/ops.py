"""Chunked token log-probabilities, with an opaque first-order CUDA backward.

[B,T,V] FP32/BF16 on CPU/CUDA; CPU also supports FP64 for mathematical checks.
Outputs accumulate in FP32 (FP64 for FP64 input); gradients have input dtype.
Targets are int64, [B,T], on the same device. Arbitrary strided inputs are accepted.
B/T may be zero; V must be positive. Invalid targets raise before kernel dispatch
(including a CUDA host synchronization); they never trigger a device assertion.
Nonfinite logits follow floating-point propagation, as in log_softmax.

CUDA scratch is O(B*T*ceil(V/1024)); backward recomputes shifted normalization
statistics from saved logits, never from target_logit - logprob. Only first-order
AD is supported. Call runtime.configure_runtime before importing Torch in runners.
"""

import torch

CHUNK = 1024


def _metadata(logits, targets):
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("expected logits shape [B,T,V] and targets shape [B,T]")
    if logits.shape[-1] <= 0:
        raise ValueError("vocab must be positive")
    if logits.device != targets.device:
        raise ValueError("logits and targets must have the same device")
    if logits.device.type not in ("cpu", "cuda", "meta"):
        raise ValueError("only CPU and CUDA devices are supported")
    allowed = (torch.float32, torch.bfloat16)
    if logits.device.type in ("cpu", "meta"):
        allowed += (torch.float64,)
    if logits.dtype not in allowed or targets.dtype != torch.int64:
        raise TypeError("unsupported logits or targets dtype")


def _dtype(logits):
    return torch.float64 if logits.dtype == torch.float64 else torch.float32


def _cpu_stats(logits):
    # Two bounded passes keep the maximum separate from log(sum(exp(x-max))).
    maximum = logits.amax(-1).to(_dtype(logits))
    total = torch.zeros_like(maximum)
    for start in range(0, logits.shape[-1], CHUNK):
        part = logits[..., start : start + CHUNK].to(_dtype(logits))
        total.add_((part - maximum[..., None]).exp().sum(-1))
    return maximum, total


@torch.library.custom_op("crucible::token_logprobs", mutates_args=())
def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _metadata(logits, targets)
    if targets.numel() and bool(((targets < 0) | (targets >= logits.shape[-1])).any()):
        raise ValueError("target index outside vocabulary")
    if not targets.numel():
        return torch.empty(targets.shape, device=logits.device, dtype=_dtype(logits))
    if logits.is_cuda:
        from kernels.logprob_triton import forward

        return forward(logits, targets)
    maximum, total = _cpu_stats(logits)
    selected = logits.gather(-1, targets[..., None]).squeeze(-1).to(_dtype(logits))
    return (selected - maximum) - total.log()


@token_logprobs.register_fake
def _fake(logits, targets):
    _metadata(logits, targets)
    return torch.empty(targets.shape, device=logits.device, dtype=_dtype(logits))


@torch.library.custom_op("crucible::_token_logprobs_backward", mutates_args=())
def _backward(logits: torch.Tensor, targets: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if not targets.numel():
        return torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
    if logits.is_cuda:
        from kernels.logprob_triton import backward

        return backward(logits, targets, grad)
    maximum, total = _cpu_stats(logits)
    result = torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
    for start in range(0, logits.shape[-1], CHUNK):
        stop = min(start + CHUNK, logits.shape[-1])
        part = logits[..., start:stop].to(_dtype(logits))
        probabilities = (part - maximum[..., None]).exp() / total[..., None]
        indices = torch.arange(start, stop, device=logits.device)
        selected = (targets[..., None] == indices).to(_dtype(logits))
        result[..., start:stop] = (selected - probabilities) * grad[..., None]
    return result


@_backward.register_fake
def _backward_fake(logits, targets, grad):
    return torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _autograd_backward(ctx, grad):
    logits, targets = ctx.saved_tensors
    return _backward(logits, targets, grad), None


token_logprobs.register_autograd(_autograd_backward, setup_context=_setup_context)
