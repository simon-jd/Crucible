"""Bounded vocabulary tiles; all normalization arithmetic is FP32.

Partial (max, shifted sum) pairs are reduced per row. Keeping max and sum separate
is essential when a large common offset would destroy precision in max+log(sum).
Backward recomputes these pairs and writes just the required dense logits gradient.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _partials(
    X,
    PM,
    PS,
    T: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    K: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk * K + tl.arange(0, K)
    x = tl.load(X + (row // T) * S0 + (row % T) * S1 + col * S2, col < V, other=-float("inf")).to(
        tl.float32
    )
    maximum = tl.max(x, 0)
    total = tl.sum(tl.exp(x - maximum), 0)
    # A completely masked (-inf) tile contributes zero if another tile is finite.
    total = tl.where(maximum == -float("inf"), 0.0, total)
    tl.store(PM + row * C + chunk, maximum)
    tl.store(PS + row * C + chunk, total)


@triton.jit
def _reduce(PM, PS, M, S, C: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, K)
    maximums = tl.load(PM + row * C + col, col < C, other=-float("inf"))
    sums = tl.load(PS + row * C + col, col < C, other=0.0)
    maximum = tl.max(maximums, 0)
    total = tl.sum(sums * tl.exp(maximums - maximum), 0)
    tl.store(M + row, maximum)
    tl.store(S + row, total)


@triton.jit
def _select(
    X,
    Y,
    M,
    S,
    O,
    T: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
):
    row = tl.program_id(0)
    b, t = row // T, row % T
    target = tl.load(Y + b * Y0 + t * Y1)
    value = tl.load(X + b * X0 + t * X1 + target * X2).to(tl.float32)
    maximum, total = tl.load(M + row), tl.load(S + row)
    tl.store(O + row, (value - maximum) - tl.log(total))


@triton.jit
def _gradient(
    X,
    Y,
    G,
    M,
    S,
    DX,
    T: tl.constexpr,
    V: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    K: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    b, t = row // T, row % T
    col = chunk * K + tl.arange(0, K)
    x = tl.load(X + b * X0 + t * X1 + col * X2, col < V, other=0.0).to(tl.float32)
    maximum, total = tl.load(M + row), tl.load(S + row)
    target = tl.load(Y + b * Y0 + t * Y1)
    grad = tl.load(G + b * G0 + t * G1)
    value = ((col == target).to(tl.float32) - tl.exp(x - maximum) / total) * grad
    tl.store(DX + row * V + col, value, col < V)


def _stats(logits):
    b, t, v = logits.shape
    rows, chunks = b * t, triton.cdiv(v, BLOCK)
    partial_max = torch.empty((rows, chunks), device=logits.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    maximum = torch.empty(rows, device=logits.device, dtype=torch.float32)
    total = torch.empty_like(maximum)
    _partials[(rows, chunks)](
        logits, partial_max, partial_sum, t, v, chunks, *logits.stride(), BLOCK
    )
    _reduce[(rows,)](
        partial_max, partial_sum, maximum, total, chunks, triton.next_power_of_2(chunks)
    )
    return maximum, total


def forward(logits, targets):
    maximum, total = _stats(logits)
    output = torch.empty(targets.shape, device=logits.device, dtype=torch.float32)
    _select[(targets.numel(),)](
        logits,
        targets,
        maximum,
        total,
        output,
        logits.shape[1],
        *logits.stride(),
        *targets.stride(),
    )
    return output


def backward(logits, targets, grad):
    maximum, total = _stats(logits)
    result = torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
    _gradient[(targets.numel(), triton.cdiv(logits.shape[-1], BLOCK))](
        logits,
        targets,
        grad,
        maximum,
        total,
        result,
        *logits.shape[1:],
        *logits.stride(),
        *targets.stride(),
        *grad.stride(),
        BLOCK,
    )
    return result
