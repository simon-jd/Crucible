"""Clipped GRPO surrogate with the k3 reference KL estimator."""

import math

import torch
from torch import Tensor


def grpo_loss(
    logp_new: Tensor,
    logp_old: Tensor,
    logp_ref: Tensor,
    adv: Tensor,
    mask: Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.001,
) -> Tensor:
    """Average over valid generated tokens, with sequence or token advantages.

    Old/reference probabilities and rewards are fixed targets. Padding is removed
    before exponentiation, so even NaN padding cannot affect loss or gradients.
    """
    if logp_new.ndim != 2 or any(t.shape != logp_new.shape for t in (logp_old, logp_ref, mask)):
        raise ValueError("logps and mask must have the same [samples, tokens] shape")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("mask must select at least one valid token")
    if not math.isfinite(clip_eps) or not 0 <= clip_eps < 1:
        raise ValueError("clip_eps must be in [0, 1)")
    if not math.isfinite(kl_coef) or kl_coef < 0:
        raise ValueError("kl_coef must be nonnegative and finite")
    if adv.shape == logp_new.shape[:1]:
        adv = adv[:, None].expand_as(logp_new)
    if adv.shape != logp_new.shape:
        raise ValueError("advantages must have sample or token shape")
    new = logp_new[mask].float()
    old = logp_old[mask].detach().float()
    ref = logp_ref[mask].detach().float()
    advantage = adv[mask].detach().float()
    if any(not torch.isfinite(t).all() for t in (new, old, ref, advantage)):
        raise ValueError("valid logps and advantages must be finite")
    ratio = (new - old).exp()
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
    )
    token_loss = -surrogate
    if kl_coef:
        delta = ref - new
        token_loss = token_loss + kl_coef * (delta.exp() - delta - 1)
    loss = token_loss.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite GRPO loss")
    return loss
