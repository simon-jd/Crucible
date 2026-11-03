"""Within-group reward normalization; groups occupy the final dimension."""

import torch
from torch import Tensor


def group_advantages(rewards: Tensor, *, scale_by_std: bool = True) -> Tensor:
    """Population std (correction=0); singleton/constant groups give zero.

    The ablation scale_by_std=False only subtracts the group mean.
    """
    if rewards.ndim == 0 or rewards.shape[-1] == 0 or not rewards.is_floating_point():
        raise ValueError("rewards must be a nonempty floating tensor of groups")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if scale_by_std:
        centered = centered / (rewards.std(dim=-1, correction=0, keepdim=True) + 1e-6)
    return centered
