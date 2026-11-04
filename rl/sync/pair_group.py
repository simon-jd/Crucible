"""A two-rank learner <-> rollout process group beside each process's own default group.

The FSDP2 learner needs its own world-1 NCCL default group and in-process vLLM
creates another world-1 default group, so the weight-broadcast group cannot be
the default group of either process. `init_pair_group` builds a separate
ProcessGroup on its own TCPStore (rank 0 hosts it) and registers the rank
mapping {0: 0, 1: 1}, so `dist.broadcast(..., src=0, group=pair)` and
`WeightSync(store, group=pair, source=0)` address pair ranks directly.

This uses Torch 2.8's private `_new_process_group_helper` the way RLHF frameworks
do; it is pinned to the validated Torch 2.8.0 runtime. The default group of the
calling process must not be bound to a device (no `device_id`), otherwise Torch
would try to split the new communicator from it. Destroy with `destroy_pair_group`.
The returned TCPStore is also the control channel between the two roles.
"""

from __future__ import annotations

from datetime import timedelta


def init_pair_group(backend, host, port, rank, *, timeout=300, name="crucible-pair"):
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    if backend not in ("gloo", "nccl"):
        raise ValueError("pair group supports gloo (CPU) or nccl (CUDA)")
    if rank not in (0, 1):
        raise ValueError("pair rank must be 0 (learner/source) or 1 (rollout)")
    if not dist.is_initialized():
        raise RuntimeError("initialize this process's own default group first")
    if c10d._get_default_group().bound_device_id is not None:
        raise RuntimeError("default group is device-bound; Torch would split the pair from it")
    wait = timedelta(seconds=timeout)
    store = dist.TCPStore(host, port, 2, is_master=rank == 0, timeout=wait)
    group, _ = c10d._new_process_group_helper(
        2, rank, [], backend, dist.PrefixStore(name + "/", store), group_name=name, timeout=wait
    )
    c10d._world.pg_group_ranks[group] = {0: 0, 1: 1}
    return group, store


def pair_rank(group):
    """This process's rank in the pair (or any registered group), not its default rank."""
    import torch.distributed as dist

    return dist.get_global_rank(group, group.rank())


def destroy_pair_group(group):
    import torch.distributed as dist

    dist.destroy_process_group(group)
