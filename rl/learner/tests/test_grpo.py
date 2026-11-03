"""Torch CPU checks, skipped by the dependency-light default suite."""

import pytest

torch = pytest.importorskip("torch")
from rl.learner.advantages import group_advantages
from rl.learner.grpo import grpo_loss
from rl.learner.staleness import filter_stale
from rl.rollout.sampler import Sample


def test_group_advantages_zero_mean():
    rewards = torch.tensor([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    result = group_advantages(rewards)
    expected = (rewards - rewards.mean(-1, keepdim=True)) / (
        rewards.std(-1, correction=0, keepdim=True) + 1e-6
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.mean(-1), torch.zeros(2), atol=1e-6, rtol=0)
    assert group_advantages(torch.tensor([5.0])).item() == 0
    torch.testing.assert_close(
        group_advantages(rewards, scale_by_std=False), rewards - rewards.mean(-1, keepdim=True)
    )


def test_grpo_loss_matches_reference():
    new = torch.tensor([[-1.0, -0.4, -2.0], [-2.0, -1.0, -0.8]], requires_grad=True)
    old = torch.tensor([[-1.2, -0.8, -3.0], [-1.5, -1.1, -0.7]])
    ref = new.detach() - 0.15
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    ratio = (new - old).exp()
    surrogate = torch.minimum(ratio * adv[:, None], ratio.clamp(0.8, 1.2) * adv[:, None])
    delta = ref - new
    expected = ((-surrogate + 0.001 * (delta.exp() - delta - 1)) * mask).sum() / mask.sum()
    actual = grpo_loss(new, old, ref, adv, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    (a_grad,) = torch.autograd.grad(actual, new, retain_graph=True)
    (e_grad,) = torch.autograd.grad(expected, new)
    torch.testing.assert_close(a_grad, e_grad, atol=1e-6, rtol=0)
    assert a_grad[~mask].eq(0).all()


def test_clip_bounds_ratio():
    new = torch.log(torch.tensor([[2.0], [0.5]], requires_grad=True))
    result = grpo_loss(
        new,
        torch.zeros_like(new),
        new.detach(),
        torch.tensor([1.0, -1.0]),
        torch.ones_like(new, dtype=torch.bool),
        kl_coef=0,
    )
    torch.testing.assert_close(result, torch.tensor(-0.2), atol=1e-6, rtol=0)
    (grad,) = torch.autograd.grad(result, new)
    assert grad.eq(0).all()


def test_mask_ignores_nonfinite_padding_and_rejects_empty():
    new = torch.tensor([[-1.0, float("nan")]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    result = grpo_loss(new, new.detach(), new.detach(), torch.tensor([1.0]), mask)
    result.backward()
    assert result.item() == -1
    assert new.grad.tolist() == [[-1.0, 0.0]]
    with pytest.raises(ValueError, match="token"):
        grpo_loss(new, new.detach(), new.detach(), torch.tensor([1.0]), ~torch.ones_like(mask))


def test_filter_stale_drops_old_versions():
    samples = [Sample("p", "", (1,), (-1.0,), version) for version in [0, 2, 3]]
    assert filter_stale(samples, 3, 1) == (samples[1:], 1)
    assert filter_stale(samples, 3, 0) == ([samples[-1]], 2)
    with pytest.raises(ValueError, match="future"):
        filter_stale(samples, 2, 1)
    with pytest.raises(ValueError):
        filter_stale(samples, 3, -1)


@pytest.mark.parametrize("coefficient", [0, 0.001, 0.04])
def test_kl_ablation_coefficients_and_fixed_targets(coefficient):
    new = torch.tensor([[-2.0]], requires_grad=True)
    old = torch.tensor([[-2.0]], requires_grad=True)
    ref = torch.tensor([[-1.0]], requires_grad=True)
    adv = torch.tensor([0.0], requires_grad=True)
    result = grpo_loss(new, old, ref, adv, torch.tensor([[True]]), kl_coef=coefficient)
    torch.testing.assert_close(result, coefficient * (torch.tensor(1.0).exp() - 2))
    result.backward()
    assert old.grad is None and ref.grad is None and adv.grad is None


def test_disabled_kl_does_not_overflow_unused_exponential():
    new = torch.tensor([[-1000.0]], requires_grad=True)
    result = grpo_loss(
        new,
        new.detach(),
        torch.tensor([[-1.0]]),
        torch.tensor([1.0]),
        torch.tensor([[True]]),
        kl_coef=0,
    )
    result.backward()
    assert result.item() == -1
    assert new.grad.item() == -1
