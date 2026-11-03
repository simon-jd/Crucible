"""CPU math/registration and opt-in actual CUDA kernel acceptance."""

import os

import pytest

torch = pytest.importorskip("torch")
from kernels.ops import token_logprobs

DEVICES = ["cpu"] + (["cuda"] if os.environ.get("CRUCIBLE_LOGPROB_GPU") == "1" else [])


def reference(logits, targets):
    value = logits if logits.dtype == torch.float64 else logits.float()
    return value.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("vocab", [1, 7, 2051])
def test_matches_reference_fp32(device, vocab):
    torch.manual_seed(10)
    x = torch.randn(2, 3, vocab, device=device, requires_grad=True)
    y = torch.randint(vocab, (2, 3), device=device)
    actual = token_logprobs(x, y)
    expected = reference(x, y)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    weight = torch.randn_like(actual)
    grad = torch.autograd.grad(actual, x, weight)[0]
    ref_grad = torch.autograd.grad(expected, x, weight)[0]
    torch.testing.assert_close(grad, ref_grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_matches_reference_bf16(device):
    torch.manual_seed(11)
    x = torch.randn(2, 4, 4099, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = torch.randint(4099, (2, 4), device=device)
    actual, expected = token_logprobs(x, y), reference(x, y)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    weight = torch.randn_like(actual)
    grad = torch.autograd.grad(actual, x, weight)[0]
    ref_grad = torch.autograd.grad(expected, x, weight)[0]
    assert grad.dtype == torch.bfloat16
    torch.testing.assert_close(grad, ref_grad, atol=1e-5, rtol=0.01)


def test_gradcheck_small():
    torch.manual_seed(12)
    x = torch.randn(2, 2, 5, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0, 4], [2, 1]])
    assert torch.autograd.gradcheck(token_logprobs, (x, y))


@pytest.mark.parametrize("device", DEVICES)
def test_torch_compile_fullgraph(device):
    torch.manual_seed(13)
    x = torch.randn(2, 3, 19, device=device, requires_grad=True)
    y = torch.randint(19, (2, 3), device=device)
    compiled = torch.compile(token_logprobs, fullgraph=True, backend="inductor")
    eager = token_logprobs(x, y)
    result = compiled(x, y)
    torch.testing.assert_close(result, eager)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), x)[0], torch.autograd.grad(eager.sum(), x)[0]
    )


@pytest.mark.parametrize("device", DEVICES)
def test_registration_opcheck(device):
    x = torch.randn(2, 3, 13, device=device, requires_grad=True)
    y = torch.randint(13, (2, 3), device=device)
    result = torch.library.opcheck(torch.ops.crucible.token_logprobs.default, (x, y))
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["offset", "extreme", "strided", "expanded"])
def test_stable_boundaries_and_layout_backward(device, case):
    torch.manual_seed(14)
    x = torch.randn(2, 3, 2051, device=device)
    y = torch.randint(2051, (3, 2), device=device).T
    if case == "offset":
        x += 1e8
    elif case == "extreme":
        x *= 1e20
    elif case == "strided":
        x = torch.randn(3, 2, 4102, device=device).transpose(0, 1)[..., ::2]
    else:
        x = x[:1, :1].expand(2, 3, -1)
    x.requires_grad_()
    actual, expected = token_logprobs(x, y), reference(x, y)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-6)
    weight = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, weight)[0],
        torch.autograd.grad(expected, x, weight)[0],
        atol=2e-6,
        rtol=2e-5,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_targets_do_not_poison_context(device):
    x = torch.randn(1, 2, 5, device=device)
    for bad in (-1, 5):
        with pytest.raises(ValueError, match="target"):
            token_logprobs(x, torch.tensor([[0, bad]], device=device))
    torch.testing.assert_close(
        token_logprobs(x, torch.zeros(1, 2, device=device, dtype=torch.long)),
        reference(x, torch.zeros(1, 2, device=device, dtype=torch.long)),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_metadata_and_empty(device):
    with pytest.raises(ValueError, match="shape"):
        token_logprobs(
            torch.randn(2, 3, device=device), torch.zeros(2, dtype=torch.long, device=device)
        )
    with pytest.raises(ValueError, match="vocab"):
        token_logprobs(
            torch.empty(1, 2, 0, device=device), torch.zeros(1, 2, dtype=torch.long, device=device)
        )
    with pytest.raises(TypeError, match="dtype"):
        token_logprobs(torch.randn(1, 2, 3, device=device), torch.zeros(1, 2, device=device))
    x = torch.empty(0, 2, 3, device=device, requires_grad=True)
    result = token_logprobs(x, torch.empty(0, 2, dtype=torch.long, device=device))
    assert result.shape == (0, 2)
    result.sum().backward()
    assert x.grad.shape == x.shape


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("CRUCIBLE_LOGPROB_GPU") != "1", reason="fixed GPU guard required"
)
def test_peak_memory_lower_than_reference():
    torch.manual_seed(15)
    x = torch.randn(2, 32, 152000, device="cuda", requires_grad=True)
    y = torch.randint(152000, (2, 32), device="cuda")

    # Both paths include forward + backward, their required dense logits gradient.
    def peak(fn):
        x.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        fn(x, y).sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - start

    peak(token_logprobs)
    peak(reference)
    candidate, baseline = peak(token_logprobs), peak(reference)
    print(
        {
            "vocab": 152000,
            "candidate_peak_increment_bytes": candidate,
            "reference_peak_increment_bytes": baseline,
        },
        flush=True,
    )
    assert candidate < baseline


def test_meta_and_rejected_dtypes():
    result = torch.ops.crucible.token_logprobs(
        torch.empty(2, 3, 13, device="meta", dtype=torch.bfloat16),
        torch.empty(2, 3, device="meta", dtype=torch.long),
    )
    assert result.device.type == "meta" and result.dtype == torch.float32
    assert result.shape == (2, 3)
    with pytest.raises(TypeError, match="dtype"):
        token_logprobs(
            torch.ones(1, 2, 3, dtype=torch.float16), torch.zeros(1, 2, dtype=torch.long)
        )
    with pytest.raises(ValueError, match="shape"):
        token_logprobs(torch.ones(1, 2, 3), torch.zeros(2, 1, dtype=torch.long))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [-float("inf"), float("inf"), float("nan")])
def test_nonfinite_propagation(device, value):
    # A complete -inf vocabulary tile must not poison another finite tile.
    x = torch.zeros(1, 2, 2051, device=device)
    x[..., :1024] = value
    y = torch.tensor([[0, 2050]], device=device)
    torch.testing.assert_close(token_logprobs(x, y), reference(x, y), equal_nan=True)


def test_learner_uses_op_for_policy_and_reference(monkeypatch):
    import copy

    from rl.learner import train
    from rl.learner.tests.test_train import TinyLM, group, trainer

    torch.manual_seed(16)
    model = TinyLM()
    candidate = trainer(copy.deepcopy(model), copy.deepcopy(model), 2)
    baseline = trainer(copy.deepcopy(model), copy.deepcopy(model), 2)
    calls = []

    def counted(logits, targets):
        calls.append(torch.is_grad_enabled())
        return token_logprobs(logits, targets)

    monkeypatch.setattr(train, "token_logprobs", counted)
    actual = candidate.step([group()])
    assert calls == [False, True]
    monkeypatch.setattr(train, "_score", train._score_reference)
    expected = baseline.step([group()])
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    for key, value in candidate.policy.state_dict().items():
        torch.testing.assert_close(value, baseline.policy.state_dict()[key], atol=1e-7, rtol=1e-6)
