import math

import pytest

from scaling.analyze_nsys import ModelStructure, mfu


def test_mfu_known_values():
    # 6*1000 + 12*2*4*8*16 = 18288 FLOPs/token.
    assert mfu(100, ModelStructure(1000, 2, 4, 8), 16, 2, 10_000_000) == pytest.approx(0.09144)


def test_mfu_global_tokens_and_active_moe_parameters():
    shape = ModelStructure(1000, 2, 4, 8)
    assert mfu(100, shape, 16, 4, 10_000_000) == mfu(100, shape, 16, 2, 10_000_000) / 2
    assert mfu(0, shape, 16, 2, 10_000_000) == 0
    # Do not clamp values > 1; inconsistent measurement assumptions must remain visible.
    assert mfu(100000, shape, 16, 2, 10_000_000) > 1


@pytest.mark.parametrize(
    "args",
    [
        (100, 8_000_000_000, 16, 2, 1e7),
        (-1, ModelStructure(1000, 2, 4, 8), 16, 2, 1e7),
        (math.inf, ModelStructure(1000, 2, 4, 8), 16, 2, 1e7),
        (100, ModelStructure(1000, 2, 4, 8), 0, 2, 1e7),
        (100, ModelStructure(1000, 2, 4, 8), 16, True, 1e7),
        (100, ModelStructure(1000, 2, 4, 8), 16, 2, math.nan),
    ],
)
def test_mfu_rejects_ambiguous_or_invalid_input(args):
    with pytest.raises((ValueError, TypeError)):
        mfu(*args)


@pytest.mark.parametrize("args", [(0, 1, 1, 1), (10, -1, 1, 1), (10, 1, 1.5, 1)])
def test_structure_requires_positive_integers(args):
    with pytest.raises(ValueError):
        ModelStructure(*args)
