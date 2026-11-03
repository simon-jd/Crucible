"""Offline by default: GPU and live/cloud tests run only when explicitly enabled."""

import os

import pytest

OPT_IN = {
    "gpu": ("CRUCIBLE_GPU_TESTS", "GPU test; set CRUCIBLE_GPU_TESTS=1 and run via scripts/with_gpu.sh"),
    "integration": ("CRUCIBLE_LIVE_TESTS", "live/cloud integration test; set CRUCIBLE_LIVE_TESTS=1"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, (variable, reason) in OPT_IN.items():
            # get_closest_marker, not item.keywords: keywords also hold parametrize IDs.
            if item.get_closest_marker(marker) and os.environ.get(variable) != "1":
                item.add_marker(pytest.mark.skip(reason=reason))
