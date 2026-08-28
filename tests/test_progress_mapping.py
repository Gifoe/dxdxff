from decimal import Decimal

import pytest

from reva_dlm.progress import (
    DEFAULT_CHECKPOINTS,
    build_progress_mapping,
    checkpoint_metadata,
    nearest_step_number,
)


def test_fixed_256_iteration_checkpoint_mapping():
    mapping = build_progress_mapping()
    assert [row["checkpoint"] for row in mapping] == list(DEFAULT_CHECKPOINTS)
    assert [row["step_number"] for row in mapping] == [64, 102, 128, 154, 192]
    assert [row["history_index"] for row in mapping] == [63, 101, 127, 153, 191]
    assert [row["normalized_progress_actual"] for row in mapping] == [
        0.25,
        102 / 256,
        0.5,
        154 / 256,
        0.75,
    ]


def test_nearest_half_up_is_not_bankers_rounding():
    # 0.25 * 10 == 2.5; ties go to 3, whereas Python round() would give 2.
    assert nearest_step_number(Decimal("0.25"), 10) == 3
    assert checkpoint_metadata("0.25", 10) == {
        "checkpoint": 0.25,
        "step_number": 3,
        "history_index": 2,
        "normalized_progress_actual": 0.3,
    }


def test_zero_maps_to_first_available_post_iteration_state():
    assert nearest_step_number(0, 256) == 1
    assert nearest_step_number(1, 256) == 256


@pytest.mark.parametrize("progress", [-0.01, 1.01, "nan", "inf"])
def test_invalid_progress_is_rejected(progress):
    with pytest.raises(ValueError):
        nearest_step_number(progress, 256)


@pytest.mark.parametrize("iterations", [0, -1])
def test_nonpositive_iteration_count_is_rejected(iterations):
    with pytest.raises(ValueError):
        nearest_step_number(0.5, iterations)
