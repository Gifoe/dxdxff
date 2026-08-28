"""Deterministic mapping from normalized progress to saved iterations."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Final, Iterable


DEFAULT_CHECKPOINTS: Final[tuple[float, ...]] = (0.25, 0.40, 0.50, 0.60, 0.75)
PROGRESS_MAPPING_RULE: Final[str] = (
    "nearest-half-up over post-iteration states; step_number is 1-based and "
    "history_index is step_number - 1"
)


def _as_decimal(progress: int | float | str | Decimal) -> Decimal:
    try:
        value = progress if isinstance(progress, Decimal) else Decimal(str(progress))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid normalized progress: {progress!r}") from exc
    if not value.is_finite() or value < 0 or value > 1:
        raise ValueError(f"normalized progress must be in [0, 1], got {progress!r}")
    return value


def nearest_step_number(
    progress: int | float | str | Decimal,
    total_iterations: int,
) -> int:
    """Map normalized progress to a 1-based saved iteration.

    ``floor(progress * total_iterations + 0.5)`` implements nearest-half-up,
    avoiding Python's bankers' rounding.  Because the trajectory contains
    post-iteration states only, progress zero is clamped to the first saved
    state rather than inventing a history index ``-1``.
    """

    if isinstance(total_iterations, bool) or not isinstance(total_iterations, int):
        raise TypeError("total_iterations must be an integer")
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")

    normalized = _as_decimal(progress)
    rounded = int(
        (normalized * Decimal(total_iterations) + Decimal("0.5")).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )
    return min(total_iterations, max(1, rounded))


def checkpoint_metadata(
    checkpoint: int | float | str | Decimal,
    total_iterations: int = 256,
) -> dict[str, int | float]:
    """Return JSON-serializable metadata for one checkpoint."""

    normalized = _as_decimal(checkpoint)
    step_number = nearest_step_number(normalized, total_iterations)
    return {
        "checkpoint": float(normalized),
        "step_number": step_number,
        "history_index": step_number - 1,
        "normalized_progress_actual": step_number / total_iterations,
    }


def build_progress_mapping(
    checkpoints: Iterable[int | float | str | Decimal] = DEFAULT_CHECKPOINTS,
    total_iterations: int = 256,
) -> list[dict[str, int | float]]:
    """Build ordered checkpoint metadata using the fixed mapping rule."""

    return [checkpoint_metadata(value, total_iterations) for value in checkpoints]


# Compatibility names for downstream scripts.
progress_to_step = nearest_step_number
map_checkpoints = build_progress_mapping


__all__ = [
    "DEFAULT_CHECKPOINTS",
    "PROGRESS_MAPPING_RULE",
    "build_progress_mapping",
    "checkpoint_metadata",
    "map_checkpoints",
    "nearest_step_number",
    "progress_to_step",
]
