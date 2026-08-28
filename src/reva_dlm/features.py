"""Leakage-safe, inexpensive features for denoising trajectories.

The public entry point in this module deliberately accepts raw histories rather
than a fully reconstructed trajectory object.  This makes its causality
contract easy to test: every statistic is computed from an inclusive prefix
ending at ``checkpoint_index``.  No ground-truth answer, correctness flag, or
final-state input is accepted by the API.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


__all__ = ["extract_causal_features"]


def _causal_prefix(history: Any, checkpoint_index: int) -> list[Any]:
    """Materialize at most ``history[0:checkpoint_index + 1]``.

    In particular, this helper never asks for the length of ``history``.  That
    matters for custom/lazy trajectory containers and prevents the full future
    horizon from becoming an accidental feature.
    """

    if history is None:
        return []
    if isinstance(history, (str, bytes)):
        return [history]

    stop = checkpoint_index + 1
    try:
        prefix = history[:stop]
    except (TypeError, AttributeError):
        # Support iterators without consuming elements beyond the checkpoint.
        prefix = []
        for index, value in enumerate(history):
            if index >= stop:
                break
            prefix.append(value)
        return prefix

    if hasattr(prefix, "detach"):
        prefix = prefix.detach()
    if hasattr(prefix, "cpu"):
        prefix = prefix.cpu()
    if hasattr(prefix, "tolist"):
        prefix = prefix.tolist()

    # A scalar array can occur in small synthetic tests.
    if not isinstance(prefix, Iterable) or isinstance(prefix, (str, bytes)):
        return [prefix]
    return list(prefix)


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def _flat_values(value: Any) -> list[Any]:
    """Convert a token or mask state to a deterministic flat Python list."""

    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except (TypeError, RuntimeError):
            pass

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [_python_scalar(item) for item in value]
        return [_python_scalar(value)]

    if array.ndim == 0:
        return [_python_scalar(array.item())]
    return [_python_scalar(item) for item in array.reshape(-1).tolist()]


def _clean_text(value: Any) -> str:
    """Canonicalize text without importing label- or answer-correctness logic."""

    if value is None:
        return ""
    value = _python_scalar(value)
    if isinstance(value, Real):
        try:
            if not np.isfinite(float(value)):
                return ""
        except (TypeError, ValueError):
            pass
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def _levenshtein_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Return Levenshtein distance using O(min(len(left), len(right))) space."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, start=1):
        current = [row]
        for column, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _normalized_edit_distance(left: Sequence[Any], right: Sequence[Any]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 0.0
    return float(_levenshtein_distance(left, right) / denominator)


def _committed_ratio(mask_state: Any, token_state: Any) -> float:
    """Interpret a Boolean mask, 0/1 mask, ratio, count, or index collection."""

    values = _flat_values(mask_state)
    token_count = len(_flat_values(token_state))
    if not values:
        return 0.0

    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        numeric = np.asarray([bool(value) for value in values], dtype=float)

    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return 0.0

    # Standard Boolean / 0-1 mask.  Equal length is important because an index
    # collection such as [0, 1] must not be mistaken for a two-token mask when
    # the candidate contains hundreds of tokens.
    is_binary = bool(np.all(np.isin(numeric, (0.0, 1.0))))
    if is_binary and (token_count == 0 or numeric.size == token_count):
        return float(np.clip(numeric.mean(), 0.0, 1.0))

    if numeric.size == 1:
        scalar = float(numeric[0])
        if 0.0 <= scalar <= 1.0:
            return scalar
        if token_count > 0:
            return float(np.clip(scalar / token_count, 0.0, 1.0))
        return 0.0

    # Some reconstruction pipelines store committed token indices rather than
    # a dense mask.  Count only valid positions when a candidate length exists.
    if token_count > 0:
        rounded = np.rint(numeric).astype(np.int64)
        valid = rounded[(rounded >= 0) & (rounded < token_count)]
        return float(np.clip(np.unique(valid).size / token_count, 0.0, 1.0))

    # Last-resort interpretation for a fractional soft mask.
    if np.all((numeric >= 0.0) & (numeric <= 1.0)):
        return float(np.clip(numeric.mean(), 0.0, 1.0))
    return 0.0


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    y = np.asarray(values, dtype=float)
    x = np.arange(y.size, dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centered_x, y - y.mean()) / denominator)


def _past_oscillations(answers: Sequence[str]) -> int:
    """Count returns to a previously left, non-empty parsed answer."""

    seen: set[str] = set()
    previous = ""
    count = 0
    for answer in answers:
        if answer and answer != previous and answer in seen:
            count += 1
        if answer:
            seen.add(answer)
        previous = answer
    return count


def extract_causal_features(
    candidate_token_history: Any,
    committed_mask_history: Any,
    parsed_answer_history: Any,
    text_history: Any,
    checkpoint_index: int,
    window: int = 16,
    *,
    total_steps: int | None = None,
) -> dict[str, float]:
    """Extract numeric G1 features from the inclusive trajectory prefix.

    Parameters
    ----------
    candidate_token_history:
        Candidate token state at each decoding checkpoint.  NumPy arrays,
        PyTorch tensors, nested lists, and other sliceable sequences are
        accepted.
    committed_mask_history:
        Per-checkpoint committed-token state.  Dense Boolean/0-1 masks are the
        preferred representation; scalar ratios/counts and collections of
        committed indices are also handled.
    parsed_answer_history:
        Canonically parsed answer at each checkpoint.  Missing answers should
        be represented as ``None`` or an empty string.
    text_history:
        Decoded text at each checkpoint.
    checkpoint_index:
        Zero-based current checkpoint.  ``progress`` is computed as
        ``(checkpoint_index + 1) / total_steps``.
    window:
        Number of *past transitions* included in recent statistics.
    total_steps:
        Immutable planned decoding horizon used for normalized progress.  Pass
        the decoding configuration value (256 for the frozen Mini G0/G1
        configuration).  If omitted, the candidate-history length is used for
        compatibility with callers that pass an already-truncated prefix.

    Returns
    -------
    dict[str, float]
        Finite numeric features computed solely from indices
        ``0..checkpoint_index``.  The dictionary intentionally contains no
        correctness, ground-truth, final-answer, label, or future statistic.

    Notes
    -----
    With explicit ``total_steps``, the caller may mutate, replace, append, or
    remove any state after ``checkpoint_index`` without changing the result.
    Omitting it is safe only when the supplied candidate history is itself the
    frozen prefix; full-trajectory callers should always pass the planned
    horizon so observed suffix length cannot influence progress.
    """

    if not isinstance(checkpoint_index, Integral):
        raise TypeError("checkpoint_index must be an integer")
    checkpoint_index = int(checkpoint_index)
    if checkpoint_index < 0:
        raise ValueError("checkpoint_index must be non-negative")
    if not isinstance(window, Integral):
        raise TypeError("window must be an integer")
    window = int(window)
    if window < 1:
        raise ValueError("window must be at least 1")

    if total_steps is None:
        try:
            total_steps = len(candidate_token_history)
        except (TypeError, AttributeError):
            # Iterators have no length.  The causal prefix materialized below
            # supplies the compatible fallback without consuming the suffix.
            total_steps = None
    elif not isinstance(total_steps, Integral):
        raise TypeError("total_steps must be an integer or None")

    # This is the only history access in the public routine.  All downstream
    # computations receive already-truncated local lists.
    token_prefix = _causal_prefix(candidate_token_history, checkpoint_index)
    commit_prefix = _causal_prefix(committed_mask_history, checkpoint_index)
    answer_prefix_raw = _causal_prefix(parsed_answer_history, checkpoint_index)
    text_prefix_raw = _causal_prefix(text_history, checkpoint_index)

    if total_steps is None:
        total_steps = len(token_prefix)
    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if checkpoint_index >= total_steps:
        raise ValueError("checkpoint_index must be smaller than total_steps")

    token_states = [_flat_values(state) for state in token_prefix]
    answers = [_clean_text(answer) for answer in answer_prefix_raw]
    texts = [_clean_text(text) for text in text_prefix_raw]

    current_tokens = token_states[-1] if token_states else []
    current_answer = answers[-1] if answers else ""
    current_text = texts[-1] if texts else ""

    # Calculate committed ratios without indexing any state beyond the shorter
    # causal prefix.  A missing mask state is represented as an unknown/zero
    # committed ratio rather than borrowing a later state.
    committed_ratios: list[float] = []
    prefix_steps = max(len(token_states), len(commit_prefix))
    for index in range(prefix_steps):
        mask_state = commit_prefix[index] if index < len(commit_prefix) else None
        token_state = token_states[index] if index < len(token_states) else []
        committed_ratios.append(_committed_ratio(mask_state, token_state))
    current_committed_ratio = committed_ratios[-1] if committed_ratios else 0.0

    answer_transition_start = max(1, len(answers) - window)
    recent_answer_distances = [
        _normalized_edit_distance(list(answers[index - 1]), list(answers[index]))
        for index in range(answer_transition_start, len(answers))
    ]
    recent_answer_changes = [distance > 0.0 for distance in recent_answer_distances]
    recent_answer_stabilities = [
        1.0 - distance for distance in recent_answer_distances
    ]

    if current_answer:
        persistence = 0
        for answer in reversed(answers):
            if answer != current_answer:
                break
            persistence += 1
        first_seen_index = answers.index(current_answer)
    else:
        persistence = 0
        first_seen_index = max(len(answers) - 1, 0)

    token_transition_start = max(1, len(token_states) - window)
    recent_token_change_rates = [
        _normalized_edit_distance(token_states[index - 1], token_states[index])
        for index in range(token_transition_start, len(token_states))
    ]
    current_token_change_rate = (
        recent_token_change_rates[-1] if recent_token_change_rates else 0.0
    )
    recent_token_stabilities = [1.0 - rate for rate in recent_token_change_rates]

    commit_transition_start = max(1, len(committed_ratios) - window)
    recent_commit_deltas = [
        committed_ratios[index] - committed_ratios[index - 1]
        for index in range(commit_transition_start, len(committed_ratios))
    ]

    answer_prefix_length = max(len(answers), 1)
    current_index_denominator = max(len(answers) - 1, 1)
    answer_stability = (
        1.0 - recent_answer_distances[-1]
        if current_answer and recent_answer_distances
        else float(bool(current_answer) and len(answers) == 1)
    )

    features = {
        "progress": float((checkpoint_index + 1) / total_steps),
        "prefix_length": float(checkpoint_index + 1),
        "current_masked_ratio": float(1.0 - current_committed_ratio),
        "current_committed_ratio": float(current_committed_ratio),
        "current_candidate_token_length": float(len(current_tokens)),
        "current_text_length": float(len(current_text)),
        "current_text_token_length": float(len(current_text.split())),
        "current_answer_present": float(bool(current_answer)),
        "current_answer_length": float(len(current_answer)),
        "current_answer_token_length": float(len(current_answer.split())),
        "current_answer_persistence": float(persistence),
        "current_answer_persistence_normalized": float(
            persistence / answer_prefix_length
        ),
        "current_answer_first_seen_time": float(first_seen_index),
        "current_answer_first_seen_normalized": float(
            first_seen_index / current_index_denominator
        ),
        "recent_answer_change_count": float(sum(recent_answer_changes)),
        "recent_answer_change_rate": float(
            np.mean(recent_answer_changes) if recent_answer_changes else 0.0
        ),
        "recent_answer_edit_distance": float(
            np.mean(recent_answer_distances) if recent_answer_distances else 0.0
        ),
        "current_answer_stability": float(answer_stability),
        "recent_answer_stability": float(
            np.mean(recent_answer_stabilities)
            if recent_answer_stabilities
            else 0.0
        ),
        "recent_answer_stability_slope": float(
            _linear_slope(recent_answer_stabilities)
        ),
        "current_token_change_rate": float(current_token_change_rate),
        "recent_token_change_rate": float(
            np.mean(recent_token_change_rates) if recent_token_change_rates else 0.0
        ),
        "current_token_stability": float(
            1.0 - current_token_change_rate if recent_token_change_rates else 0.0
        ),
        "recent_stability": float(
            np.mean(recent_token_stabilities) if recent_token_stabilities else 0.0
        ),
        "recent_stability_slope": float(_linear_slope(recent_token_stabilities)),
        "current_commit_speed": float(
            recent_commit_deltas[-1] if recent_commit_deltas else 0.0
        ),
        "recent_commit_speed": float(
            np.mean(recent_commit_deltas) if recent_commit_deltas else 0.0
        ),
        "past_oscillation_count": float(_past_oscillations(answers)),
        "distinct_answers_seen": float(len({answer for answer in answers if answer})),
    }

    # Defensive postcondition: downstream parquet writers and scikit-learn
    # should never receive objects, NaNs, or infinities from this extractor.
    for name, value in features.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"feature {name!r} is not numeric: {type(value).__name__}")
        if not np.isfinite(float(value)):
            raise ValueError(f"feature {name!r} is not finite")
    return {name: float(value) for name, value in features.items()}
