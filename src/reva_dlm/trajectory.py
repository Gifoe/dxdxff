"""CPU-only reconstruction of Prophet decoding trajectories.

Prophet saves raw model argmax predictions in ``x0_history`` *before* it
overwrites previously committed positions.  Consequently, concatenating the
history and decoding its last row is not a reconstruction of the generated
output.  This module replays ``true_indices_history`` in block-major order and
uses the raw prediction from the same step for every newly committed token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch


DEFAULT_MASK_TOKEN_ID: Final[int] = 126336
REQUIRED_TRAJECTORY_KEYS: Final[tuple[str, ...]] = (
    "x0_history",
    "true_indices_history",
    "prompt_token_len",
)


@dataclass(frozen=True)
class TrajectoryState:
    """Generation-only candidate and commitment state after one iteration."""

    history_index: int
    step_number: int
    block_index: int
    step_in_block: int
    candidate_token_ids: torch.Tensor
    committed_token_ids: torch.Tensor
    committed_mask: torch.Tensor
    committed_count: int
    newly_committed_positions: tuple[int, ...]

    @property
    def masked_count(self) -> int:
        return int(self.committed_mask.numel()) - self.committed_count

    @property
    def committed_ratio(self) -> float:
        return self.committed_count / int(self.committed_mask.numel())

    @property
    def masked_ratio(self) -> float:
        return 1.0 - self.committed_ratio


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Load a trusted, fixed Prophet source file onto CPU.

    Prophet trajectory checkpoints contain Python lists and metadata, so the
    trusted public source requires ``weights_only=False``.  The argument is
    explicit to avoid behaviour changes across PyTorch releases.
    """

    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"trajectory must be a dict, got {type(loaded).__name__}")
    missing = [key for key in REQUIRED_TRAJECTORY_KEYS if key not in loaded]
    if missing:
        raise KeyError(f"trajectory is missing required keys: {missing}")
    return loaded


def _x0_rows_with_coordinates(
    x0_history: Sequence[Any] | torch.Tensor,
) -> list[tuple[int, int, torch.Tensor]]:
    rows: list[tuple[int, int, torch.Tensor]] = []

    if isinstance(x0_history, torch.Tensor):
        if x0_history.ndim == 2:
            blocks: list[Any] = [x0_history]
        elif x0_history.ndim == 3:
            blocks = list(x0_history)
        else:
            raise ValueError(
                "x0_history tensor must have shape [steps, seq] or "
                "[blocks, steps, seq]"
            )
    elif isinstance(x0_history, Sequence) and not isinstance(x0_history, (str, bytes)):
        blocks = list(x0_history)
    else:
        raise TypeError("x0_history must be a tensor or block sequence")

    for block_index, block in enumerate(blocks):
        if isinstance(block, torch.Tensor):
            if block.ndim == 1:
                step_values: list[Any] = [block]
            elif block.ndim == 2:
                step_values = list(block)
            else:
                raise ValueError(
                    f"x0_history block {block_index} must be 1-D or 2-D, "
                    f"got shape {tuple(block.shape)}"
                )
        elif isinstance(block, Sequence) and not isinstance(block, (str, bytes)):
            step_values = list(block)
        else:
            raise TypeError(f"x0_history block {block_index} is not tensor-like")

        for step_in_block, value in enumerate(step_values):
            row = torch.as_tensor(value).detach().to(device="cpu")
            if row.ndim == 2 and row.shape[0] == 1:
                row = row.squeeze(0)
            if row.ndim != 1:
                raise ValueError(
                    f"x0_history[{block_index}][{step_in_block}] must be a token row, "
                    f"got shape {tuple(row.shape)}"
                )
            rows.append((block_index, step_in_block, row))

    if not rows:
        raise ValueError("x0_history is empty")
    width = int(rows[0][2].numel())
    if width == 0 or any(int(row.numel()) != width for _, _, row in rows):
        raise ValueError("x0_history rows must have one common non-zero sequence length")
    return rows


def flatten_x0_history(x0_history: Sequence[Any] | torch.Tensor) -> torch.Tensor:
    """Flatten ``x0_history`` in official block-major/local-step order."""

    rows = _x0_rows_with_coordinates(x0_history)
    return torch.stack([row for _, _, row in rows], dim=0)


def _normalize_commit_indices(value: Any, *, label: str) -> torch.Tensor:
    indices = torch.as_tensor(value, dtype=torch.long).detach().to(device="cpu")
    if indices.numel() == 0:
        return indices.reshape(0, 2)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"{label} must have shape [N, 2], got {tuple(indices.shape)}")
    return indices


def flatten_true_indices_history(true_indices_history: Sequence[Any]) -> list[torch.Tensor]:
    """Flatten commit-index tensors in block-major/local-step order."""

    if not isinstance(true_indices_history, Sequence) or isinstance(
        true_indices_history, (str, bytes)
    ):
        raise TypeError("true_indices_history must be a block sequence")

    flattened: list[torch.Tensor] = []
    for block_index, block in enumerate(true_indices_history):
        if isinstance(block, torch.Tensor):
            # A 2-D [N, 2] tensor denotes one step; a 3-D tensor can represent
            # multiple fixed-width steps.  Published Prophet data uses lists
            # because N may vary.
            if block.ndim == 2:
                step_values: list[Any] = [block]
            elif block.ndim == 3:
                step_values = list(block)
            else:
                raise ValueError(
                    f"true_indices_history block {block_index} has invalid shape "
                    f"{tuple(block.shape)}"
                )
        elif isinstance(block, Sequence) and not isinstance(block, (str, bytes)):
            step_values = list(block)
        else:
            raise TypeError(
                f"true_indices_history block {block_index} is not a step sequence"
            )

        for step_in_block, value in enumerate(step_values):
            flattened.append(
                _normalize_commit_indices(
                    value,
                    label=f"true_indices_history[{block_index}][{step_in_block}]",
                )
            )

    if not flattened:
        raise ValueError("true_indices_history is empty")
    return flattened


def reconstruct_states(
    trajectory_or_x0_history: Mapping[str, Any] | Sequence[Any] | torch.Tensor,
    true_indices_history: Sequence[Any] | None = None,
    prompt_token_len: int | None = None,
    *,
    generation_length: int | None = None,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
) -> list[TrajectoryState]:
    """Replay a Prophet trajectory and return every post-iteration state.

    The first argument can be a loaded trajectory mapping or ``x0_history``.
    Candidates are generation-only token IDs.  At each step the candidate is
    the raw x0 row with all *past* committed tokens overwritten by replayed
    values; tokens committed at the current step use that same raw row.

    ``gen_ids`` is intentionally neither required nor read: it is absent from
    the published target files despite being advertised by the README.
    """

    if isinstance(trajectory_or_x0_history, Mapping):
        record = trajectory_or_x0_history
        missing = [key for key in REQUIRED_TRAJECTORY_KEYS if key not in record]
        if missing:
            raise KeyError(f"trajectory is missing required keys: {missing}")
        x0_history = record["x0_history"]
        if true_indices_history is None:
            true_indices_history = record["true_indices_history"]
        if prompt_token_len is None:
            prompt_token_len = int(record["prompt_token_len"])
    else:
        x0_history = trajectory_or_x0_history

    if true_indices_history is None or prompt_token_len is None:
        raise TypeError(
            "true_indices_history and prompt_token_len are required when the first "
            "argument is not a trajectory mapping"
        )
    if isinstance(prompt_token_len, bool) or not isinstance(prompt_token_len, int):
        raise TypeError("prompt_token_len must be an integer")
    if prompt_token_len < 0:
        raise ValueError("prompt_token_len must be non-negative")

    coordinate_rows = _x0_rows_with_coordinates(x0_history)
    commit_steps = flatten_true_indices_history(true_indices_history)
    if len(coordinate_rows) != len(commit_steps):
        raise ValueError(
            "history length mismatch: "
            f"{len(coordinate_rows)} x0 rows vs {len(commit_steps)} commit steps"
        )

    sequence_length = int(coordinate_rows[0][2].numel())
    inferred_generation_length = sequence_length - prompt_token_len
    if generation_length is None:
        generation_length = inferred_generation_length
    if isinstance(generation_length, bool) or not isinstance(generation_length, int):
        raise TypeError("generation_length must be an integer")
    if generation_length <= 0:
        raise ValueError("generation_length must be positive")
    if prompt_token_len + generation_length != sequence_length:
        raise ValueError(
            "prompt_token_len + generation_length must equal x0 sequence length: "
            f"{prompt_token_len} + {generation_length} != {sequence_length}"
        )

    committed_ids = torch.full(
        (generation_length,), int(mask_token_id), dtype=coordinate_rows[0][2].dtype
    )
    committed_mask = torch.zeros(generation_length, dtype=torch.bool)
    states: list[TrajectoryState] = []

    for history_index, ((block_index, step_in_block, raw_row), indices) in enumerate(
        zip(coordinate_rows, commit_steps, strict=True)
    ):
        candidate = raw_row[prompt_token_len:].clone()
        # Raw x0 may change tokens committed during earlier iterations.  Replay
        # those values before interpreting the current candidate.
        candidate[committed_mask] = committed_ids[committed_mask]

        relative_positions: list[int] = []
        if indices.numel():
            if torch.any(indices[:, 0] != 0):
                raise ValueError("only batch-size-one Prophet trajectories are supported")
            for absolute_position in indices[:, 1].tolist():
                relative_position = int(absolute_position) - prompt_token_len
                if relative_position < 0 or relative_position >= generation_length:
                    raise ValueError(
                        "committed absolute position lies outside generated segment: "
                        f"{absolute_position}"
                    )
                if bool(committed_mask[relative_position]):
                    raise ValueError(
                        f"generated position {relative_position} was committed more than once"
                    )
                token_id = raw_row[int(absolute_position)]
                committed_ids[relative_position] = token_id
                committed_mask[relative_position] = True
                # This assignment is redundant for an unmodified raw candidate,
                # but states the invariant explicitly: a current-step commit uses
                # the token from the same x0 row.
                candidate[relative_position] = token_id
                relative_positions.append(relative_position)

        states.append(
            TrajectoryState(
                history_index=history_index,
                step_number=history_index + 1,
                block_index=block_index,
                step_in_block=step_in_block,
                candidate_token_ids=candidate,
                committed_token_ids=committed_ids.clone(),
                committed_mask=committed_mask.clone(),
                committed_count=int(committed_mask.sum().item()),
                newly_committed_positions=tuple(relative_positions),
            )
        )

    return states


def reconstruct_final_token_ids(
    trajectory_or_x0_history: Mapping[str, Any] | Sequence[Any] | torch.Tensor,
    true_indices_history: Sequence[Any] | None = None,
    prompt_token_len: int | None = None,
    *,
    generation_length: int | None = None,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    require_all_committed: bool = True,
) -> torch.Tensor:
    """Return generation-only final token IDs reconstructed from commits."""

    states = reconstruct_states(
        trajectory_or_x0_history,
        true_indices_history,
        prompt_token_len,
        generation_length=generation_length,
        mask_token_id=mask_token_id,
    )
    final_state = states[-1]
    if require_all_committed and final_state.committed_count != final_state.committed_mask.numel():
        raise ValueError(
            "trajectory ended before every generated position was committed: "
            f"{final_state.committed_count}/{final_state.committed_mask.numel()}"
        )
    # A complete final candidate and committed replay are identical.  Returning
    # the committed replay makes the source of truth unambiguous.
    return final_state.committed_token_ids.clone()


# Explicit aliases used in some analysis scripts.
reconstruct_trajectory = reconstruct_states
reconstruct_final_ids = reconstruct_final_token_ids


__all__ = [
    "DEFAULT_MASK_TOKEN_ID",
    "REQUIRED_TRAJECTORY_KEYS",
    "TrajectoryState",
    "flatten_true_indices_history",
    "flatten_x0_history",
    "load_trajectory",
    "reconstruct_final_ids",
    "reconstruct_final_token_ids",
    "reconstruct_states",
    "reconstruct_trajectory",
]
