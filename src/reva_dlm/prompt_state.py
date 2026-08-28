"""Exact Prophet prompt reconstruction and frozen denoising-input semantics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .trajectory import flatten_true_indices_history, flatten_x0_history


QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()
QUERY_TEMPLATE_SHA256 = hashlib.sha256(QUERY_TEMPLATE.encode("utf-8")).hexdigest()
# Frozen from external/Prophet/analysis/collect_decoding_traj_gsm8k.py at
# Prophet revision 460afe41c7063a29a9893675aca07b985997bb83.  Keeping a literal
# digest makes template drift fail at import instead of silently changing all
# prompt token IDs.
FROZEN_QUERY_TEMPLATE_SHA256 = (
    "231d269274b6a04711d192c935b7a4785da99e6823afbc817972a1d51f48425e"
)
if QUERY_TEMPLATE_SHA256 != FROZEN_QUERY_TEMPLATE_SHA256:
    raise RuntimeError("the exact frozen Prophet GSM8K QUERY_TEMPLATE has changed")
MASK_TOKEN_ID = 126336


def build_prompt_text(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    return QUERY_TEMPLATE.format(Question=question)


def load_gsm8k_test(
    path: str | Path,
    *,
    expected_count: int | None = 1319,
) -> pd.DataFrame:
    frame = pd.read_parquet(Path(path))
    if list(frame.columns) != ["question", "answer"]:
        raise AssertionError(f"unexpected GSM8K columns: {list(frame.columns)}")
    if expected_count is not None and len(frame) != expected_count:
        raise AssertionError(
            f"expected {expected_count} GSM8K test rows, got {len(frame)}"
        )
    if frame[["question", "answer"]].isna().any().any():
        raise AssertionError("GSM8K test contains missing question/answer")
    if not frame["question"].map(lambda value: isinstance(value, str)).all():
        raise AssertionError("GSM8K test contains a non-string question")
    if not frame["answer"].map(lambda value: isinstance(value, str)).all():
        raise AssertionError("GSM8K test contains a non-string answer")
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class DenoisingInputState:
    """The exact model input that produced ``x0_history[history_index]``."""

    history_index: int
    step_number: int
    input_token_ids: torch.Tensor
    generation_token_ids: torch.Tensor
    committed_mask_before_step: torch.Tensor
    committed_count_before_step: int


def reconstruct_pre_step_input(
    trajectory: dict[str, Any],
    prompt_token_ids: list[int] | torch.Tensor,
    history_index: int,
    *,
    generation_length: int = 256,
    mask_token_id: int = MASK_TOKEN_ID,
) -> DenoisingInputState:
    """Reconstruct the forward input that generated raw x0 at one history row.

    Commits from indices strictly smaller than ``history_index`` are replayed;
    all other generated positions remain the LLaDA mask token.  This is the
    pre-forward ``x`` in Prophet's ``generate.py``.  The current step's commits
    are deliberately excluded because they are selected only after that model
    forward produces x0/logits.
    """

    if history_index < 0:
        raise ValueError("history_index must be non-negative")
    raw = flatten_x0_history(trajectory["x0_history"])
    commits = flatten_true_indices_history(trajectory["true_indices_history"])
    if history_index >= len(commits) or len(commits) != raw.shape[0]:
        raise ValueError("history_index is outside the frozen trajectory")
    prompt = torch.as_tensor(prompt_token_ids, dtype=torch.long).detach().cpu().flatten()
    prompt_len = int(trajectory["prompt_token_len"])
    if prompt.numel() != prompt_len:
        raise AssertionError(
            f"prompt token length mismatch: {prompt.numel()} != {prompt_len}"
        )
    generation = torch.full((generation_length,), int(mask_token_id), dtype=torch.long)
    committed = torch.zeros(generation_length, dtype=torch.bool)
    for prior_index in range(history_index):
        for row in commits[prior_index]:
            if int(row[0]) != 0:
                raise AssertionError("only batch-size-one trajectories are supported")
            absolute = int(row[1])
            relative = absolute - prompt_len
            if not 0 <= relative < generation_length:
                raise AssertionError("commit is outside generated segment")
            if committed[relative]:
                raise AssertionError("generated position committed more than once")
            generation[relative] = raw[prior_index, absolute]
            committed[relative] = True
    full = torch.cat([prompt, generation])
    if full.numel() != raw.shape[1]:
        raise AssertionError("reconstructed pre-step width differs from x0 history")
    return DenoisingInputState(
        history_index=history_index,
        step_number=history_index + 1,
        input_token_ids=full,
        generation_token_ids=generation,
        committed_mask_before_step=committed,
        committed_count_before_step=int(committed.sum().item()),
    )


__all__ = [
    "DenoisingInputState",
    "FROZEN_QUERY_TEMPLATE_SHA256",
    "MASK_TOKEN_ID",
    "QUERY_TEMPLATE",
    "QUERY_TEMPLATE_SHA256",
    "build_prompt_text",
    "load_gsm8k_test",
    "reconstruct_pre_step_input",
]
