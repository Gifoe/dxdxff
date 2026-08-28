"""CPU-only utilities for the ReVA-DLM Mini G0/G1 audit."""

from .parser import (
    PARSER_NAME,
    PARSER_VERSION,
    answers_equal,
    canonical_decimal,
    extract_gsm8k_answer,
    parse_gsm8k_answer,
)
from .progress import (
    DEFAULT_CHECKPOINTS,
    PROGRESS_MAPPING_RULE,
    build_progress_mapping,
    checkpoint_metadata,
    nearest_step_number,
)
from .trajectory import (
    DEFAULT_MASK_TOKEN_ID,
    TrajectoryState,
    flatten_true_indices_history,
    flatten_x0_history,
    load_trajectory,
    reconstruct_final_token_ids,
    reconstruct_states,
)

__all__ = [
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_MASK_TOKEN_ID",
    "PARSER_NAME",
    "PARSER_VERSION",
    "PROGRESS_MAPPING_RULE",
    "TrajectoryState",
    "answers_equal",
    "build_progress_mapping",
    "canonical_decimal",
    "checkpoint_metadata",
    "extract_gsm8k_answer",
    "flatten_true_indices_history",
    "flatten_x0_history",
    "load_trajectory",
    "nearest_step_number",
    "parse_gsm8k_answer",
    "reconstruct_final_token_ids",
    "reconstruct_states",
]
