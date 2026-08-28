"""Frozen Mini G0/G1 protocol constants."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_VERSION = "reva_gsm8k_mini_v2"
RANDOM_SEED = 20260827
AUDIT_CHECKPOINTS = (0.25, 0.40, 0.50, 0.60, 0.75)
PREREGISTERED_PRIMARY_CHECKPOINT = 0.50
# This checkpoint was selected on the parser-v1.0 DEV diagnostic and is frozen
# before the parser-v1.1 rerun.  It may be used to complete an explicitly
# exploratory G1 audit, but it is not eligible for an ordinary READY cache.
FROZEN_EXPLORATORY_CHECKPOINT = 0.60
GEN_LENGTH = 256
TOTAL_STEPS = 256
BLOCK_LENGTH = 32
FEATURE_WINDOW = 16

DATASET_REPO = "YefanZhou98/DLM-Decoding-Analysis"
DATASET_REVISION = "91beb881acaa0b6edfccd88e8d19c08ec5e1225b"
TRAJECTORY_FOLDER = "question_histories_low_conf_none_index_genlen_step256_blocklen32"
TOKENIZER_REPO = "GSAI-ML/LLaDA-8B-Instruct"
TOKENIZER_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PROPHET_REVISION = "460afe41c7063a29a9893675aca07b985997bb83"

RAW_REPO_DIR = PROJECT_ROOT / "data" / "raw" / "DLM-Decoding-Analysis"
TRAJECTORY_DIR = RAW_REPO_DIR / TRAJECTORY_FOLDER
TOKENIZER_DIR = PROJECT_ROOT / "data" / "raw" / "LLaDA-8B-Instruct-tokenizer"
GSM8K_DIR = PROJECT_ROOT / "data" / "raw" / "GSM8K"
GSM8K_TEST_FILE = GSM8K_DIR / "main" / "test-00000-of-00001.parquet"
CACHE_DIR = PROJECT_ROOT / "cache" / CACHE_VERSION


GATE_PROTOCOL_VERSION = "2.0.0"
G0_GATE_PROTOCOL = {
    "version": GATE_PROTOCOL_VERSION,
    "strong_checkpoint": PREREGISTERED_PRIMARY_CHECKPOINT,
    "strong_min_rcr_joint": 0.03,
    "strong_min_corruption": 30,
    "strong_min_rescue": 30,
    "fail_max_rcr_joint_exclusive": 0.01,
    "exploratory_checkpoint": FROZEN_EXPLORATORY_CHECKPOINT,
    "exploratory_origin": (
        "selected on parser-v1.0 DEV diagnostic and frozen before parser-v1.1 rerun"
    ),
    "exploratory_min_rcr_joint": 0.01,
    "exploratory_min_corruption": 10,
    "exploratory_min_rescue": 30,
    "exploratory_required_adjacent_checkpoint": 0.50,
    "ordinary_ready_eligible_statuses": ["G0_PASS", "G0_STRONG_PASS"],
    "g1_run_eligible_statuses": [
        "G0_PASS",
        "G0_STRONG_PASS",
        "G0_EXPLORATORY_PASS",
    ],
}
G1_GATE_PROTOCOL = {
    "version": GATE_PROTOCOL_VERSION,
    "model_selection": "maximum AUROC over the frozen cheap-model universe",
    "model_universe": [
        "one balanced logistic model per frozen single causal feature",
        "balanced logistic regression over all frozen causal features",
        "balanced HistGradientBoosting over all frozen causal features",
    ],
    "primary_fail_auroc_min": 0.80,
    "primary_fail_ci_lower_min": 0.70,
    "key_matched_tasks": ["A_same_current_wrong", "C_same_final_correct"],
    "matched_fail_auroc_min": 0.80,
    "matched_fail_ci_lower_min": 0.70,
    "strong_primary_ci_upper_max": 0.70,
    "minimum_count_evaluable_matched_positive": 10,
    "cv_max_folds": 5,
    "cv_repeats": 3,
    "bootstrap_samples": 1000,
    "hard_threshold": 0.5,
    "n_jobs": 1,
    "origin": "operationalized after v1.0 DEV audit and frozen before parser-v1.1 rerun",
}
