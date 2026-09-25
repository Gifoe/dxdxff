from __future__ import annotations

PATH_ENV_KEYS = {
    "feature_cache_path": "DRE_FEATURE_CACHE_PATH",
    "raw_cache_path": "DRE_RAW_CACHE_PATH",
    "output_dir": "DRE_OUTCOME_OUTPUT_DIR",
    "pretrained_model_root": "DRE_PRETRAINED_MODEL_ROOT",
}

MAIN_VARIANTS = (
    "H1_SUMMARY_ML",
    "H2_HIER_POOL",
    "H3_ATTENTION_MIL",
    "H4_MULTI_CORE",
    "H5_ANCHORED_CORE",
    "H6_ANCHORED_UOT_DESC",
    "H7_TRANSPORT_GRAPH",
    "H8_RECURRENCE",
    "H9_FM_RECURRENCE",
    "H10_LATE_FUSION",
)

AUDIT_VARIANTS = ("H0_METADATA_SHORTCUT",)

FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "labels",
        "labels_ez",
        "labels_nez",
        "label_mask",
        "clinical_ez",
        "clinical_nez",
        "clinical_soz",
        "soz",
        "ez",
        "resection_mask",
        "resection",
        "resected",
        "true_ez_count",
        "ez_channel_count",
        "ez_fraction",
        "clinical_selected_channel_set",
        "postoperative_imaging",
        "postoperative_resection_extent",
    }
)

FORBIDDEN_KEY_FRAGMENTS = (
    "task1",
    "task_1",
    "v3_score",
    "hnc_score",
    "oof_channel",
    "channel_prediction",
    "clinical_selected",
    "postoperative",
    "post_op",
    "resect",
)

MODEL_INPUT_WHITELIST = frozenset(
    {
        "feature_x",
        "raw_x",
        "fm_x",
        "seizure_mask",
        "window_mask",
        "channel_mask",
        "seizure_channel_mask",
        "window_channel_mask",
        "window_centers",
        "canonical_index",
        "topology",
    }
)

__all__ = [
    "AUDIT_VARIANTS",
    "FORBIDDEN_EXACT_KEYS",
    "FORBIDDEN_KEY_FRAGMENTS",
    "MAIN_VARIANTS",
    "MODEL_INPUT_WHITELIST",
    "PATH_ENV_KEYS",
]
