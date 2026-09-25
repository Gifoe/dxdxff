from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from neuroez_c.raw_brainbert_data import normalize_channel_name as _repository_normalize_channel_name


class LedgerSchemaError(ValueError):
    pass


_ALIASES: dict[str, tuple[str, ...]] = {
    "outer_fold": ("fold_idx", "fold_id", "outer_fold"),
    "subject_id": ("subject_id", "patient_id"),
    "center": ("center", "source_center", "source_dataset"),
    "channel_name_original": ("channel_name", "channel_names_norm", "channel_name_norm", "contact_name"),
    "clinical_true_ez": ("clinical_true_ez", "true_ez", "label_ez", "ez_label"),
    "clinical_true_nez": ("clinical_true_nez", "true_nez", "label_nez", "nez_label"),
    "old_v3_score_ez": ("old_v3_score_ez", "score_ez_probability", "score_ez_final", "score_ez", "score_eval"),
    "old_v3_rank": ("old_v3_rank", "rank_ez_desc", "rank_eval", "rank"),
    "old_v3_pred_ez": ("old_v3_pred_ez", "predicted_ez", "pred_topk", "pred_ez"),
}


def normalize_channel_name(value: Any) -> str:
    """Use the repository's canonical channel semantics without erasing primes."""
    return _repository_normalize_channel_name(value)


def _first_present(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    available = set(columns)
    return next((name for name in aliases if name in available), None)


def _binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise LedgerSchemaError(f"{name} must be present and binary")
    return values.astype(int)


def build_canonical_ledger(frame: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map a V3 OOF ledger to A12's explicit patient-channel contract."""
    mapping = {target: _first_present(frame.columns, aliases) for target, aliases in _ALIASES.items()}
    missing = [key for key in ("outer_fold", "subject_id", "channel_name_original", "clinical_true_ez", "clinical_true_nez", "old_v3_score_ez", "old_v3_pred_ez") if mapping[key] is None]
    if missing:
        raise LedgerSchemaError(f"V3 ledger missing required semantic fields: {missing}")
    out = pd.DataFrame(index=frame.index)
    for target, source in mapping.items():
        if source is not None:
            out[target] = frame[source]
    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    if out["subject_id"].eq("").any():
        raise LedgerSchemaError("subject_id cannot be empty")
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="coerce")
    if out["outer_fold"].isna().any():
        raise LedgerSchemaError("outer_fold must be numeric")
    out["outer_fold"] = out["outer_fold"].astype(int)
    out["center"] = out.get("center", "unknown").fillna("unknown").astype(str)
    out["channel_name_original"] = out["channel_name_original"].astype(str)
    out["channel_name_norm"] = out["channel_name_original"].map(normalize_channel_name)
    if out["channel_name_norm"].eq("").any():
        raise LedgerSchemaError("channel_name cannot normalize to empty")
    out["clinical_true_ez"] = _binary(out["clinical_true_ez"], "clinical_true_ez")
    out["clinical_true_nez"] = _binary(out["clinical_true_nez"], "clinical_true_nez")
    if not (out["clinical_true_ez"] + out["clinical_true_nez"] == 1).all():
        raise LedgerSchemaError("clinical_true_ez and clinical_true_nez must be complementary")
    out["old_v3_score_ez"] = pd.to_numeric(out["old_v3_score_ez"], errors="coerce")
    if out["old_v3_score_ez"].isna().any():
        raise LedgerSchemaError("old_v3_score_ez must be finite")
    out["old_v3_pred_ez"] = _binary(out["old_v3_pred_ez"], "old_v3_pred_ez")
    out["old_v3_selected"] = out["old_v3_pred_ez"]
    if "old_v3_rank" in out:
        out["old_v3_rank"] = pd.to_numeric(out["old_v3_rank"], errors="coerce")
    if "old_v3_rank" not in out or out["old_v3_rank"].isna().any():
        out["old_v3_rank"] = out.groupby("subject_id")["old_v3_score_ez"].rank(ascending=False, method="first")
    out["old_v3_rank"] = out["old_v3_rank"].astype(int)
    duplicate = out.duplicated(["subject_id", "channel_name_norm"], keep=False)
    if duplicate.any() and strict:
        preview = out.loc[duplicate, ["subject_id", "channel_name_original", "channel_name_norm"]].head(20).to_dict("records")
        raise LedgerSchemaError(f"duplicate subject-channel rows after normalization: {preview}")
    out["old_v3_k"] = out.groupby("subject_id")["old_v3_selected"].transform("sum").astype(int)
    out["n_patient_channels"] = out.groupby("subject_id")["subject_id"].transform("size").astype(int)
    out["label_semantics"] = "clinical_true_nez=1,clinical_true_ez=1"
    out["prediction_semantics"] = "pred_ez=1 means selected as EZ"
    keep = [
        "subject_id", "center", "outer_fold", "channel_name_original", "channel_name_norm", "clinical_true_nez",
        "clinical_true_ez", "old_v3_score_ez", "old_v3_rank", "old_v3_pred_ez", "old_v3_selected", "old_v3_k",
        "n_patient_channels", "label_semantics", "prediction_semantics",
    ]
    return out[keep].sort_values(["outer_fold", "subject_id", "old_v3_rank"], kind="mergesort").reset_index(drop=True), {"mapping": mapping, "columns": keep}
