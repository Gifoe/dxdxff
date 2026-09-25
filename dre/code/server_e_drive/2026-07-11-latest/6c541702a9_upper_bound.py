from __future__ import annotations

import glob
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from baseline_common.reporting import frame_to_markdown
from task1_baselines.fold_protocol import fold_manifest_hash, freeze_v3_fold_manifest
from task1_baselines.metrics import compute_task1_metrics


class UpperBoundAuditError(ValueError):
    """Raised when an OOF ledger violates the upper-bound audit contract."""


LABEL_NEZ_COLUMNS = ("label_nez", "clinical_true_nez", "true_nez")
LABEL_EZ_COLUMNS = ("clinical_true_ez", "true_ez")
NEZ_PROBABILITY_COLUMNS = ("score_nez_probability", "probability_nez", "prob_nez")
EZ_PROBABILITY_COLUMNS = ("score_ez_probability", "probability_ez", "prob_ez")
NEZ_GENERIC_COLUMNS = ("score_nez",)
EZ_GENERIC_COLUMNS = ("score_ez",)
FORBIDDEN_THRESHOLD_SOURCES = ("outer_test", "test_label", "oracle")


def normalize_channel_name(value: Any) -> str:
    """Normalize only case and ordinary whitespace; preserve electrode identity."""
    return str(value).strip().upper().replace(" ", "")


def _binary(series: pd.Series, name: str, *, allow_missing: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce" if allow_missing else "raise")
    present = values.dropna()
    if not present.isin([0, 1]).all():
        bad = present[~present.isin([0, 1])].head(5).tolist()
        raise UpperBoundAuditError(f"{name} must contain only 0/1; examples={bad}")
    return values.astype("Int64" if allow_missing or values.isna().any() else int)


def _first(columns: Iterable[str], available: Iterable[str]) -> str | None:
    names = set(available)
    return next((column for column in columns if column in names), None)


def _assert_consistent(candidates: list[tuple[str, pd.Series]], kind: str) -> pd.Series:
    if not candidates:
        raise UpperBoundAuditError(f"Ledger has no compatible {kind} column.")
    name, canonical = candidates[0]
    for other_name, other in candidates[1:]:
        mismatch = canonical.notna() & other.notna() & (canonical.astype(float) != other.astype(float))
        if mismatch.any():
            indices = mismatch[mismatch].index[:5].tolist()
            raise UpperBoundAuditError(
                f"Conflicting {kind} columns {name!r} and {other_name!r} at rows {indices}."
            )
    return canonical


def _infer_model_seed(path: Path) -> tuple[str, int, dict[str, str]]:
    text = path.as_posix()
    lower = text.lower()
    if "v3_oof_channel_ledger" in lower:
        model = "v3"
        model_rule = "filename contains v3_oof_channel_ledger"
    else:
        parent = path.parent.name
        model = re.sub(r"[^A-Za-z0-9_.-]+", "_", parent or path.stem).strip("_").lower()
        model_rule = "parent directory name"
    seed_match = re.search(r"(?:seed[_-]?|[/_-]s)(\d+)(?:\D|$)", lower)
    seed = int(seed_match.group(1)) if seed_match else 0
    seed_rule = "path seed token" if seed_match else "default 0 (no path seed token)"
    return model, seed, {"model": model_rule, "seed": seed_rule}


def standardize_ledger(
    ledger: pd.DataFrame | str | Path,
    *,
    source_path: str | Path | None = None,
    default_model: str | None = None,
    default_seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert a channel OOF ledger to the strict internal audit schema."""
    if isinstance(ledger, (str, Path)):
        path = Path(ledger)
        raw = pd.read_csv(path)
    else:
        raw = ledger.copy()
        path = Path(source_path or "<memory>")
    if raw.empty:
        raise UpperBoundAuditError(f"Ledger is empty: {path}")

    subject_column = _first(("subject_id", "patient_id"), raw.columns)
    channel_column = _first(("channel_name", "channel_name_norm", "contact_name"), raw.columns)
    fold_column = _first(("outer_fold", "fold_idx"), raw.columns)
    if subject_column is None or channel_column is None:
        raise UpperBoundAuditError(f"Ledger lacks subject/channel keys: {path}")

    inferred_model, inferred_seed, inference = _infer_model_seed(path)
    output = pd.DataFrame(index=raw.index)
    output["model"] = raw["model"].astype(str) if "model" in raw else str(default_model or inferred_model)
    output["seed"] = (
        pd.to_numeric(raw["seed"], errors="raise").astype(int)
        if "seed" in raw
        else int(inferred_seed if default_seed is None else default_seed)
    )
    output["subject_id"] = raw[subject_column].astype(str).str.strip()
    output["center"] = raw["center"].astype(str).str.strip().str.lower() if "center" in raw else pd.NA
    output["outer_fold"] = (
        pd.to_numeric(raw[fold_column], errors="raise").astype(int) if fold_column else pd.NA
    )
    output["channel_name"] = raw[channel_column].map(normalize_channel_name)

    label_candidates: list[tuple[str, pd.Series]] = []
    for column in LABEL_NEZ_COLUMNS:
        if column in raw:
            label_candidates.append((column, _binary(raw[column], column)))
    for column in LABEL_EZ_COLUMNS:
        if column in raw:
            label_candidates.append((column, 1 - _binary(raw[column], column)))
    output["label_nez"] = _assert_consistent(label_candidates, "NEZ label").astype(int)

    probability_columns = [
        column for column in (*NEZ_PROBABILITY_COLUMNS, *EZ_PROBABILITY_COLUMNS) if column in raw
    ]
    for column in probability_columns:
        values = pd.to_numeric(raw[column], errors="raise").astype(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise UpperBoundAuditError(f"Probability column {column!r} must be finite and in [0,1].")
    for direction, columns in (
        ("NEZ", [column for column in NEZ_PROBABILITY_COLUMNS if column in raw]),
        ("EZ", [column for column in EZ_PROBABILITY_COLUMNS if column in raw]),
    ):
        if len(columns) > 1:
            reference = pd.to_numeric(raw[columns[0]], errors="raise").to_numpy(dtype=float)
            for column in columns[1:]:
                other = pd.to_numeric(raw[column], errors="raise").to_numpy(dtype=float)
                if not np.allclose(reference, other, atol=1e-10, rtol=0):
                    raise UpperBoundAuditError(
                        f"Conflicting {direction} probability columns {columns[0]!r} and {column!r}."
                    )

    nez_probability = _first(NEZ_PROBABILITY_COLUMNS, raw.columns)
    ez_probability = _first(EZ_PROBABILITY_COLUMNS, raw.columns)
    nez_generic = _first(NEZ_GENERIC_COLUMNS, raw.columns)
    ez_generic = _first(EZ_GENERIC_COLUMNS, raw.columns)
    score_kind = "hard_only"
    if nez_probability is not None or ez_probability is not None:
        score_kind = "probability"
        if nez_probability is not None:
            score_nez = pd.to_numeric(raw[nez_probability], errors="raise").astype(float)
            score_ez = 1.0 - score_nez
        else:
            score_ez = pd.to_numeric(raw[ez_probability], errors="raise").astype(float)
            score_nez = 1.0 - score_ez
        if nez_probability is not None and ez_probability is not None:
            supplied_ez = pd.to_numeric(raw[ez_probability], errors="raise").astype(float)
            if not np.allclose(score_ez, supplied_ez, atol=1e-10, rtol=0):
                raise UpperBoundAuditError("NEZ/EZ probability columns are not complements.")
    elif nez_generic is not None or ez_generic is not None:
        score_kind = "generic"
        if nez_generic is not None:
            score_nez = pd.to_numeric(raw[nez_generic], errors="raise").astype(float)
            score_ez = -score_nez
        else:
            score_ez = pd.to_numeric(raw[ez_generic], errors="raise").astype(float)
            score_nez = -score_ez
        if not np.isfinite(score_nez).all() or not np.isfinite(score_ez).all():
            raise UpperBoundAuditError("Generic score columns must be finite.")
    else:
        score_nez = pd.Series(np.nan, index=raw.index, dtype=float)
        score_ez = pd.Series(np.nan, index=raw.index, dtype=float)
    output["score_nez"] = score_nez
    output["score_ez"] = score_ez
    output["score_kind"] = score_kind

    predicted_candidates: list[tuple[str, pd.Series]] = []
    if "predicted_nez" in raw:
        predicted_candidates.append(("predicted_nez", _binary(raw["predicted_nez"], "predicted_nez", allow_missing=True)))
    if "predicted_ez" in raw:
        predicted_candidates.append(("predicted_ez", 1 - _binary(raw["predicted_ez"], "predicted_ez", allow_missing=True)))
    if predicted_candidates:
        output["predicted_nez"] = _assert_consistent(predicted_candidates, "prediction").astype("Int64")
    else:
        output["predicted_nez"] = pd.Series(pd.NA, index=raw.index, dtype="Int64")

    output["selected_threshold"] = (
        pd.to_numeric(raw["selected_threshold"], errors="coerce").astype(float)
        if "selected_threshold" in raw
        else np.nan
    )
    output["threshold_source"] = raw["threshold_source"].astype(str) if "threshold_source" in raw else pd.NA
    output["source_path"] = str(path.resolve()) if str(path) != "<memory>" else "<memory>"
    for metadata in ("cohort_hash", "fold_ledger_hash", "cohort_name", "config_hash"):
        output[metadata] = raw[metadata] if metadata in raw else pd.NA

    has_prediction = output["predicted_nez"].notna().any()
    if not has_prediction and score_kind == "probability" and output["selected_threshold"].notna().all():
        output["predicted_nez"] = (output["score_nez"] >= output["selected_threshold"]).astype("Int64")
        inference["prediction"] = "reconstructed from probability score and selected_threshold"
    elif has_prediction:
        inference["prediction"] = "stored predicted_nez/predicted_ez"
    else:
        inference["prediction"] = "unavailable"

    audit = {
        "source_path": output["source_path"].iloc[0],
        "rows": int(len(output)),
        "model_inference": "stored model column" if "model" in raw else inference["model"],
        "seed_inference": "stored seed column" if "seed" in raw else inference["seed"],
        "prediction_source": inference["prediction"],
        "score_kind": score_kind,
        "source_columns": list(raw.columns),
    }
    return output.reset_index(drop=True), audit


def _key_index(table: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(table[["subject_id", "channel_name"]])


def validate_candidate(
    table: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate one model/seed candidate without mutating the ledger."""
    if table[["model", "seed"]].drop_duplicates().shape[0] != 1:
        raise UpperBoundAuditError("validate_candidate requires exactly one model/seed group.")
    candidate_id = f"{table['model'].iloc[0]}::seed_{int(table['seed'].iloc[0])}"
    duplicate = table.duplicated(["subject_id", "channel_name"], keep=False)
    if duplicate.any():
        examples = table.loc[duplicate, ["subject_id", "channel_name"]].head(5).to_dict("records")
        raise UpperBoundAuditError(f"{candidate_id} has duplicate patient-channel keys: {examples}")

    manifest = fold_manifest.set_index("subject_id")
    unknown = sorted(set(table["subject_id"]) - set(manifest.index))
    if unknown:
        raise UpperBoundAuditError(f"{candidate_id} contains subjects outside frozen old-90: {unknown[:10]}")
    missing_subjects = sorted(set(manifest.index) - set(table["subject_id"]))
    if table.groupby("subject_id").size().min() < 1:
        raise UpperBoundAuditError(f"{candidate_id} has a patient with no channels.")
    joined = table.join(manifest[["center", "outer_fold"]], on="subject_id", rsuffix="_manifest")
    if table["outer_fold"].isna().any() or table["center"].isna().any():
        raise UpperBoundAuditError(f"{candidate_id} must provide center and outer_fold for every row.")
    fold_present = joined["outer_fold"].notna()
    fold_mismatch = fold_present & (joined["outer_fold"].astype("Int64") != joined["outer_fold_manifest"].astype("Int64"))
    if fold_mismatch.any():
        raise UpperBoundAuditError(f"{candidate_id} outer_fold conflicts with frozen manifest.")
    center_present = joined["center"].notna()
    center_mismatch = center_present & (
        joined["center"].astype(str).str.lower() != joined["center_manifest"].astype(str).str.lower()
    )
    if center_mismatch.any():
        raise UpperBoundAuditError(f"{candidate_id} center conflicts with frozen manifest.")

    if table["score_kind"].iloc[0] != "hard_only":
        if not np.isfinite(table["score_nez"]).all() or not np.isfinite(table["score_ez"]).all():
            raise UpperBoundAuditError(f"{candidate_id} has NaN/inf continuous scores.")

    canonical_index = _key_index(canonical)
    candidate_index = _key_index(table)
    missing_keys = canonical_index.difference(candidate_index)
    extra_keys = candidate_index.difference(canonical_index)
    common = candidate_index.intersection(canonical_index)
    left = table.set_index(["subject_id", "channel_name"]).loc[common, "label_nez"]
    right = canonical.set_index(["subject_id", "channel_name"]).loc[common, "label_nez"]
    label_conflicts = int((left.astype(int) != right.astype(int)).sum())
    if label_conflicts:
        raise UpperBoundAuditError(f"{candidate_id} has {label_conflicts} label conflicts with V3 canonical ledger.")

    warnings: list[str] = []
    if table["predicted_nez"].isna().any():
        warnings.append("strict_prediction_unavailable")
    if table["score_kind"].iloc[0] == "hard_only":
        warnings.append("ranking_unavailable_hard_only")

    threshold_rows = table[table["selected_threshold"].notna()]
    if not threshold_rows.empty:
        for fold, group in threshold_rows.groupby("outer_fold", dropna=False):
            if group["selected_threshold"].nunique(dropna=False) != 1:
                raise UpperBoundAuditError(f"{candidate_id} fold {fold} has multiple selected_threshold values.")
            sources = group["threshold_source"].dropna().astype(str).str.lower().unique().tolist()
            if len(sources) > 1:
                raise UpperBoundAuditError(f"{candidate_id} fold {fold} has multiple threshold_source values.")
            if sources and any(token in sources[0] for token in FORBIDDEN_THRESHOLD_SOURCES):
                raise UpperBoundAuditError(f"{candidate_id} uses forbidden formal threshold source {sources[0]!r}.")
        if table["score_kind"].iloc[0] == "probability" and table["predicted_nez"].notna().all():
            recomputed = (table["score_nez"] >= table["selected_threshold"]).astype(int)
            mismatch = recomputed != table["predicted_nez"].astype(int)
            if mismatch.any():
                message = f"{candidate_id} has {int(mismatch.sum())} stored prediction/threshold mismatches."
                if strict:
                    raise UpperBoundAuditError(message)
                warnings.append(message)

    protocol_fields: dict[str, list[str]] = {}
    for field in ("cohort_hash", "fold_ledger_hash", "cohort_name", "config_hash"):
        values = table[field].dropna().astype(str).unique().tolist()
        protocol_fields[field] = values
        if len(values) > 1:
            raise UpperBoundAuditError(f"{candidate_id} has inconsistent {field} values.")
        if not values:
            warnings.append(f"missing_{field}")

    return {
        "candidate_id": candidate_id,
        "model": str(table["model"].iloc[0]),
        "seed": int(table["seed"].iloc[0]),
        "rows": int(len(table)),
        "subjects": int(table["subject_id"].nunique()),
        "missing_subjects": len(missing_subjects),
        "missing_keys": int(len(missing_keys)),
        "extra_keys": int(len(extra_keys)),
        "label_conflicts": label_conflicts,
        "aligned_for_library": not missing_subjects and not len(missing_keys) and not len(extra_keys),
        "score_kind": str(table["score_kind"].iloc[0]),
        "strict_available": bool(table["predicted_nez"].notna().all()),
        "ranking_available": bool(table["score_kind"].iloc[0] != "hard_only"),
        "warnings": warnings,
        "protocol_fields": protocol_fields,
    }


def _safe_f1(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator), dtype=float),
        where=np.asarray(denominator) != 0,
    )


def _curve_metrics(y_nez_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_nez = np.asarray(y_nez_sorted, dtype=int)
    n = len(y_nez)
    k = np.arange(n + 1)
    true_ez = int((y_nez == 0).sum())
    true_nez = n - true_ez
    cumulative_ez = np.concatenate([[0], np.cumsum(y_nez == 0)])
    cumulative_nez = k - cumulative_ez
    tp_ez = cumulative_ez
    fp_ez = cumulative_nez
    fn_ez = true_ez - tp_ez
    ez_f1 = _safe_f1(2 * tp_ez, 2 * tp_ez + fp_ez + fn_ez)
    tp_nez = true_nez - cumulative_nez
    fp_nez = true_ez - tp_ez
    fn_nez = cumulative_nez
    nez_f1 = _safe_f1(2 * tp_nez, 2 * tp_nez + fp_nez + fn_nez)
    macro = (ez_f1 + nez_f1) / 2.0
    accuracy = (tp_ez + tp_nez) / max(n, 1)
    return macro, ez_f1, nez_f1, accuracy


def _ordered_patient(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(["score_ez", "channel_name"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _best_index(macro: np.ndarray, ez: np.ndarray, allowed: np.ndarray) -> int:
    return max((int(k) for k in allowed), key=lambda k: (float(macro[k]), float(ez[k]), -k, -k))


def patient_oracle(group: pd.DataFrame, *, allow_tie_split: bool = False) -> dict[str, Any]:
    ordered = _ordered_patient(group)
    y = ordered["label_nez"].to_numpy(dtype=int)
    scores = ordered["score_ez"].to_numpy(dtype=float)
    macro, ez, nez, accuracy = _curve_metrics(y)
    if allow_tie_split:
        allowed = np.arange(len(ordered) + 1)
    else:
        boundaries = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True]) + 1
        allowed = np.r_[0, boundaries]
    k = _best_index(macro, ez, allowed)
    prediction = np.ones(len(ordered), dtype=int)
    prediction[:k] = 0
    selected = ordered.loc[: k - 1, "channel_name"].tolist() if k else []
    if k == 0:
        threshold = float("inf")
    else:
        threshold = float(scores[k - 1])
    counts = pd.Series(scores).value_counts()
    tie_fraction = float(counts[counts > 1].sum() / len(scores)) if len(scores) else 0.0
    return {
        "macro_f1": float(macro[k]),
        "ez_f1": float(ez[k]),
        "nez_f1": float(nez[k]),
        "accuracy": float(accuracy[k]),
        "threshold": threshold,
        "ez_count": int(k),
        "selected_channels": selected,
        "predicted_nez_ordered": prediction,
        "ordered_keys": list(zip(ordered["subject_id"], ordered["channel_name"])),
        "n_unique_scores": int(pd.Series(scores).nunique()),
        "score_tie_fraction": tie_fraction,
    }


def true_k_result(group: pd.DataFrame) -> dict[str, Any]:
    ordered = _ordered_patient(group)
    y = ordered["label_nez"].to_numpy(dtype=int)
    true_k = int((y == 0).sum())
    macro, ez, nez, accuracy = _curve_metrics(y)
    selected = ordered.loc[: true_k - 1, "channel_name"].tolist() if true_k else []
    return {
        "macro_f1": float(macro[true_k]),
        "ez_f1": float(ez[true_k]),
        "nez_f1": float(nez[true_k]),
        "accuracy": float(accuracy[true_k]),
        "true_k": true_k,
        "predicted_ez_count": true_k,
        "selected_channels": selected,
    }


def _patient_discrete_metrics(group: pd.DataFrame) -> dict[str, float]:
    y = group["label_nez"].to_numpy(dtype=int)
    pred = group["predicted_nez"].to_numpy(dtype=int)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "ez_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "nez_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def evaluate_candidate(table: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute strict and diagnostic metrics for one model/seed candidate."""
    candidate_id = f"{table['model'].iloc[0]}::seed_{int(table['seed'].iloc[0])}"
    patient_rows: list[dict[str, Any]] = []
    ranking = table["score_kind"].iloc[0] != "hard_only"
    strict_available = table["predicted_nez"].notna().all()
    for (subject, center, fold), group in table.groupby(["subject_id", "center", "outer_fold"], sort=True, dropna=False):
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "model": str(table["model"].iloc[0]),
            "seed": int(table["seed"].iloc[0]),
            "subject_id": subject,
            "center": center,
            "outer_fold": int(fold),
            "n_channels": int(len(group)),
            "true_ez_count": int((group["label_nez"] == 0).sum()),
        }
        if strict_available:
            strict_values = _patient_discrete_metrics(group)
            row.update({f"strict_patient_{name}": value for name, value in strict_values.items()})
        else:
            row.update({f"strict_patient_{name}": np.nan for name in ("macro_f1", "ez_f1", "nez_f1", "accuracy")})
        if ranking:
            true_k = true_k_result(group)
            oracle = patient_oracle(group)
            prefix = patient_oracle(group, allow_tie_split=True)
            row.update(
                {
                    "true_k_patient_macro_f1": true_k["macro_f1"],
                    "true_k_patient_ez_f1": true_k["ez_f1"],
                    "true_k_patient_nez_f1": true_k["nez_f1"],
                    "true_k": true_k["true_k"],
                    "predicted_ez_count": true_k["predicted_ez_count"],
                    "true_k_selected_channels": "|".join(true_k["selected_channels"]),
                    "oracle_threshold_patient_macro_f1": oracle["macro_f1"],
                    "oracle_threshold_patient_ez_f1": oracle["ez_f1"],
                    "oracle_threshold_patient_nez_f1": oracle["nez_f1"],
                    "oracle_threshold": oracle["threshold"],
                    "oracle_ez_count": oracle["ez_count"],
                    "oracle_count_error": oracle["ez_count"] - true_k["true_k"],
                    "oracle_prefix_patient_macro_f1": prefix["macro_f1"],
                    "oracle_prefix_patient_ez_f1": prefix["ez_f1"],
                    "oracle_prefix_patient_nez_f1": prefix["nez_f1"],
                    "oracle_prefix_ez_count": prefix["ez_count"],
                    "n_unique_scores": oracle["n_unique_scores"],
                    "score_tie_fraction": oracle["score_tie_fraction"],
                }
            )
            row["oracle_gain_over_strict"] = row["oracle_threshold_patient_macro_f1"] - row["strict_patient_macro_f1"]
            row["oracle_gain_over_true_k"] = row["oracle_threshold_patient_macro_f1"] - row["true_k_patient_macro_f1"]
        else:
            for column in (
                "true_k_patient_macro_f1", "true_k_patient_ez_f1", "true_k_patient_nez_f1",
                "true_k", "predicted_ez_count", "true_k_selected_channels",
                "oracle_threshold_patient_macro_f1", "oracle_threshold_patient_ez_f1",
                "oracle_threshold_patient_nez_f1", "oracle_threshold", "oracle_ez_count",
                "oracle_count_error", "oracle_prefix_patient_macro_f1", "oracle_prefix_patient_ez_f1",
                "oracle_prefix_patient_nez_f1", "oracle_prefix_ez_count", "n_unique_scores",
                "score_tie_fraction", "oracle_gain_over_strict", "oracle_gain_over_true_k",
            ):
                row[column] = np.nan
        patient_rows.append(row)
    patients = pd.DataFrame(patient_rows)

    pooled_y = table["label_nez"].to_numpy(dtype=int)
    pooled_pred = table["predicted_nez"].to_numpy(dtype=int) if strict_available else np.array([], dtype=int)
    reused_metrics: dict[str, Any] | None = None
    if strict_available and table["score_kind"].iloc[0] == "probability":
        compatible = table[["subject_id", "label_nez", "predicted_nez"]].copy()
        compatible["score_nez_probability"] = table["score_nez"].to_numpy(dtype=float)
        reused_metrics = compute_task1_metrics(compatible)
    summary: dict[str, Any] = {
        "candidate_id": candidate_id,
        "model": str(table["model"].iloc[0]),
        "seed": int(table["seed"].iloc[0]),
        "strict_count_free_patient_macro_f1": float(reused_metrics["patient_macro_f1"] if reused_metrics else patients["strict_patient_macro_f1"].mean()),
        "strict_count_free_patient_ez_f1": float(reused_metrics["patient_ez_f1"] if reused_metrics else patients["strict_patient_ez_f1"].mean()),
        "strict_count_free_patient_nez_f1": float(reused_metrics["patient_nez_f1"] if reused_metrics else patients["strict_patient_nez_f1"].mean()),
        "strict_count_free_patient_accuracy": float(reused_metrics["patient_accuracy"] if reused_metrics else patients["strict_patient_accuracy"].mean()),
        "strict_pooled_macro_f1": float(reused_metrics["f1_macro"] if reused_metrics else f1_score(pooled_y, pooled_pred, average="macro", labels=[0, 1], zero_division=0)) if strict_available else np.nan,
        "strict_pooled_ez_f1": float(reused_metrics["f1_ez"] if reused_metrics else f1_score(pooled_y, pooled_pred, pos_label=0, zero_division=0)) if strict_available else np.nan,
        "strict_pooled_nez_f1": float(reused_metrics["f1_nez"] if reused_metrics else f1_score(pooled_y, pooled_pred, pos_label=1, zero_division=0)) if strict_available else np.nan,
        "true_k_patient_macro_f1": float(patients["true_k_patient_macro_f1"].mean()),
        "true_k_patient_ez_f1": float(patients["true_k_patient_ez_f1"].mean()),
        "true_k_patient_nez_f1": float(patients["true_k_patient_nez_f1"].mean()),
        "patient_oracle_threshold_macro_f1": float(patients["oracle_threshold_patient_macro_f1"].mean()),
        "patient_oracle_threshold_ez_f1": float(patients["oracle_threshold_patient_ez_f1"].mean()),
        "patient_oracle_threshold_nez_f1": float(patients["oracle_threshold_patient_nez_f1"].mean()),
        "patient_oracle_prefix_macro_f1": float(patients["oracle_prefix_patient_macro_f1"].mean()),
        "oracle_gain_over_strict": float(patients["oracle_gain_over_strict"].mean()),
        "n_patients": int(table["subject_id"].nunique()),
        "n_channels": int(len(table)),
        "ranking_status": "AVAILABLE" if ranking else "UNAVAILABLE_HARD_PREDICTIONS_ONLY",
    }
    return summary, patients


def patient_percentile_scores(table: pd.DataFrame) -> pd.Series:
    return table.groupby("subject_id", sort=False)["score_ez"].rank(method="average", pct=True, ascending=True)


@dataclass(frozen=True)
class _PreparedScoreBase:
    subjects: np.ndarray
    subject_codes: np.ndarray
    group_indices: tuple[np.ndarray, ...]
    channels: np.ndarray
    labels_nez: np.ndarray


def _prepare_score_base(base: pd.DataFrame) -> _PreparedScoreBase:
    subjects, codes = np.unique(base["subject_id"].astype(str).to_numpy(), return_inverse=True)
    return _PreparedScoreBase(
        subjects=subjects,
        subject_codes=codes,
        group_indices=tuple(np.flatnonzero(codes == code) for code in range(len(subjects))),
        channels=base["channel_name"].astype(str).to_numpy(),
        labels_nez=base["label_nez"].to_numpy(dtype=int),
    )


def _fast_best_index(macro: np.ndarray, ez: np.ndarray, allowed: np.ndarray) -> int:
    macro_values = macro[allowed]
    candidates = allowed[macro_values == macro_values.max()]
    ez_values = ez[candidates]
    candidates = candidates[ez_values == ez_values.max()]
    return int(candidates.min())


def _score_vector_metrics(
    base: pd.DataFrame,
    scores: np.ndarray,
    *,
    details: bool = False,
    prepared: _PreparedScoreBase | None = None,
) -> dict[str, Any]:
    prepared = prepared or _prepare_score_base(base)
    score_values = np.asarray(scores, dtype=float)
    oracle_values: list[float] = []
    true_k_values: list[float] = []
    patient_details: list[dict[str, Any]] = []
    for code, indices in enumerate(prepared.group_indices):
        local_order = np.lexsort((prepared.channels[indices], -score_values[indices]))
        ordered_indices = indices[local_order]
        y = prepared.labels_nez[ordered_indices]
        ordered_scores = score_values[ordered_indices]
        macro, ez, _, _ = _curve_metrics(y)
        boundaries = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True]) + 1
        allowed = np.r_[0, boundaries]
        oracle_k = _fast_best_index(macro, ez, allowed)
        true_k = int((y == 0).sum())
        oracle_value = float(macro[oracle_k])
        true_k_value = float(macro[true_k])
        oracle_values.append(oracle_value)
        true_k_values.append(true_k_value)
        if details:
            patient_details.append(
                {
                    "subject_id": prepared.subjects[code],
                    "oracle_macro_f1": oracle_value,
                    "true_k_macro_f1": true_k_value,
                }
            )
    global_result = _global_threshold_arrays(prepared, score_values)
    return {
        "patient_oracle_threshold_macro_f1": float(np.mean(oracle_values)),
        "true_k_patient_macro_f1": float(np.mean(true_k_values)),
        "optimistic_in_sample_global_threshold_f1": global_result["macro_f1"],
        "optimistic_in_sample_global_threshold": global_result["threshold"],
        "patient_details": patient_details,
    }


def _global_threshold_arrays(prepared: _PreparedScoreBase, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_y = prepared.labels_nez[order]
    inverse_order = np.empty(len(order), dtype=int)
    inverse_order[order] = np.arange(len(order))
    n_patients = len(prepared.subjects)
    delta_macro = np.zeros(len(order), dtype=float)
    delta_ez = np.zeros(len(order), dtype=float)
    baseline_macro = 0.0
    baseline_ez = 0.0
    for indices in prepared.group_indices:
        positions = np.sort(inverse_order[indices])
        y = ordered_y[positions]
        macro, ez, _, _ = _curve_metrics(y)
        baseline_macro += float(macro[0])
        baseline_ez += float(ez[0])
        delta_macro[positions] = np.diff(macro)
        delta_ez[positions] = np.diff(ez)
    cumulative_macro = baseline_macro + np.cumsum(delta_macro)
    cumulative_ez = baseline_ez + np.cumsum(delta_ez)
    boundaries = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    positions = np.r_[-1, boundaries]
    macro_values = np.r_[baseline_macro / n_patients, cumulative_macro[boundaries] / n_patients]
    ez_values = np.r_[baseline_ez / n_patients, cumulative_ez[boundaries] / n_patients]
    candidates = np.flatnonzero(macro_values == macro_values.max())
    candidates = candidates[ez_values[candidates] == ez_values[candidates].max()]
    chosen = int(candidates[np.argmin(positions[candidates] + 1)])
    index = int(positions[chosen])
    threshold = float("inf") if index < 0 else float(ordered_scores[index])
    return {
        "macro_f1": float(macro_values[chosen]),
        "ez_f1": float(ez_values[chosen]),
        "threshold": threshold,
        "predicted_ez_count": index + 1,
    }


def _global_threshold_score(table: pd.DataFrame) -> dict[str, float]:
    """Optimize one full-OOF threshold efficiently; this is diagnostic only."""
    return _global_threshold_arrays(_prepare_score_base(table), table["score_ez"].to_numpy(dtype=float))


def _model_score_matrix(candidates: dict[str, pd.DataFrame], canonical: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    key_columns = ["subject_id", "channel_name"]
    matrix = canonical[key_columns + ["label_nez", "center", "outer_fold"]].copy()
    models: list[str] = []
    for model in sorted({str(table["model"].iloc[0]) for table in candidates.values()}):
        seeds = [table for table in candidates.values() if str(table["model"].iloc[0]) == model]
        seed_scores: list[np.ndarray] = []
        for table in seeds:
            aligned = canonical[key_columns].merge(
                table[key_columns + ["score_ez"]], on=key_columns, how="left", validate="one_to_one"
            )
            ranked = aligned.assign(subject_id=canonical["subject_id"].to_numpy()).groupby("subject_id", sort=False)["score_ez"].rank(
                method="average", pct=True, ascending=True
            )
            seed_scores.append(ranked.to_numpy(dtype=float))
        matrix[model] = np.mean(seed_scores, axis=0)
        models.append(model)
    return matrix, models


def library_oracles(
    patient_metrics: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    valid = patient_metrics.dropna(subset=["oracle_threshold_patient_macro_f1"]).copy()
    best_rows = (
        valid.sort_values(
            ["subject_id", "oracle_threshold_patient_macro_f1", "candidate_id"],
            ascending=[True, False, True], kind="stable",
        )
        .groupby("subject_id", as_index=False, sort=True)
        .first()
    )
    model_seed_oracle = float(best_rows["oracle_threshold_patient_macro_f1"].mean())

    model_matrix, models = _model_score_matrix(candidates, canonical)
    seed_ensemble_patient_rows: list[dict[str, Any]] = []
    for model in models:
        temp = canonical[["subject_id", "channel_name", "label_nez", "center", "outer_fold"]].copy()
        temp["score_ez"] = model_matrix[model]
        for subject, group in temp.groupby("subject_id", sort=True):
            result = patient_oracle(group)
            seed_ensemble_patient_rows.append({"subject_id": subject, "model": model, "oracle_f1": result["macro_f1"]})
    seed_patient = pd.DataFrame(seed_ensemble_patient_rows)
    seed_best = (
        seed_patient.sort_values(["subject_id", "oracle_f1", "model"], ascending=[True, False, True], kind="stable")
        .groupby("subject_id", as_index=False)
        .first()
    )
    seed_model_oracle = float(seed_best["oracle_f1"].mean())
    best_single = float(
        patient_metrics.groupby("candidate_id")["oracle_threshold_patient_macro_f1"].mean().max()
    )
    summary = {
        "oracle_model_seed_selection_macro_f1": model_seed_oracle,
        "oracle_model_selection_after_seed_ensemble_macro_f1": seed_model_oracle,
        "best_single_model_oracle": best_single,
        "library_oracle_gain": model_seed_oracle - best_single,
    }
    selections = best_rows[
        ["subject_id", "candidate_id", "model", "seed", "oracle_threshold_patient_macro_f1"]
    ].rename(
        columns={
            "candidate_id": "best_candidate_id",
            "model": "best_model",
            "seed": "best_seed",
            "oracle_threshold_patient_macro_f1": "best_patient_oracle_f1",
        }
    )
    return selections, summary, seed_best


def _weight_candidates(n_models: int, random_candidates: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    for index in range(n_models):
        weights = np.zeros(n_models)
        weights[index] = 1.0
        rows.append((f"one_hot_{index}", weights))
    rows.append(("equal_weight", np.full(n_models, 1.0 / n_models)))
    for left in range(n_models):
        for right in range(left + 1, n_models):
            for step in range(1, 20):
                weights = np.zeros(n_models)
                weights[left] = step / 20.0
                weights[right] = 1.0 - weights[left]
                rows.append((f"pair_{left}_{right}_{step:02d}", weights))
    rng = np.random.default_rng(int(seed))
    for index, weights in enumerate(rng.dirichlet(np.ones(n_models), size=int(random_candidates))):
        rows.append((f"dirichlet_{index:05d}", weights))
    deduplicated: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, weights in rows:
        key = tuple(np.round(weights, 12))
        if key not in seen:
            seen.add(key)
            deduplicated.append((name, weights))
    return deduplicated


def ensemble_search(
    candidates: dict[str, pd.DataFrame],
    canonical: pd.DataFrame,
    *,
    random_candidates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    matrix, models = _model_score_matrix(candidates, canonical)
    if not models:
        raise UpperBoundAuditError("No aligned continuous-score models are available for ensemble search.")
    values = matrix[models].to_numpy(dtype=float)
    prepared = _prepare_score_base(canonical)
    rows: list[dict[str, Any]] = []
    weights_by_name: dict[str, np.ndarray] = {}
    for name, weights in _weight_candidates(len(models), random_candidates, random_seed):
        metrics = _score_vector_metrics(canonical, values @ weights, prepared=prepared)
        rows.append({"candidate": name, **metrics, **{f"weight_{model}": float(weight) for model, weight in zip(models, weights)}})
        weights_by_name[name] = weights
    metrics_frame = pd.DataFrame(rows).drop(columns=["patient_details"])
    best_row = metrics_frame.sort_values(
        ["patient_oracle_threshold_macro_f1", "true_k_patient_macro_f1", "candidate"],
        ascending=[False, False, True], kind="stable",
    ).iloc[0]
    best_true_k_row = metrics_frame.sort_values(
        ["true_k_patient_macro_f1", "patient_oracle_threshold_macro_f1", "candidate"],
        ascending=[False, False, True], kind="stable",
    ).iloc[0]
    best_global_row = metrics_frame.sort_values(
        ["optimistic_in_sample_global_threshold_f1", "candidate"],
        ascending=[False, True], kind="stable",
    ).iloc[0]
    best_name = str(best_row["candidate"])
    best_weights = weights_by_name[best_name]
    best_scores = values @ best_weights
    channel_scores = canonical[["subject_id", "center", "outer_fold", "channel_name", "label_nez"]].copy()
    channel_scores["ensemble_score_ez"] = best_scores
    best_details = _score_vector_metrics(canonical, best_scores, details=True, prepared=prepared)
    patient_details = pd.DataFrame(best_details["patient_details"])
    equal_row = metrics_frame.loc[metrics_frame["candidate"] == "equal_weight"].iloc[0]
    best_single = float(metrics_frame.loc[metrics_frame["candidate"].str.startswith("one_hot_"), "patient_oracle_threshold_macro_f1"].max())
    summary = {
        "models": models,
        "best_candidate": best_name,
        "best_oracle_candidate": best_name,
        "best_true_k_candidate": str(best_true_k_row["candidate"]),
        "best_global_threshold_candidate": str(best_global_row["candidate"]),
        "weight_objective": "patient_oracle_threshold_macro_f1",
        "weights": {model: float(weight) for model, weight in zip(models, best_weights)},
        "equal_weight_oracle_f1": float(equal_row["patient_oracle_threshold_macro_f1"]),
        "best_simplex_ensemble_oracle_f1": float(best_row["patient_oracle_threshold_macro_f1"]),
        "best_simplex_ensemble_true_k_f1": float(best_true_k_row["true_k_patient_macro_f1"]),
        "optimistic_in_sample_global_threshold_f1": float(best_global_row["optimistic_in_sample_global_threshold_f1"]),
        "optimistic_in_sample_global_threshold": float(best_global_row["optimistic_in_sample_global_threshold"]),
        "ensemble_gain_over_best_single": float(best_row["patient_oracle_threshold_macro_f1"] - best_single),
    }
    return metrics_frame, summary, channel_scores, patient_details


def patient_bootstrap(
    vectors: dict[str, pd.Series],
    *,
    samples: int,
    seed: int,
    gap_pairs: Sequence[tuple[str, str, str]] = (),
) -> pd.DataFrame:
    """Bootstrap aligned patient metric vectors; never resample channels."""
    if not vectors:
        return pd.DataFrame()
    common = sorted(set.intersection(*(set(series.index.astype(str)) for series in vectors.values())))
    if not common:
        raise UpperBoundAuditError("Bootstrap vectors have no common patients.")
    arrays = {name: series.rename(index=str).loc[common].to_numpy(dtype=float) for name, series in vectors.items()}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(common), size=(int(samples), len(common)))
    rows: list[dict[str, Any]] = []
    for name, values in arrays.items():
        estimates = values[draws].mean(axis=1)
        lower, upper = np.percentile(estimates, [2.5, 97.5])
        rows.append({"metric": name, "estimate": float(values.mean()), "ci_lower": float(lower), "ci_upper": float(upper), "bootstrap_samples": int(samples)})
    for name, high, low in gap_pairs:
        differences = arrays[high] - arrays[low]
        estimates = differences[draws].mean(axis=1)
        lower, upper = np.percentile(estimates, [2.5, 97.5])
        rows.append({"metric": name, "estimate": float(differences.mean()), "ci_lower": float(lower), "ci_upper": float(upper), "bootstrap_samples": int(samples)})
    return pd.DataFrame(rows)


def _aggregate_slices(patient_metrics: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    metrics = [
        "strict_patient_macro_f1", "true_k_patient_macro_f1",
        "oracle_threshold_patient_macro_f1", "oracle_prefix_patient_macro_f1",
    ]
    grouped = patient_metrics.groupby(by, dropna=False, as_index=False)[metrics].mean()
    counts = patient_metrics.groupby(by, dropna=False).agg(n_patients=("subject_id", "nunique"), n_channels=("n_channels", "sum")).reset_index()
    return grouped.merge(counts, on=by, how="left")


def _model_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in by_seed.select_dtypes(include=[np.number]).columns if column != "seed"]
    rows: list[dict[str, Any]] = []
    for model, group in by_seed.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model, "n_seeds": int(group["seed"].nunique())}
        for column in numeric:
            values = group[column].astype(float)
            row.update(
                {
                    f"{column}_mean": float(values.mean()),
                    f"{column}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    f"{column}_min": float(values.min()),
                    f"{column}_max": float(values.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_clean_template(canonical: pd.DataFrame, path: Path) -> None:
    template = canonical[["subject_id", "center", "outer_fold", "channel_name", "label_nez"]].copy()
    template = template.rename(columns={"label_nez": "observed_label_nez"})
    template["observed_label_ez"] = 1 - template["observed_label_nez"]
    template["reviewer1_label"] = ""
    template["reviewer2_label"] = ""
    template["adjudicated_label_nez"] = ""
    template["confidence"] = ""
    template["include_clean_subset"] = ""
    template["label_source"] = ""
    template["notes"] = ""
    template.to_csv(path, index=False)


def _clean_label_metrics(
    clean_path: Path | None,
    candidates: dict[str, pd.DataFrame],
) -> tuple[str, pd.DataFrame]:
    if clean_path is None:
        return "NOT_AVAILABLE", pd.DataFrame()
    clean = pd.read_csv(clean_path)
    required = {"subject_id", "channel_name", "adjudicated_label_nez", "include_clean_subset"}
    if not required.issubset(clean):
        raise UpperBoundAuditError(f"Clean-label CSV missing columns: {sorted(required - set(clean))}")
    clean["subject_id"] = clean["subject_id"].astype(str).str.strip()
    clean["channel_name"] = clean["channel_name"].map(normalize_channel_name)
    include = clean["include_clean_subset"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    labels = pd.to_numeric(clean["adjudicated_label_nez"], errors="coerce")
    valid = clean[include & labels.isin([0, 1])][["subject_id", "channel_name"]].copy()
    valid["label_nez"] = labels[include & labels.isin([0, 1])].astype(int)
    if valid.empty:
        return "AVAILABLE_NO_VALID_ADJUDICATED_CHANNELS", pd.DataFrame()
    rows = []
    for candidate_id, table in candidates.items():
        subset = table.drop(columns=["label_nez"]).merge(valid, on=["subject_id", "channel_name"], how="inner", validate="one_to_one")
        if subset.empty:
            continue
        summary, _ = evaluate_candidate(subset)
        rows.append(
            {
                "candidate_id": candidate_id,
                "clean_strict_patient_macro_f1": summary["strict_count_free_patient_macro_f1"],
                "clean_true_k_patient_macro_f1": summary["true_k_patient_macro_f1"],
                "clean_oracle_threshold_patient_macro_f1": summary["patient_oracle_threshold_macro_f1"],
                "n_clean_patients": int(subset["subject_id"].nunique()),
                "n_clean_channels": int(len(subset)),
            }
        )
    return "AVAILABLE", pd.DataFrame(rows)


def _git_metadata(repository: Path) -> dict[str, str]:
    def command(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    return {"branch": command("branch", "--show-current"), "commit": command("rev-parse", "HEAD")}


@dataclass
class UpperBoundAuditOutputs:
    output_dir: Path
    by_model_seed: pd.DataFrame
    library_summary: dict[str, Any]
    ensemble_summary: dict[str, Any]
    audit: dict[str, Any]


def run_upper_bound_audit(
    *,
    task1_output_dir: str | Path,
    v3_ledger: str | Path,
    output_dir: str | Path,
    models: Sequence[str] | None = None,
    extra_ledgers: Sequence[str | Path] = (),
    extra_ledger_globs: Sequence[str] = (),
    bootstrap_samples: int = 2000,
    ensemble_random_candidates: int = 5000,
    random_seed: int = 42,
    clean_labels_csv: str | Path | None = None,
    strict: bool = False,
    repository: str | Path | None = None,
) -> UpperBoundAuditOutputs:
    output = Path(output_dir)
    directories = {name: output / name for name in ("audit", "manifests", "ledgers", "metrics", "comparison", "reports")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    v3_path = Path(v3_ledger).resolve()
    v3_raw = pd.read_csv(v3_path)
    fold_manifest = freeze_v3_fold_manifest(v3_raw)
    v3_standard, v3_read_audit = standardize_ledger(v3_raw, source_path=v3_path, default_model="v3", default_seed=0)
    canonical = v3_standard[["subject_id", "center", "outer_fold", "channel_name", "label_nez"]].copy()
    if canonical.duplicated(["subject_id", "channel_name"]).any():
        raise UpperBoundAuditError("V3 canonical ledger has duplicate normalized patient-channel keys.")
    canonical = canonical.sort_values(["subject_id", "channel_name"], kind="stable").reset_index(drop=True)
    canonical.to_csv(directories["manifests"] / "canonical_channel_manifest.csv", index=False)

    task1_root = Path(task1_output_dir)
    discovered = sorted((task1_root / "oof_ledgers").glob("*/seed_*_channel_oof.csv"))
    paths: list[Path] = [v3_path, *discovered, *(Path(path) for path in extra_ledgers)]
    for pattern in extra_ledger_globs:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern, recursive=True)))
    unique_paths: list[Path] = []
    seen_paths: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique_paths.append(path.resolve())

    inventory_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    candidates: dict[str, pd.DataFrame] = {}
    protocol_reference: dict[str, str] = {}
    read_audits: list[dict[str, Any]] = []
    requested = {value.lower() for value in models} if models else None
    for path in unique_paths:
        try:
            standardized, read_audit = standardize_ledger(
                v3_raw if path == v3_path else path,
                source_path=path,
                default_model="v3" if path == v3_path else None,
                default_seed=0 if path == v3_path else None,
            )
            read_audits.append(read_audit)
        except Exception as exc:
            rejected_rows.append({"source_path": str(path), "reason": "read_or_schema_error", "detail": str(exc)})
            if strict and path in [Path(item).resolve() for item in extra_ledgers]:
                raise
            continue
        for (model, seed), group in standardized.groupby(["model", "seed"], sort=True):
            model_name = str(model)
            candidate_id = f"{model_name}::seed_{int(seed)}"
            if requested is not None and model_name.lower() not in requested:
                inventory_rows.append({"source_path": str(path), "candidate_id": candidate_id, "status": "filtered_by_models"})
                continue
            try:
                validation = validate_candidate(group.reset_index(drop=True), fold_manifest, canonical, strict=strict)
            except Exception as exc:
                rejected_rows.append({"source_path": str(path), "candidate_id": candidate_id, "reason": "validation_error", "detail": str(exc)})
                if strict and requested is not None and model_name.lower() in requested:
                    raise
                continue
            if candidate_id in candidates:
                old = candidates[candidate_id].sort_values(["subject_id", "channel_name"]).reset_index(drop=True)
                new = group.sort_values(["subject_id", "channel_name"]).reset_index(drop=True)
                compare = ["subject_id", "channel_name", "label_nez", "score_nez", "score_ez", "predicted_nez"]
                identical = old[compare].equals(new[compare])
                reason = "duplicate_candidate_identical" if identical else "duplicate_candidate_conflict"
                rejected_rows.append({"source_path": str(path), "candidate_id": candidate_id, "reason": reason, "detail": "candidate_id already loaded"})
                if not identical:
                    raise UpperBoundAuditError(f"Conflicting sources define {candidate_id}.")
                continue
            protocol_conflicts = []
            for field, values in validation["protocol_fields"].items():
                if not values:
                    continue
                value = values[0]
                if field in protocol_reference and protocol_reference[field] != value:
                    protocol_conflicts.append(f"{field}: {value} != {protocol_reference[field]}")
                else:
                    protocol_reference.setdefault(field, value)
            if protocol_conflicts:
                detail = "; ".join(protocol_conflicts)
                rejected_rows.append({"source_path": str(path), "candidate_id": candidate_id, "reason": "protocol_metadata_conflict", "detail": detail})
                if strict and requested is not None and model_name.lower() in requested:
                    raise UpperBoundAuditError(f"{candidate_id} protocol metadata conflicts: {detail}")
                continue
            candidates[candidate_id] = group.reset_index(drop=True)
            inventory_rows.append({"source_path": str(path), "candidate_id": candidate_id, "status": "accepted", **validation})
            alignment_rows.append({key: value for key, value in validation.items() if key not in {"warnings", "protocol_fields"}})

    if requested is not None:
        loaded_models = {str(table["model"].iloc[0]).lower() for table in candidates.values()}
        missing_requested = sorted(requested - loaded_models)
        if missing_requested and strict:
            raise UpperBoundAuditError(f"Explicitly requested models were not accepted: {missing_requested}")
    if not candidates:
        raise UpperBoundAuditError("No legal channel OOF ledgers were accepted; training was not started.")

    inventory = pd.DataFrame(inventory_rows)
    alignment = pd.DataFrame(alignment_rows)
    rejected = pd.DataFrame(rejected_rows, columns=["source_path", "candidate_id", "reason", "detail"])
    inventory.to_csv(directories["audit"] / "ledger_inventory.csv", index=False)
    alignment.to_csv(directories["audit"] / "ledger_alignment.csv", index=False)
    rejected.to_csv(directories["audit"] / "rejected_ledgers.csv", index=False)

    accepted_parts = []
    summary_rows = []
    patient_parts = []
    for candidate_id, table in candidates.items():
        summary, patients = evaluate_candidate(table)
        summary_rows.append(summary)
        patient_parts.append(patients)
        accepted_parts.append(table.assign(candidate_id=candidate_id))
    by_seed = pd.DataFrame(summary_rows).sort_values(["model", "seed"]).reset_index(drop=True)
    patient_metrics = pd.concat(patient_parts, ignore_index=True)
    score_matrix_long = pd.concat(accepted_parts, ignore_index=True)
    score_matrix_long.to_csv(directories["ledgers"] / "oof_score_matrix.csv.gz", index=False, compression="gzip")
    by_seed.to_csv(directories["metrics"] / "upper_bound_by_model_seed.csv", index=False)
    model_summary = _model_summary(by_seed)
    model_summary.to_csv(directories["metrics"] / "upper_bound_by_model.csv", index=False)
    by_center = _aggregate_slices(patient_metrics, ["candidate_id", "model", "seed", "center"])
    by_fold = _aggregate_slices(patient_metrics, ["candidate_id", "model", "seed", "outer_fold"])
    by_center.to_csv(directories["metrics"] / "upper_bound_by_center.csv", index=False)
    by_fold.to_csv(directories["metrics"] / "upper_bound_by_fold.csv", index=False)
    patient_metrics.to_csv(directories["metrics"] / "upper_bound_by_patient.csv", index=False)

    aligned_candidates = {
        candidate_id: table
        for candidate_id, table in candidates.items()
        if bool(alignment.loc[alignment["candidate_id"] == candidate_id, "aligned_for_library"].iloc[0])
        and table["score_kind"].iloc[0] != "hard_only"
    }
    selections, library_summary, seed_ensemble_best = library_oracles(
        patient_metrics[patient_metrics["candidate_id"].isin(aligned_candidates)], aligned_candidates, canonical
    )
    pd.DataFrame([library_summary]).to_csv(directories["metrics"] / "library_oracle_summary.csv", index=False)
    selections.to_csv(directories["comparison"] / "library_oracle_patient_selections.csv", index=False)
    seed_ensemble_best.to_csv(directories["comparison"] / "seed_ensemble_library_oracle_patient_selections.csv", index=False)

    ensemble_metrics, ensemble_summary, ensemble_channels, ensemble_patients = ensemble_search(
        aligned_candidates, canonical,
        random_candidates=ensemble_random_candidates, random_seed=random_seed,
    )
    ensemble_metrics.to_csv(directories["comparison"] / "ensemble_candidate_metrics.csv", index=False)
    (directories["comparison"] / "ensemble_best_weights.json").write_text(
        json.dumps(ensemble_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    ensemble_channels.to_csv(directories["ledgers"] / "ensemble_best_channel_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame([ensemble_summary]).to_csv(directories["metrics"] / "ensemble_upper_bound_summary.csv", index=False)

    best_oracle_id = str(by_seed.sort_values("patient_oracle_threshold_macro_f1", ascending=False).iloc[0]["candidate_id"])
    primary = patient_metrics[patient_metrics["candidate_id"] == best_oracle_id].set_index("subject_id")
    library_vector = selections.set_index("subject_id")["best_patient_oracle_f1"]
    seed_library_vector = seed_ensemble_best.set_index("subject_id")["oracle_f1"]
    ensemble_vector = ensemble_patients.set_index("subject_id")["oracle_macro_f1"]
    vectors = {
        "strict_count_free_patient_macro_f1": primary["strict_patient_macro_f1"],
        "true_k_patient_macro_f1": primary["true_k_patient_macro_f1"],
        "oracle_threshold_patient_macro_f1": primary["oracle_threshold_patient_macro_f1"],
        "best_ensemble_oracle_f1": ensemble_vector,
        "library_oracle_f1": library_vector,
        "seed_ensemble_library_oracle_f1": seed_library_vector,
    }
    bootstrap = patient_bootstrap(
        vectors, samples=bootstrap_samples, seed=random_seed,
        gap_pairs=(
            ("oracle_gain_over_strict", "oracle_threshold_patient_macro_f1", "strict_count_free_patient_macro_f1"),
            ("true_k_gain_over_strict", "true_k_patient_macro_f1", "strict_count_free_patient_macro_f1"),
        ),
    )
    bootstrap.to_csv(directories["comparison"] / "upper_bound_bootstrap.csv", index=False)

    clean_template = directories["reports"] / "task1_label_adjudication_template.csv"
    _write_clean_template(canonical, clean_template)
    clean_status, clean_metrics = _clean_label_metrics(Path(clean_labels_csv) if clean_labels_csv else None, candidates)
    if not clean_metrics.empty:
        clean_metrics.to_csv(directories["metrics"] / "clean_label_upper_bound.csv", index=False)

    repository_path = Path(repository or Path(__file__).resolve().parents[1])
    git = _git_metadata(repository_path)
    audit = {
        "branch": git["branch"],
        "git_commit": git["commit"],
        "old90_subject_count": int(len(fold_manifest)),
        "folds": sorted(int(value) for value in fold_manifest["outer_fold"].unique()),
        "centers": sorted(str(value) for value in fold_manifest["center"].unique()),
        "label_direction": {"NEZ": 1, "EZ": 0},
        "fold_manifest_hash": fold_manifest_hash(fold_manifest),
        "canonical_channel_count": int(len(canonical)),
        "canonical_source": str(v3_path),
        "input_paths": [str(path) for path in unique_paths],
        "accepted_candidates": sorted(candidates),
        "rejected_count": int(len(rejected)),
        "clean_label_status": clean_status,
        "read_audits": read_audits,
        "bootstrap_samples": int(bootstrap_samples),
        "ensemble_random_candidates": int(ensemble_random_candidates),
        "random_seed": int(random_seed),
    }
    (directories["audit"] / "upper_bound_input_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    best_strict = float(by_seed["strict_count_free_patient_macro_f1"].max())
    best_true_k = float(by_seed["true_k_patient_macro_f1"].max())
    best_oracle = float(by_seed["patient_oracle_threshold_macro_f1"].max())
    best_prefix = float(by_seed["patient_oracle_prefix_macro_f1"].max())
    conclusions = []
    if best_oracle < 0.70:
        conclusions.append("当前任何单模型排序都不支持 0.70；继续调 threshold、EMA 或 prevalence head 不足以解决问题。")
    elif best_oracle >= 0.75 and best_strict < 0.65:
        conclusions.append("排序信息可能足够，主要瓶颈是患者级决策边界、prevalence 或 calibration。")
    if library_summary["library_oracle_gain"] >= 0.05:
        conclusions.append("模型在患者层面存在明显互补，值得做 cross-fitted model selector 或 mixture-of-experts。")
    else:
        conclusions.append("模型库 oracle 增益低于 0.05，现有模型错误高度相关。")
    if ensemble_summary["best_simplex_ensemble_oracle_f1"] < 0.70:
        conclusions.append("现有模型家族缺乏可组合到 0.70 的通道级信息，应增加新生理特征或修复标签。")
    elif ensemble_summary["best_simplex_ensemble_oracle_f1"] >= 0.75:
        conclusions.append("互补信息存在；但该 full-OOF 权重与 oracle threshold 仍不可部署，必须另做 cross-fitting。")
    if clean_status != "AVAILABLE":
        conclusions.append("无法判断临床真实标签上限，只能判断 observed-label ceiling。")

    main_columns = [
        "model", "seed", "strict_count_free_patient_macro_f1", "true_k_patient_macro_f1",
        "patient_oracle_threshold_macro_f1", "patient_oracle_prefix_macro_f1", "oracle_gain_over_strict",
        "n_patients", "n_channels",
    ]
    report_model_columns = ["model", "n_seeds"]
    for metric in (
        "strict_count_free_patient_macro_f1", "true_k_patient_macro_f1",
        "patient_oracle_threshold_macro_f1", "patient_oracle_prefix_macro_f1",
    ):
        report_model_columns.extend([f"{metric}_mean", f"{metric}_std"])
    report = [
        "# Task 1 Upper-Bound Audit",
        "",
        "## Data and protocol",
        "",
        f"- Branch/commit: `{git['branch']}` / `{git['commit']}`",
        f"- Frozen cohort: {len(fold_manifest)} patients, folds {audit['folds']}, centers {audit['centers']}",
        f"- Canonical channels: {len(canonical)} from `{v3_path}`",
        "- Label direction: NEZ=1, EZ=0",
        f"- Clean-label status: **{clean_status}**",
        f"- Accepted candidates: {', '.join(sorted(candidates))}",
        f"- Rejected/duplicate inputs: {len(rejected)} (see `audit/rejected_ledgers.csv`)",
        "",
        "## Main table",
        "",
        frame_to_markdown(by_seed[main_columns]),
        "",
        "True-K, patient oracle threshold, oracle prefix, library oracle, and the full-OOF ensemble are **DIAGNOSTIC / ORACLE / NOT DEPLOYABLE**.",
        "Oracle prefix is **ULTRA-OPTIMISTIC, TIE-SPLIT ALLOWED**. The primary ranking-ceiling judgment uses oracle threshold, not oracle prefix.",
        "",
        "## Model mean ± spread",
        "",
        frame_to_markdown(model_summary[report_model_columns]),
        "",
        "## By center",
        "",
        frame_to_markdown(by_center),
        "",
        "## By outer fold",
        "",
        frame_to_markdown(by_fold),
        "",
        "## Model-library oracle",
        "",
        frame_to_markdown(pd.DataFrame([library_summary])),
        "",
        "## Constrained global ensemble upper bound",
        "",
        frame_to_markdown(pd.DataFrame([ensemble_summary]).drop(columns=["weights", "models"])),
        "",
        f"Best weights (full-OOF optimized; diagnostic only): `{json.dumps(ensemble_summary['weights'], sort_keys=True)}`",
        "",
        "## Patient bootstrap (95% percentile CI)",
        "",
        frame_to_markdown(bootstrap),
        "",
        "## Interpretation",
        "",
        *[f"- {text}" for text in conclusions],
        "",
        "## Deployability boundary",
        "",
        "**DEPLOYABLE / STRICT:** stored outer-test OOF predictions only.",
        "",
        "**DIAGNOSTIC ONLY:** true-K, patient oracle threshold, oracle prefix, patient-wise model-library oracle, and full-OOF optimized ensemble/threshold.",
        "",
        f"Best strict count-free patient-macro F1: **{best_strict:.6f}**",
        f"Best true-K F1: **{best_true_k:.6f}**",
        f"Best single-model oracle-threshold F1: **{best_oracle:.6f}**",
        f"Best oracle-prefix F1: **{best_prefix:.6f}**",
    ]
    (directories["reports"] / "TASK1_UPPER_BOUND_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    return UpperBoundAuditOutputs(output, by_seed, library_summary, ensemble_summary, audit)


__all__ = [
    "UpperBoundAuditError", "UpperBoundAuditOutputs", "ensemble_search", "evaluate_candidate",
    "normalize_channel_name", "patient_bootstrap", "patient_oracle", "patient_percentile_scores",
    "run_upper_bound_audit", "standardize_ledger", "true_k_result", "validate_candidate",
]
