#!/usr/bin/env python3
"""Run the pre-registered D-MIL five-fold TRAIN-only experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dmil_core import (
    FEATURES,
    VARIANTS,
    RobustScaler,
    binary_metrics,
    choose_threshold,
    distribution_features,
    fit_adapter,
    logit,
    ranking_metrics,
    sigmoid,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def center_name(patient: str) -> str:
    lowered = patient.casefold()
    if lowered.startswith("sub-hup"):
        return "HUP"
    if lowered.startswith("sub-openieeg"):
        return "Open-iEEG"
    if lowered.startswith("sub-sourcesink"):
        return "SourceSink"
    if lowered.startswith("sub-zurich"):
        return "Zurich"
    raise RuntimeError(f"Unrecognized source center for {patient}")


def load_cache(cache: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    files = sorted(cache.glob("*.npz"))
    started = time.monotonic()
    for ordinal, path in enumerate(files, 1):
        marker = path.with_suffix(".json")
        if not marker.is_file():
            raise RuntimeError(f"Missing cache marker for {path.name}")
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
        if marker_value.get("output_sha256") != sha256(path):
            raise RuntimeError(f"Cache marker mismatch for {path.name}")
        with np.load(path, allow_pickle=False) as source:
            patient = str(source["patient"])
            edf = str(source["edf"])
            channels = np.asarray(source["channel_names"]).astype(str)
            labels = np.asarray(source["pathological_labels"], dtype=np.int8)
            offsets = np.asarray(source["segment_offsets"], dtype=np.int64)
            flat_logits = np.asarray(source["segment_logits"], dtype=np.float64)
        if len(offsets) != len(channels) + 1 or len(labels) != len(channels):
            raise RuntimeError(f"Malformed cache dimensions in {path.name}")
        for index, (channel, label) in enumerate(zip(channels, labels)):
            logits = flat_logits[offsets[index] : offsets[index + 1]]
            pathological_probability = 1.0 - sigmoid(logits)
            features = distribution_features(pathological_probability)
            identity = hashlib.sha256(f"{patient}|{edf}|{channel}".encode("utf-8")).hexdigest()[:24]
            rows.append(
                {
                    "row_id": identity,
                    "patient": patient,
                    "center": center_name(patient),
                    "edf": edf,
                    "channel": channel,
                    "y": int(label),
                    **features,
                }
            )
        if ordinal % 50 == 0 or ordinal == len(files):
            print(json.dumps({"stage": "load_cache", "ordinal": ordinal, "of": len(files)}), flush=True)
    frame = pd.DataFrame(rows)
    audit = {
        "files": len(files),
        "patients": int(frame.patient.nunique()),
        "all_channel_units": len(frame),
        "labeled_channel_units": int(frame.y.isin([0, 1]).sum()),
        "pathological_units": int((frame.y == 1).sum()),
        "normal_units": int((frame.y == 0).sum()),
        "segments": int(frame.segment_count.sum()),
        "elapsed_seconds": time.monotonic() - started,
        "embeddings_read": False,
        "waveforms_read": False,
        "test_data_read": False,
    }
    return frame, audit


def make_split(labeled: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    stats = labeled.groupby(["patient", "center"], as_index=False).agg(
        labeled_units=("y", "size"), pathological_units=("y", "sum")
    )
    stats["prevalence"] = stats.pathological_units / stats.labeled_units
    positive = stats.prevalence[stats.pathological_units > 0]
    if len(positive):
        q1, q2 = np.quantile(positive, [1 / 3, 2 / 3], method="linear")
    else:
        q1 = q2 = 0.0

    def prevalence_band(row) -> str:
        if row.pathological_units == 0:
            return "zero"
        if row.prevalence <= q1:
            return "low"
        if row.prevalence <= q2:
            return "mid"
        return "high"

    stats["prevalence_band"] = stats.apply(prevalence_band, axis=1)
    stats["stratum"] = stats.center + "|" + stats.prevalence_band
    rare = set(stats.stratum.value_counts()[lambda value: value < 5].index)
    stats.loc[stats.stratum.isin(rare), "stratum"] = stats.center + "|" + np.where(
        stats.pathological_units > 0, "positive", "zero"
    )
    rare = set(stats.stratum.value_counts()[lambda value: value < 5].index)
    # If a center-specific fallback is still too small, merge every stratum of
    # that center.  Mapping only the rare rows would leave a second small label.
    rare_centers = set(stats.loc[stats.stratum.isin(rare), "center"])
    stats.loc[stats.center.isin(rare_centers), "stratum"] = stats.center
    if (stats.stratum.value_counts() < 5).any():
        raise RuntimeError(f"Cannot create five stratified folds: {stats.stratum.value_counts().to_dict()}")
    stats = stats.sort_values("patient").reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    stats["fold"] = -1
    for fold, (_, held) in enumerate(splitter.split(stats.patient, stats.stratum), 1):
        stats.loc[held, "fold"] = fold
    if stats.fold.min() != 1 or stats.fold.max() != 5 or stats.patient.duplicated().any():
        raise RuntimeError("Invalid patient-disjoint split")
    return stats


def score(beta: np.ndarray, scaler: RobustScaler, rows: pd.DataFrame) -> np.ndarray:
    raw = rows.loc[:, FEATURES].to_numpy(dtype=np.float64)
    return sigmoid(logit(rows.mean_score.to_numpy(dtype=np.float64)) + scaler.transform(raw) @ beta)


def metrics_for_rows(rows: pd.DataFrame, threshold: float) -> dict:
    metrics = binary_metrics(rows.y, rows.score, threshold=threshold)
    return {**metrics, **ranking_metrics(rows)}


def aggregate_oof(rows: pd.DataFrame) -> dict:
    metrics = binary_metrics(rows.y, rows.score, predictions=rows.pred)
    return {**metrics, **ranking_metrics(rows)}


def standardized_difference(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or len(right) < 2:
        return None
    pooled = np.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) /
                     (len(left) + len(right) - 2))
    return float((right.mean() - left.mean()) / pooled) if pooled > 0 else None


def feature_label_analysis(labeled: pd.DataFrame, draws: int = 10000) -> pd.DataFrame:
    patients = np.asarray(sorted(labeled.patient.unique()))
    rng = np.random.default_rng(42)
    output: list[dict] = []
    for feature in ("mean_score", *FEATURES):
        grouped = labeled.groupby(["patient", "y"])[feature].mean().unstack()
        normal = grouped.get(0, pd.Series(index=grouped.index, dtype=float)).to_numpy(dtype=float)
        pathological = grouped.get(1, pd.Series(index=grouped.index, dtype=float)).to_numpy(dtype=float)
        group_index = {patient: index for index, patient in enumerate(grouped.index)}
        deltas: list[float] = []
        for _ in range(draws):
            sampled = patients[rng.integers(0, len(patients), size=len(patients))]
            indices = [group_index[patient] for patient in sampled if patient in group_index]
            left = normal[indices]
            right = pathological[indices]
            left = left[np.isfinite(left)]
            right = right[np.isfinite(right)]
            if len(left) and len(right):
                deltas.append(float(right.mean() - left.mean()))
        finite_normal = normal[np.isfinite(normal)]
        finite_pathological = pathological[np.isfinite(pathological)]
        output.append(
            {
                "feature": feature,
                "normal_patient_mean": float(finite_normal.mean()),
                "normal_patient_median": float(np.median(finite_normal)),
                "normal_patients": len(finite_normal),
                "pathological_patient_mean": float(finite_pathological.mean()),
                "pathological_patient_median": float(np.median(finite_pathological)),
                "pathological_patients": len(finite_pathological),
                "patient_equal_mean_difference": float(finite_pathological.mean() - finite_normal.mean()),
                "standardized_effect_size": standardized_difference(finite_normal, finite_pathological),
                "bootstrap_ci_low": float(np.quantile(deltas, 0.025)),
                "bootstrap_ci_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(output)


def conditional_analysis(labeled: pd.DataFrame) -> pd.DataFrame:
    current = labeled.copy()
    current["mean_quintile"] = pd.qcut(
        current.mean_score.rank(method="first"), 5, labels=False
    ).astype(int) + 1
    output: list[dict] = []
    for quintile, group in current.groupby("mean_quintile"):
        for feature in FEATURES:
            patient_class = group.groupby(["patient", "y"])[feature].mean().unstack()
            normal = patient_class.get(0, pd.Series(dtype=float)).dropna().to_numpy(dtype=float)
            pathological = patient_class.get(1, pd.Series(dtype=float)).dropna().to_numpy(dtype=float)
            output.append(
                {
                    "mean_quintile": int(quintile),
                    "mean_score_min": float(group.mean_score.min()),
                    "mean_score_max": float(group.mean_score.max()),
                    "feature": feature,
                    "channel_units": len(group),
                    "normal_patients": len(normal),
                    "pathological_patients": len(pathological),
                    "normal_patient_mean": float(normal.mean()) if len(normal) else None,
                    "pathological_patient_mean": float(pathological.mean()) if len(pathological) else None,
                    "patient_equal_difference": float(pathological.mean() - normal.mean()) if len(normal) and len(pathological) else None,
                    "standardized_effect_size": standardized_difference(normal, pathological),
                }
            )
    return pd.DataFrame(output)


def correlation_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["mean_score", *FEATURES, "segment_count"]
    correlation = frame[columns].corr(method="pearson")
    rows = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "pearson_correlation": value,
                    "absolute_gt_0_95": abs(value) > 0.95,
                    "channel_units": len(frame),
                }
            )
    return pd.DataFrame(rows)


def segment_count_robustness(labeled: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    median = float(labeled.segment_count.median())
    group = np.where(labeled.segment_count < median, "low", np.where(labeled.segment_count > median, "high", "medium"))
    mapping = dict(zip(labeled.row_id, group))
    base = oof[oof.model == "V0_MEAN"].set_index("row_id")
    full = oof[oof.model == "V4_FULL_DMIL"].set_index("row_id")
    rows: list[dict] = []
    for feature in FEATURES:
        rows.append(
            {
                "analysis": "correlation",
                "segment_group": "all",
                "feature": feature,
                "value": float(labeled.segment_count.corr(labeled[feature])),
                "channel_units": len(labeled),
                "baseline_auroc": None,
                "full_auroc": None,
                "delta_auroc": None,
                "segment_count_median": median,
            }
        )
    for name in ("low", "medium", "high"):
        ids = [row_id for row_id, current in mapping.items() if current == name]
        common = base.index.intersection(ids)
        y = base.loc[common, "y"].to_numpy(dtype=np.int8)
        base_score = base.loc[common, "score"].to_numpy(dtype=float)
        full_score = full.loc[common, "score"].to_numpy(dtype=float)
        base_auc = float(roc_auc_score(y, base_score)) if len(np.unique(y)) == 2 else None
        full_auc = float(roc_auc_score(y, full_score)) if len(np.unique(y)) == 2 else None
        rows.append(
            {
                "analysis": "stratified_auroc",
                "segment_group": name,
                "feature": "all",
                "value": None,
                "channel_units": len(common),
                "baseline_auroc": base_auc,
                "full_auroc": full_auc,
                "delta_auroc": None if base_auc is None or full_auc is None else full_auc - base_auc,
                "segment_count_median": median,
            }
        )
    return pd.DataFrame(rows)


def parameter_sign_audit(parameters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        subset = parameters[parameters.model == variant]
        for coefficient in ("beta_T", "beta_Q", "beta_S"):
            values = subset[coefficient].to_numpy(dtype=float)
            nonzero = values[np.abs(values) > 1e-12]
            same_sign = int(max((nonzero > 0).sum(), (nonzero < 0).sum())) if len(nonzero) else 0
            rows.append(
                {
                    "model": variant,
                    "coefficient": coefficient,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "sd": float(values.std(ddof=0)),
                    "positive_folds": int((values > 0).sum()),
                    "negative_folds": int((values < 0).sum()),
                    "zero_folds": int((np.abs(values) <= 1e-12).sum()),
                    "same_sign_nonzero_folds": same_sign,
                    "sign_stable_4_of_5": same_sign >= 4,
                }
            )
    return pd.DataFrame(rows)


def report(summary: pd.DataFrame, gate: dict, sign: pd.DataFrame) -> str:
    lines = [
        "# D-MIL TRAIN OOF report",
        "",
        "| Model | Params | AUROC | AP | Macro-F1 | patient-equal AP | MRR | Top1 | Delta AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    parameter_count = {"V0_MEAN": 0, "V1_TAIL_EXCESS": 1, "V2_Q90_EXCESS": 1, "V3_HETEROGENEITY": 1, "V4_FULL_DMIL": 3}
    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {parameter_count[row.model]} | {row.auroc:.6f} | {row.ap:.6f} | "
            f"{row.macro_f1:.6f} | {row.patient_equal_ap:.6f} | {row.mrr:.6f} | {row.top1:.6f} | {row.delta_auroc_vs_mean:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Terminal: **`{gate['terminal']}`**",
            "",
            f"Full D-MIL delta AUROC: `{gate['delta_auroc']:+.6f}`; delta AP: `{gate['delta_ap']:+.6f}`; "
            f"nonnegative folds: `{gate['nonnegative_folds']}/5`.",
            "",
            "No official TEST data were read by this TRAIN-only run.",
            "",
            "## Parameter signs",
            "",
        ]
    )
    full_sign = sign[(sign.model == "V4_FULL_DMIL")]
    for row in full_sign.itertuples():
        lines.append(
            f"- `{row.coefficient}` median `{row.median:+.6f}`, positive/negative/zero folds "
            f"`{row.positive_folds}/{row.negative_folds}/{row.zero_folds}`."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    if not str(baseline.get("status", "")).startswith("PASS") or abs(
        baseline["observed_auroc"] - lock["expected_test_auroc"]
    ) >= lock["baseline_tolerance"]:
        raise RuntimeError("BASELINE_REPLAY_FAILED")
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)

    all_rows, cache_audit = load_cache(args.train_cache)
    expected = lock["expected_train"]
    if cache_audit["patients"] != expected["patients"] or cache_audit["files"] != expected["edfs"] or cache_audit["labeled_channel_units"] != expected["labeled_edf_channel_pairs"]:
        raise RuntimeError(f"Frozen TRAIN cache mismatch: {cache_audit}")
    labeled = all_rows[all_rows.y.isin([0, 1])].copy().reset_index(drop=True)
    split = make_split(labeled, seed=lock["seed"])
    fold_by_patient = dict(zip(split.patient, split.fold))
    labeled["fold"] = labeled.patient.map(fold_by_patient).astype(int)
    all_rows["fold"] = all_rows.patient.map(fold_by_patient).astype(int)
    atomic_csv(args.private_output / "TRAIN_OOF_SPLIT.csv", split)
    atomic_csv(
        args.output / "TRAIN_OOF_SPLIT.csv",
        split.assign(patient_hash=split.patient.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:16])).drop(columns="patient"),
    )
    atomic_json(args.output / "TRAIN_CACHE_AUDIT.json", cache_audit)

    feature_matrix_all = all_rows.loc[:, FEATURES].to_numpy(dtype=float)
    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    parameter_rows: list[dict] = []
    scalers: dict[str, dict] = {}
    for fold in range(1, 6):
        fit_all = all_rows[all_rows.fold != fold]
        fit = labeled[labeled.fold != fold]
        held = labeled[labeled.fold == fold]
        if set(fit.patient) & set(held.patient):
            raise RuntimeError("Patient leakage")
        scaler = RobustScaler.fit(fit_all.loc[:, FEATURES].to_numpy(dtype=float))
        scalers[str(fold)] = scaler.as_dict()
        fit_z = scaler.transform(fit.loc[:, FEATURES].to_numpy(dtype=float))
        fit_base = logit(fit.mean_score.to_numpy(dtype=float))
        held_z = scaler.transform(held.loc[:, FEATURES].to_numpy(dtype=float))
        held_base = logit(held.mean_score.to_numpy(dtype=float))
        fold_current_metrics: dict[str, dict] = {}
        for model, active in VARIANTS.items():
            started = time.monotonic()
            selected = fit_adapter(
                fit_base,
                fit_z,
                fit.y.to_numpy(dtype=np.int8),
                fit.patient.to_numpy(dtype=str),
                active=active,
                max_epochs=lock["training"]["max_epochs"],
                patience=lock["training"]["patience"],
                learning_rate=lock["training"]["learning_rate"],
            )
            beta = np.asarray(selected["beta"], dtype=float)
            fit_score = sigmoid(fit_base + fit_z @ beta)
            threshold = choose_threshold(fit.y.to_numpy(dtype=np.int8), fit_score)
            held_score = sigmoid(held_base + held_z @ beta)
            current = held[["row_id", "patient", "center", "edf", "channel", "y", "mean_score", *FEATURES, "segment_count", "fold"]].copy()
            current["model"] = model
            current["score"] = held_score
            current["threshold"] = float(threshold["threshold"])
            current["pred"] = (current.score >= current.threshold).astype(np.int8)
            oof_parts.append(current)
            metric = metrics_for_rows(current, float(threshold["threshold"]))
            fold_current_metrics[model] = metric
            fold_metrics.append(
                {
                    "fold": fold,
                    "model": model,
                    "fit_patients": int(fit.patient.nunique()),
                    "held_patients": int(held.patient.nunique()),
                    "held_units": len(held),
                    "threshold_from_fold_train": float(threshold["threshold"]),
                    **metric,
                }
            )
            parameter_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "beta_T": float(beta[0]),
                    "beta_Q": float(beta[1]),
                    "beta_S": float(beta[2]),
                    "selected_epoch": int(selected["selected_epoch"]),
                    "epochs_run": int(selected["epochs_run"]),
                    "fold_train_patient_equal_bce": float(selected["loss"]),
                    "threshold_from_fold_train": float(threshold["threshold"]),
                    "fit_seconds": time.monotonic() - started,
                }
            )
        baseline_auc = fold_current_metrics["V0_MEAN"]["auroc"]
        for row in fold_metrics:
            if row["fold"] == fold:
                row["delta_auroc_vs_mean"] = None if row["auroc"] is None else row["auroc"] - baseline_auc
        print(json.dumps({"stage": "fold_complete", "fold": fold}), flush=True)

    oof = pd.concat(oof_parts, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    fold_frame = pd.DataFrame(fold_metrics)
    summaries: list[dict] = []
    for model in VARIANTS:
        current = oof[oof.model == model]
        metric = aggregate_oof(current)
        summaries.append({"model": model, "channel_units": len(current), "patients": int(current.patient.nunique()), **metric})
    summary = pd.DataFrame(summaries)
    baseline_auc = float(summary.loc[summary.model == "V0_MEAN", "auroc"].iloc[0])
    baseline_ap = float(summary.loc[summary.model == "V0_MEAN", "ap"].iloc[0])
    summary["delta_auroc_vs_mean"] = summary.auroc - baseline_auc
    summary["delta_ap_vs_mean"] = summary.ap - baseline_ap
    full = summary[summary.model == "V4_FULL_DMIL"].iloc[0]
    components = summary[summary.model.isin(["V1_TAIL_EXCESS", "V2_Q90_EXCESS", "V3_HETEROGENEITY"])]
    full_fold = fold_frame[fold_frame.model == "V4_FULL_DMIL"]
    nonnegative_folds = int((full_fold.delta_auroc_vs_mean >= -1e-15).sum())
    best_component_auc = float(components.auroc.max())
    gate_pass = bool(
        full.delta_auroc_vs_mean >= lock["gate"]["minimum_full_delta_auroc"]
        and full.delta_ap_vs_mean >= lock["gate"]["minimum_full_delta_ap"]
        and nonnegative_folds >= lock["gate"]["minimum_nonnegative_folds"]
        and full.auroc >= best_component_auc + lock["gate"]["full_vs_best_component_tolerance"]
    )
    gate = {
        "status": "PASS" if gate_pass else "FAIL",
        "terminal": "DMIL_TRAIN_GATE_PASSED" if gate_pass else "STOP_DMIL_TRAIN_GATE_FAILED",
        "baseline_oof_auroc": baseline_auc,
        "full_oof_auroc": float(full.auroc),
        "delta_auroc": float(full.delta_auroc_vs_mean),
        "baseline_oof_ap": baseline_ap,
        "full_oof_ap": float(full.ap),
        "delta_ap": float(full.delta_ap_vs_mean),
        "nonnegative_folds": nonnegative_folds,
        "best_single_component": str(components.sort_values("auroc", ascending=False).iloc[0].model),
        "best_single_component_auroc": best_component_auc,
        "full_vs_best_component_delta": float(full.auroc - best_component_auc),
        "strong_train_signal": bool(full.delta_auroc_vs_mean >= 0.01),
        "single_component_dominant": bool(full.auroc < best_component_auc - 0.001),
        "official_test_accessed": False,
    }

    sign = parameter_sign_audit(parameters)
    feature_analysis = feature_label_analysis(labeled)
    conditional = conditional_analysis(labeled)
    correlation = correlation_audit(all_rows)
    count_robustness = segment_count_robustness(labeled, oof)
    atomic_csv(args.output / "TRAIN_OOF_FOLD_METRICS.csv", fold_frame)
    private_predictions = oof.sort_values(["model", "fold", "patient", "edf", "channel"])
    atomic_csv(args.private_output / "TRAIN_OOF_PREDICTIONS.csv", private_predictions)
    # Preserve cross-variant pairing without publishing a deterministic hash of
    # the public BIDS patient/EDF/channel tuple (which would be dictionary
    # reversible). The anonymous index has meaning only inside this artifact.
    unit_order = {value: index for index, value in enumerate(private_predictions.row_id.unique())}
    public_predictions = private_predictions.drop(columns=["row_id", "patient", "edf", "channel"]).copy()
    public_predictions.insert(0, "anonymous_unit_index", private_predictions.row_id.map(unit_order).to_numpy())
    atomic_csv(args.output / "TRAIN_OOF_PREDICTIONS.csv", public_predictions)
    atomic_csv(args.output / "TRAIN_OOF_PARAMETERS.csv", parameters)
    atomic_csv(args.output / "TRAIN_OOF_SUMMARY.csv", summary)
    atomic_csv(args.output / "PARAMETER_SIGN_AUDIT.csv", sign)
    atomic_csv(args.output / "FEATURE_LABEL_ANALYSIS.csv", feature_analysis)
    atomic_csv(args.output / "CONDITIONAL_FEATURE_ANALYSIS.csv", conditional)
    atomic_csv(args.output / "FEATURE_CORRELATION_AUDIT.csv", correlation)
    atomic_csv(args.output / "SEGMENT_COUNT_ROBUSTNESS.csv", count_robustness)
    atomic_json(args.output / "TRAIN_OOF_SCALERS.json", scalers)
    atomic_json(args.output / "TRAIN_GATE.json", gate)
    report_text = report(summary, gate, sign)
    (args.output / "TRAIN_OOF_REPORT.md").write_text(report_text, encoding="utf-8")
    (args.output / "FINAL_REPORT.md").write_text(report_text, encoding="utf-8")
    atomic_json(
        args.output / "RUN_STATUS.json",
        {
            "status": gate["terminal"],
            "protocol_sha256": sha256(args.protocol),
            "baseline_audit_sha256": sha256(args.baseline_audit),
            "private_predictions_sha256": sha256(args.private_output / "TRAIN_OOF_PREDICTIONS.csv"),
            "public_predictions_deidentified": True,
            "official_test_accessed": False,
        },
    )
    print(json.dumps(gate, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
