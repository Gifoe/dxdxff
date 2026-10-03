#!/usr/bin/env python3
"""Evaluate only the hash-bound frozen Omni TimeConv-CNN test cache.

This program never imports torch, the CNN, raw EEG, or any training feature.
It turns the already saved segment logits into the exact official
EDF-channel score, validates the historic pooled result, then applies the
predeclared patient/channel mean rule.  Identifying rows are written only to
``--private-output``; all files in ``--output`` are aggregate-safe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
FROZEN_CHANNEL_PREDICTIONS_SHA256 = "d8a500a578c613d177e6513c8692040ab25d5fe0da64fba22631b7b9ae4eaa14"
EXPECTED_CACHE_SHA256 = "3f5cf9a9111fad6c5843eeaba43108057df961dd6a918d83fc40cc34bb5053ad"
EXPECTED_AUROC = 0.7987673466324111
EXPECTED_AP = 0.33030
EXPECTED_MACRO_F1_HALF = 0.6597535251408214
TOLERANCE = 1e-5
THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.npz")):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def center_name(dataset: str) -> str:
    value = str(dataset).casefold().replace("-", "")
    if value == "hup":
        return "HUP"
    if value == "openieeg":
        return "Open-iEEG"
    if value == "sourcesink":
        return "SourceSink"
    if value == "zurich":
        return "Zurich"
    # The split manifest also records Multicenter rows.  They are excluded by
    # the frozen official Task2 cache; retaining a name here lets us build a
    # complete manifest index without silently changing that cache population.
    if value == "multicenter":
        return "Multicenter"
    raise RuntimeError(f"Unrecognized dataset center: {dataset!r}")


def patient_center(split_csv: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with split_csv.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            patient = row["patient_name"]
            center = center_name(row["dataset"])
            previous = result.setdefault(patient, center)
            if previous != center:
                raise RuntimeError(f"Patient has conflicting centers: {patient}")
    return result


def load_frozen_cache(root: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(root.glob("*.npz"))
    if len(files) != 237:
        raise RuntimeError(f"Expected 237 frozen TEST cache files, found {len(files)}")
    rows: list[dict] = []
    total_segments = 0
    for path in files:
        marker = path.with_suffix(".json")
        if not marker.is_file():
            raise RuntimeError(f"Missing frozen cache marker: {marker.name}")
        binding = json.loads(marker.read_text(encoding="utf-8"))
        if binding.get("output_sha256") != sha256(path):
            raise RuntimeError(f"Frozen cache marker hash mismatch: {path.name}")
        if binding.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise RuntimeError(f"Frozen checkpoint mismatch: {path.name}")
        if binding.get("official_source_sha256") != OFFICIAL_SOURCE_SHA256:
            raise RuntimeError(f"Frozen official source mismatch: {path.name}")
        if binding.get("mode") != "test":
            raise RuntimeError(f"Non-test cache contaminates evaluation: {path.name}")
        with np.load(path, allow_pickle=False) as source:
            patient = str(source["patient"])
            edf = str(source["edf"])
            channels = np.asarray(source["channel_names"]).astype(str)
            labels = np.asarray(source["pathological_labels"], dtype=np.int8)
            logits = np.asarray(source["segment_logits"], dtype=np.float64)
            offsets = np.asarray(source["segment_offsets"], dtype=np.int64)
        if len(labels) != len(channels) or len(offsets) != len(channels) + 1:
            raise RuntimeError(f"Malformed EDF/channel schema: {path.name}")
        if offsets[0] != 0 or offsets[-1] != len(logits) or np.any(np.diff(offsets) < 1):
            raise RuntimeError(f"Malformed segment offsets: {path.name}")
        for index, (channel, label) in enumerate(zip(channels, labels)):
            segment_logits = logits[offsets[index] : offsets[index + 1]]
            # The official CNN logit is normal-oriented.  Flip only after the
            # per-segment sigmoid average, exactly as the saved reference did.
            pathological_score = float(1.0 - sigmoid(segment_logits).mean())
            rows.append({"patient": patient, "edf": edf, "channel": str(channel),
                         "y": int(label), "pathological_score": pathological_score,
                         "n_segments": int(len(segment_logits))})
            total_segments += len(segment_logits)
    frame = pd.DataFrame(rows)
    if frame[["patient", "edf", "channel"]].duplicated().any():
        raise RuntimeError("Duplicate frozen EDF-channel identity")
    return frame, {"cache_files": len(files), "cache_channel_units": len(frame),
                   "cache_segments": int(total_segments), "cache_sha256": cache_digest(root)}


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores)) if np.any(labels == 1) else float("nan")


def one_patient_metrics(group: pd.DataFrame, threshold: float) -> dict:
    labels = group.y.to_numpy(dtype=np.int8)
    score = group.pathological_score.to_numpy(dtype=np.float64)
    predicted = (score >= threshold).astype(np.int8)
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    has_both = bool(np.any(labels == 0) and np.any(labels == 1))
    order = np.argsort(-score, kind="stable")
    if np.any(labels == 1):
        positive_ranks = np.flatnonzero(labels[order] == 1)
        reciprocal = float(1.0 / (positive_ranks[0] + 1))
        top1 = float(labels[order[0]] == 1)
        discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
        ideal = float((np.sort(labels)[::-1] * discounts).sum())
        ndcg = float((labels[order] * discounts).sum() / ideal) if ideal else float("nan")
    else:
        reciprocal = top1 = ndcg = float("nan")
    return {
        "n_patient_channels": int(len(group)), "n_pathological": int(np.sum(labels == 1)),
        "n_normal": int(np.sum(labels == 0)), "auroc": safe_auc(labels, score),
        "ap": safe_ap(labels, score),
        "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "pathological_f1": float(f1_score(labels, predicted, labels=[0, 1], pos_label=1, zero_division=0)),
        "normal_f1": float(f1_score(labels, predicted, labels=[0, 1], pos_label=0, zero_division=0)),
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": float((sensitivity + specificity) / 2) if has_both else float("nan"),
        "predicted_pathological_fraction": float(np.mean(predicted)),
        "true_pathological_fraction": float(np.mean(labels)),
        "mrr": reciprocal, "top1": top1, "ndcg": ndcg,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "auroc_estimable": has_both,
        "class_support": "both" if has_both else ("pathological_only" if np.any(labels == 1) else "normal_only"),
    }


def describe(values: pd.Series) -> dict:
    values = values[np.isfinite(values.to_numpy(dtype=float))].astype(float)
    if not len(values):
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan")}
    return {"n": int(len(values)), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()), "q25": float(values.quantile(.25)),
            "q75": float(values.quantile(.75))}


def aggregate_patient_metrics(private: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_scope = {
        "auroc": "both_class_patients_only",
        "macro_f1": "all_patients_fixed_0.5",
        "pathological_f1": "all_patients_fixed_0.5",
        "normal_f1": "all_patients_fixed_0.5",
        "balanced_accuracy": "both_class_patients_only",
        "predicted_pathological_fraction": "all_patients_fixed_0.5",
        "true_pathological_fraction": "all_patients",
        "ap": "patients_with_at_least_one_pathological_channel",
        "mrr": "patients_with_at_least_one_pathological_channel",
        "top1": "patients_with_at_least_one_pathological_channel",
        "ndcg": "patients_with_at_least_one_pathological_channel",
    }
    for metric, scope in metric_scope.items():
        rows.append({"metric": metric, "scope": scope, **describe(private[metric])})
    return pd.DataFrame(rows)


def center_summary(private: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center in ("HUP", "Open-iEEG", "SourceSink", "Zurich"):
        group = private.loc[private.center == center]
        auroc = describe(group.auroc) if len(group) else describe(pd.Series(dtype=float))
        metric = {name: describe(group[name])["mean"] if len(group) else float("nan")
                  for name in ("macro_f1", "pathological_f1", "normal_f1", "balanced_accuracy", "ap", "mrr", "top1", "ndcg")}
        rows.append({"center": center, "n_patients": int(len(group)),
                     "n_auroc_estimable_patients": int(group.auroc_estimable.sum()) if len(group) else 0,
                     "n_single_class_patients": int((~group.auroc_estimable).sum()) if len(group) else 0,
                     "patient_equal_auroc": auroc["mean"], "patient_auroc_sd": auroc["sd"],
                     "patient_auroc_median": auroc["median"], "patient_auroc_q25": auroc["q25"],
                     "patient_auroc_q75": auroc["q75"],
                     "auroc_status": "estimable" if auroc["n"] else "not_estimable", **metric})
    return pd.DataFrame(rows)


def public_report(summary: pd.DataFrame, baseline: dict, collapse: dict, threshold_available: bool,
                  patient_summary: pd.DataFrame, centers: pd.DataFrame) -> str:
    official = summary.loc[summary.evaluation_system == "Official pooled / EDF-channel"].iloc[0]
    a1 = summary.loc[summary.evaluation_system == "A1-style patient-equal / patient-channel @0.5"].iloc[0]
    auroc = patient_summary.loc[patient_summary.metric == "auroc"].iloc[0]
    macro = patient_summary.loc[patient_summary.metric == "macro_f1"].iloc[0]
    delta = float(official.auroc - a1.auroc)
    if a1.auroc >= .75 and a1.macro_f1 < .60:
        case = "Case C"
        explanation = "within-patient ranking is high by the predeclared descriptive cutoff, but fixed-global-threshold Macro-F1 is low"
    elif a1.auroc >= .75:
        case = "Case A"
        explanation = "both pooled and patient-equal AUROC are high by the predeclared descriptive cutoff"
    elif official.auroc >= .75:
        case = "Case B"
        explanation = "pooled AUROC is high while patient-equal AUROC is materially lower under the same frozen scores"
    else:
        case = "Case D"
        explanation = "neither pooled-to-patient conversion nor fixed-threshold patient localization is high by the predeclared descriptive cutoffs"
    center_lines = "\n".join(f"| {r.center} | {int(r.n_patients)} | {int(r.n_auroc_estimable_patients)} | {('not_estimable' if r.auroc_status == 'not_estimable' else f'{r.patient_equal_auroc:.6f}')} |" for _, r in centers.iterrows())
    threshold_sentence = "No. The only recovered full-record TRAIN artifact contains in-sample predictions from this model's fit population, not a patient-held-out validation artifact; selecting a threshold there would be optimistic reuse. No threshold was fabricated." if not threshold_available else "Yes; the frozen validation threshold is reported in the summary table."
    return f"""# Official Omni TimeConv-CNN under an A1-style patient-equal protocol

**Terminal: `{case}` — {explanation}.** This is an additional A1-style patient/channel localization protocol applied to the frozen Omni-iEEG Task2 TEST predictions. It is **not** a replacement official Omni benchmark and, because this TEST cohort has been viewed previously, is an exploratory repeated-test analysis.

| Evaluation system | Unit | AUROC | Macro-F1 | AP | MRR | Top1 | NDCG |
|---|---|---:|---:|---:|---:|---:|---:|
| Official pooled | EDF-channel | {official.auroc:.6f} | {official.macro_f1:.6f} | {official.ap:.6f} | — | — | — |
| A1-style patient-equal | patient-channel | {a1.auroc:.6f} | {a1.macro_f1:.6f} | {a1.ap:.6f} | {a1.mrr:.6f} | {a1.top1:.6f} | {a1.ndcg:.6f} |

## Direct answers

1. **Frozen pooled replay:** yes. AUROC `{official.auroc:.10f}` is within the locked tolerance of `{EXPECTED_AUROC:.10f}`.
2. **Official labeled EDF-channel units:** `{int(official.n_units)}` ({baseline['normal_pairs']} normal, {baseline['pathological_pairs']} pathological).
3. **Unique patient-channel units after fixed EDF mean:** `{collapse['unique_patient_channel_units']}`.
4. **TEST patients:** `{collapse['patients']}`.
5. **Patients with both classes:** `{int(auroc['n'])}`; single-class patients: `{collapse['single_class_patients']}`.
6. **Patient-equal AUROC:** `{a1.auroc:.6f}`.
7. **Patient AUROC distribution:** SD `{auroc['sd']:.6f}`, median `{auroc['median']:.6f}`, IQR `[{auroc['q25']:.6f}, {auroc['q75']:.6f}]` across the `{int(auroc['n'])}` estimable patients.
8. **Patient-equal Macro-F1 at 0.5:** `{a1.macro_f1:.6f}`. This averages every patient, including single-class patients, with `labels=[0,1]` and `zero_division=0`.
9. **Leakage-free A1-style validation threshold available?** {threshold_sentence}
10. **Thresholded TEST Macro-F1 from a train/validation selection:** unavailable by design.
11. **Pooled minus patient-equal AUROC:** `{delta:+.6f}`. This is descriptive, not causal decomposition.
12. **Center variation:** see table below; no center or patient was removed based on results.
13. **Zurich estimable?** No: all Zurich test patients are single-class under this frozen label set, so patient AUROC is explicitly undefined rather than set to 0.5.
14. **Where is TimeConv stronger?** The observed comparison is stated by `{case}` above; it distinguishes global pooled discrimination from within-patient localization without changing either metric definition.
15. **Implication for unified modeling:** this measurement alone cannot establish a mechanism. It only indicates whether retaining patient-relative channel structure is a reasonable hypothesis to test prospectively; it does not validate a modified model.

## Patient AUROC by center

| Center | Patients | AUROC-estimable patients | Patient-equal AUROC |
|---|---:|---:|---:|
{center_lines}

## Provenance and safeguards

- The CNN was not loaded and no neural inference was run. Scores were recomputed only from the frozen segment-logit cache.
- The cache hard-bound the official checkpoint `{CHECKPOINT_SHA256}` and source `{OFFICIAL_SOURCE_SHA256}`. Its cache digest was `{baseline['test_cache_sha256']}` and it is linked by earlier replay provenance to frozen channel-prediction fingerprint `{FROZEN_CHANNEL_PREDICTIONS_SHA256}`.
- The score is `mean_t(1 - sigmoid(normal_logit_t))`; this is sigmoid per segment followed by averaging, never `sigmoid(mean(logit))`.
- Official raw label `1=normal, 0=SOZ/pathological, -1=excluded` was converted once during frozen cache construction to `pathological_labels = 1 - official_label`; this evaluation used those frozen values unchanged.
- Identifying patient/channel rows are retained only in the private runtime directory. Git-tracked files are aggregate-only.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing public output: {args.output}")
    if args.private_output.exists() and any(args.private_output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing private output: {args.private_output}")
    # A prior failed engineering invocation can leave empty destinations; the
    # non-empty guard above still prevents replacement of any result artifact.
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)
    centers = patient_center(args.split_csv)
    raw, cache = load_frozen_cache(args.test_cache)
    if cache["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError(f"Frozen cache digest mismatch: {cache['cache_sha256']}")
    if len(raw) != 16543 or cache["cache_segments"] != 240074:
        raise RuntimeError("Frozen cache cohort/segment count differs from audited TEST representation cache")
    if not set(raw.y.unique()).issubset({-1, 0, 1}):
        raise RuntimeError("Frozen cache contains unsupported label values")
    labeled = raw.loc[raw.y.isin([0, 1])].copy()
    if len(labeled) != 8104 or int((labeled.y == 0).sum()) != 7297 or int((labeled.y == 1).sum()) != 807:
        raise RuntimeError("Official labeled EDF-channel population mismatch")
    if labeled.patient.nunique() != 96:
        raise RuntimeError("Frozen labeled patient count mismatch")
    pooled_auc = float(roc_auc_score(labeled.y, labeled.pathological_score))
    pooled_ap = float(average_precision_score(labeled.y, labeled.pathological_score))
    pooled_macro = float(f1_score(labeled.y, labeled.pathological_score >= THRESHOLD,
                                 labels=[0, 1], average="macro", zero_division=0))
    if abs(pooled_auc - EXPECTED_AUROC) > TOLERANCE or abs(pooled_macro - EXPECTED_MACRO_F1_HALF) > TOLERANCE:
        raise RuntimeError("Hard stop: frozen pooled baseline did not reproduce")
    if any(patient not in centers for patient in labeled.patient.unique()):
        raise RuntimeError("Missing center mapping for a frozen test patient")
    labeled["center"] = labeled.patient.map(centers)

    label_counts = labeled.groupby(["patient", "channel"], sort=True).y.nunique()
    conflict = label_counts.loc[label_counts != 1]
    if len(conflict):
        raise RuntimeError(f"Hard stop: {len(conflict)} patient/channel labels conflict across EDFs")
    collapsed = labeled.groupby(["patient", "channel"], as_index=False, sort=True).agg(
        y=("y", "first"), pathological_score=("pathological_score", "mean"),
        center=("center", "first"), edf_count=("edf", "nunique"),
        segment_count=("n_segments", "sum"),
    )
    if collapsed.center.isna().any() or collapsed.groupby(["patient", "channel"]).size().max() != 1:
        raise RuntimeError("Patient-channel aggregation invalid")
    private_rows = []
    for patient, group in collapsed.groupby("patient", sort=True):
        row = one_patient_metrics(group, THRESHOLD)
        row.update({"patient": patient, "center": str(group.center.iloc[0])})
        private_rows.append(row)
    private = pd.DataFrame(private_rows).sort_values("patient").reset_index(drop=True)
    if len(private) != 96:
        raise RuntimeError("Patient-level aggregation changed test population")
    atomic_csv(args.private_output / "PATIENT_CHANNEL_PRIVATE.csv", collapsed)
    atomic_csv(args.private_output / "PATIENT_LEVEL_PRIVATE.csv", private)

    aggregate = aggregate_patient_metrics(private)
    centerwise = center_summary(private)
    support = private.groupby("class_support", as_index=False).agg(
        n_patients=("patient", "size"), mean_patient_channels=("n_patient_channels", "mean"),
        median_patient_channels=("n_patient_channels", "median"),
        mean_true_pathological_fraction=("true_pathological_fraction", "mean"),
    )
    distribution = collapsed.edf_count.value_counts().sort_index()
    aggregation_rows = [
        {"section": "counts", "name": "edf_channel_units_before_aggregation", "value": int(len(labeled))},
        {"section": "counts", "name": "unique_patient_channel_units_after_aggregation", "value": int(len(collapsed))},
        {"section": "counts", "name": "patients", "value": int(len(private))},
        {"section": "counts", "name": "duplicated_patient_channel_identities_across_edfs", "value": int((collapsed.edf_count > 1).sum())},
    ] + [{"section": "edf_count_distribution", "name": str(int(count)), "value": int(n)} for count, n in distribution.items()]
    aggregation_audit = pd.DataFrame(aggregation_rows)
    # No valid held-out validation artifact bound to this frozen CNN was
    # recovered.  The full-record TRAIN cache is explicitly in-sample and is
    # recorded only as a rejected candidate rather than used to tune a cutoff.
    threshold_audit = pd.DataFrame([
        {"strategy": "fixed_0.5_diagnostic", "status": "APPLIED", "threshold": THRESHOLD,
         "selection_data": "none", "test_labels_used_for_selection": False,
         "reason": "Predeclared fixed diagnostic."},
        {"strategy": "a1_style_train_validation_patient_macro_f1", "status": "UNAVAILABLE", "threshold": np.nan,
         "selection_data": "none", "test_labels_used_for_selection": False,
         "reason": "No saved patient-held-out validation prediction artifact is bound to the frozen checkpoint. The legal full-record TRAIN cache is in-sample and was rejected to avoid optimistic threshold reuse."},
    ])
    pe_auc = describe(private.auroc)["mean"]
    pe_macro = describe(private.macro_f1)["mean"]
    pe_ap = describe(private.ap)["mean"]
    pe_mrr = describe(private.mrr)["mean"]
    pe_top1 = describe(private.top1)["mean"]
    pe_ndcg = describe(private.ndcg)["mean"]
    summary = pd.DataFrame([
        {"evaluation_system": "Official pooled / EDF-channel", "unit": "EDF-channel", "auroc": pooled_auc,
         "macro_f1": pooled_macro, "threshold": THRESHOLD, "ap": pooled_ap, "mrr": np.nan, "top1": np.nan,
         "ndcg": np.nan, "n_units": len(labeled), "n_patients": len(private),
         "n_auroc_estimable_patients": int(private.auroc_estimable.sum())},
        {"evaluation_system": "A1-style patient-equal / patient-channel @0.5", "unit": "patient-channel", "auroc": pe_auc,
         "macro_f1": pe_macro, "threshold": THRESHOLD, "ap": pe_ap, "mrr": pe_mrr, "top1": pe_top1,
         "ndcg": pe_ndcg, "n_units": len(collapsed), "n_patients": len(private),
         "n_auroc_estimable_patients": int(private.auroc_estimable.sum())},
        {"evaluation_system": "A1-style patient-equal / patient-channel @TRAIN-VAL threshold", "unit": "patient-channel", "auroc": pe_auc,
         "macro_f1": np.nan, "threshold": np.nan, "ap": pe_ap, "mrr": pe_mrr, "top1": pe_top1,
         "ndcg": pe_ndcg, "n_units": len(collapsed), "n_patients": len(private),
         "n_auroc_estimable_patients": int(private.auroc_estimable.sum()),
         "availability": "UNAVAILABLE_NO_LEAKAGE_FREE_VALIDATION_ARTIFACT"},
    ])
    baseline = {
        "status": "PASS_REUSED_FROZEN_REPRESENTATION_CACHE", "checkpoint_sha256": CHECKPOINT_SHA256,
        "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
        "test_cache_sha256": cache["cache_sha256"],
        "linked_historical_channel_prediction_sha256": FROZEN_CHANNEL_PREDICTIONS_SHA256,
        "labeled_edf_channel_pairs": int(len(labeled)), "normal_pairs": int((labeled.y == 0).sum()),
        "pathological_pairs": int((labeled.y == 1).sum()), "patient_clusters": int(labeled.patient.nunique()),
        "segment_rows": int(cache["cache_segments"]), "pooled_auroc": pooled_auc, "pooled_ap": pooled_ap,
        "pooled_macro_f1_at_0_5": pooled_macro, "expected_auroc": EXPECTED_AUROC,
        "expected_half_macro_f1": EXPECTED_MACRO_F1_HALF, "tolerance": TOLERANCE,
        "neural_inference_run": False, "predictions_adjusted": False,
        "score_aggregation": "1 - mean(sigmoid(normal_oriented_segment_logit)) per EDF-channel",
    }
    lock = {
        "experiment": "omni_timeconv_a1style_patient_eval_seed42_v1", "mode": "evaluation_only_frozen_predictions",
        "checkpoint_sha256": CHECKPOINT_SHA256, "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
        "frozen_channel_prediction_sha256": FROZEN_CHANNEL_PREDICTIONS_SHA256,
        "representation_cache_sha256": EXPECTED_CACHE_SHA256, "expected_labeled_edf_channel_units": 8104,
        "expected_normal": 7297, "expected_pathological": 807, "expected_pooled_auroc": EXPECTED_AUROC,
        "fixed_patient_channel_aggregation": "mean EDF-channel pathological score within (patient, channel)",
        "fixed_diagnostic_threshold": THRESHOLD,
        "a1_threshold_policy": "unavailable without a frozen patient-held-out validation artifact; do not use in-sample TRAIN predictions",
        "forbidden": ["training", "neural inference", "test threshold optimization", "patient-specific threshold", "test-selected aggregation", "population modification"],
    }
    collapse_stats = {"unique_patient_channel_units": int(len(collapsed)), "patients": int(len(private)),
                      "single_class_patients": int((~private.auroc_estimable).sum()),
                      "duplicated_patient_channel_identities": int((collapsed.edf_count > 1).sum())}
    atomic_json(args.output / "PROTOCOL_LOCK.json", lock)
    atomic_json(args.output / "BASELINE_REPLAY_AUDIT.json", baseline)
    atomic_csv(args.output / "PATIENT_CHANNEL_AGGREGATION_AUDIT.csv", aggregation_audit)
    atomic_csv(args.output / "PATIENT_CLASS_SUPPORT.csv", support)
    atomic_csv(args.output / "PATIENT_LEVEL_METRICS.csv", aggregate)
    atomic_csv(args.output / "CENTERWISE_PATIENT_METRICS.csv", centerwise)
    atomic_csv(args.output / "THRESHOLD_SELECTION_AUDIT.csv", threshold_audit)
    atomic_csv(args.output / "SUMMARY_TABLE.csv", summary)
    label_audit = f"""# Label semantics audit

The official Task2 raw construction is `-1` for excluded channels, `1` for a normal channel (`patient_outcome == 1 and channel_resection == 0`), and `0` for an SOZ/pathological channel (`channel_soz == 1`). The frozen representation extractor applied `pathological_labels = where(official_label >= 0, 1 - official_label, -1)` before this evaluation.

This run read only the resulting frozen `pathological_labels` field. Its values were restricted to `-1, 0, 1`; the `-1` rows were excluded. The retained test cohort contains `{int((labeled.y == 0).sum())}` normal (`y_pathological=0`) and `{int((labeled.y == 1).sum())}` pathological/SOZ (`y_pathological=1`) EDF-channel units. No labels were recomputed, repaired, or reinterpreted.

The source population/filtering was already frozen in the cache: official Task2 `test`, frequency >900, interictal, length >=62, dataset != Multicenter, and official good channels. This script did not reread raw recordings or test metadata to alter membership.
"""
    (args.output / "LABEL_SEMANTICS_AUDIT.md").write_text(label_audit, encoding="utf-8")
    (args.output / "FINAL_REPORT.md").write_text(public_report(summary, baseline, collapse_stats, False, aggregate, centerwise), encoding="utf-8")
    atomic_json(args.output / "REQUIRED_OUTPUTS_STATUS.json", {"status": "PASS", "private_rows_retained_server_only": True,
                                                                  "threshold_train_validation_available": False})
    print(json.dumps({"status": "PASS", "pooled_auroc": pooled_auc, "patient_equal_auroc": pe_auc,
                      "patient_equal_macro_f1_at_0_5": pe_macro, "patient_channels": len(collapsed),
                      "patients": len(private), "auroc_estimable_patients": int(private.auroc_estimable.sum())}, sort_keys=True))


if __name__ == "__main__":
    main()
