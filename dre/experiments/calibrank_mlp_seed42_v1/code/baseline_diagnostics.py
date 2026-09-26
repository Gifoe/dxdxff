"""Retrospective B0/oracle/ranking diagnostics; never fits or selects a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def macro(y_nez: np.ndarray, predicted_nez: np.ndarray) -> float:
    return float(f1_score(y_nez, predicted_nez, labels=[0, 1], average="macro", zero_division=0))


def metrics(y_nez: np.ndarray, predicted_nez: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": macro(y_nez, predicted_nez),
        "ez_f1": float(f1_score(y_nez == 0, predicted_nez == 0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_nez, predicted_nez)),
    }


def ranking(y_nez: np.ndarray, score_ez: np.ndarray) -> dict[str, float]:
    ez = (y_nez == 0).astype(int)
    order = np.argsort(-score_ez, kind="mergesort")
    positives = np.flatnonzero(ez[order] == 1)
    count = int(ez.sum())
    return {
        "ez_auprc": float(average_precision_score(ez, score_ez)) if count else 0.0,
        "ez_auroc": float(roc_auc_score(ez, score_ez)) if 0 < count < len(ez) else float("nan"),
        "ez_mrr": float(1.0 / (positives[0] + 1)) if len(positives) else 0.0,
        "top1_is_ez": float(ez[order[0]]),
        "true_k_recall": float(ez[order[:count]].sum() / count) if count else float("nan"),
    }


def oracle_shift(y_nez: np.ndarray, score_ez: np.ndarray, selected_threshold_nez: float) -> tuple[float, np.ndarray]:
    """Exact finite decision-region enumeration; tie favors smallest |shift|."""
    p = np.clip(score_ez.astype(float), 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p))
    ez_threshold = float(np.clip(1 - selected_threshold_nez, 1e-7, 1 - 1e-7))
    threshold_logit = float(np.log(ez_threshold / (1 - ez_threshold)))
    boundary = np.unique(np.sort(threshold_logit - logits))
    candidates = [0.0, float(boundary[0] - 1), float(boundary[-1] + 1)]
    candidates.extend(float((lo + hi) / 2) for lo, hi in zip(boundary[:-1], boundary[1:]))
    best: tuple[float, float, float] | None = None
    best_prediction = None
    for shift in candidates:
        predicted_nez = (logits + shift <= threshold_logit).astype(int)
        value = macro(y_nez, predicted_nez)
        key = (value, -abs(shift), -shift)
        if best is None or key > best:
            best = key
            best_prediction = predicted_nez
            best_shift = shift
    assert best_prediction is not None
    return best_shift, best_prediction


def true_k(y_nez: np.ndarray, score_ez: np.ndarray) -> np.ndarray:
    count = int((y_nez == 0).sum())
    prediction = np.ones(len(y_nez), dtype=int)
    prediction[np.argsort(-score_ez, kind="mergesort")[:count]] = 0
    return prediction


def top_fraction_disagreement(left: np.ndarray, right: np.ndarray, fraction: float = 0.1) -> float:
    count = max(1, round(len(left) * fraction))
    a = set(np.argsort(-left, kind="mergesort")[:count].tolist())
    b = set(np.argsort(-right, kind="mergesort")[:count].tolist())
    return float(1 - len(a & b) / len(a | b))


def mean_finite(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-ledger", required=True, type=Path)
    parser.add_argument("--cdel-ledger", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    b0 = pd.read_csv(args.b0_ledger)
    cdel = pd.read_csv(args.cdel_ledger)
    split = pd.read_csv(args.split_manifest)
    feature = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    keys = ["subject_id", "outer_fold", "channel_name"]
    if b0[keys].duplicated().any() or cdel[keys].duplicated().any():
        raise RuntimeError("Duplicated patient/fold/channel key")
    reference = cdel[keys + ["label_nez", "p2_score_ez", "v3_score_ez"]].rename(columns={
        "label_nez": "reference_label_nez", "p2_score_ez": "prq_ez", "v3_score_ez": "bcr_ez"
    })
    aligned = b0.merge(reference, on=keys, how="inner", validate="one_to_one")
    if len(aligned) != len(b0) or len(aligned) != len(cdel) or len(aligned) != 7635:
        raise RuntimeError("B0 and historical CDEL channel keys do not align")
    if not np.array_equal(aligned.label_nez.to_numpy(int), aligned.reference_label_nez.to_numpy(int)):
        raise RuntimeError("B0 and historical CDEL labels differ")
    if len(aligned.subject_id.unique()) != 80 or sorted(aligned.outer_fold.unique()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected 80 patients in five historical folds")
    historical_split = split.rename(columns={"split_role": "partition"})
    for fold, group in aligned.groupby("outer_fold"):
        expected = set(historical_split.loc[(historical_split.outer_fold == fold) & (historical_split.partition == "test"), "subject_id"])
        if expected != set(group.subject_id):
            raise RuntimeError(f"Historical test membership differs in fold {fold}")
    patient_rows = []
    for (fold, patient_id), group in aligned.groupby(["outer_fold", "subject_id"], sort=True):
        y = group.label_nez.to_numpy(int)
        p_ez = group.score_ez_probability.to_numpy(float)
        p_nez = group.score_nez_probability.to_numpy(float)
        threshold_values = group.selected_threshold.to_numpy(float)
        if np.ptp(threshold_values) > 1e-12:
            raise RuntimeError("Selected threshold varies within a patient")
        base_pred = (p_nez >= float(threshold_values[0])).astype(int)
        if not np.array_equal(base_pred, group.predicted_nez.to_numpy(int)):
            raise RuntimeError("B0 prediction replay mismatch")
        shifted, shifted_pred = oracle_shift(y, p_ez, float(threshold_values[0]))
        topk_pred = true_k(y, p_ez)
        row = {"subject_id": patient_id, "outer_fold": int(fold), "channels": len(y), "true_ez_count": int((y == 0).sum()),
               "oracle_logit_shift": shifted}
        for prefix, pred in (("b0", base_pred), ("oracle_shift", shifted_pred), ("oracle_true_k", topk_pred)):
            row.update({f"{prefix}_{name}": value for name, value in metrics(y, pred).items()})
        for prefix, values in (("b0", p_ez), ("prq", group.prq_ez.to_numpy(float)), ("bcr", group.bcr_ez.to_numpy(float))):
            row.update({f"{prefix}_{name}": value for name, value in ranking(y, values).items()})
        for left, right in (("b0", "prq"), ("b0", "bcr"), ("prq", "bcr")):
            a = p_ez if left == "b0" else group[f"{left}_ez"].to_numpy(float)
            b = p_ez if right == "b0" else group[f"{right}_ez"].to_numpy(float)
            correlation = spearmanr(a, b).statistic
            row[f"{left}_{right}_spearman"] = float(correlation) if np.isfinite(correlation) else float("nan")
            row[f"{left}_{right}_top10pct_disagreement"] = top_fraction_disagreement(a, b)
        patient_rows.append(row)
    patients = pd.DataFrame(patient_rows)
    b0_macro = mean_finite(patients, "b0_macro_f1")
    if abs(b0_macro - 0.6161672347405565) > 1e-8:
        raise RuntimeError(f"Published B0 seed42 Macro-F1 did not replay: {b0_macro}")
    summary = {
        "status": "RETROSPECTIVE_ORACLE_ONLY",
        "seed": 42, "patients": len(patients), "channels": len(aligned),
        "feature_profile": feature.get("feature_profile"), "feature_dimension": feature.get("final_dimension"),
        "b0": {name: mean_finite(patients, f"b0_{name}") for name in ("macro_f1", "ez_f1", "balanced_accuracy", "ez_auprc", "ez_auroc", "ez_mrr", "top1_is_ez", "true_k_recall")},
        "oracle_shift": {name: mean_finite(patients, f"oracle_shift_{name}") for name in ("macro_f1", "ez_f1", "balanced_accuracy")},
        "oracle_true_k": {name: mean_finite(patients, f"oracle_true_k_{name}") for name in ("macro_f1", "ez_f1", "balanced_accuracy")},
        "ranking": {name: {metric: mean_finite(patients, f"{name}_{metric}") for metric in ("ez_auprc", "ez_auroc", "ez_mrr", "top1_is_ez", "true_k_recall")}
                    for name in ("b0", "prq", "bcr")},
        "agreement": {f"{left}_{right}": {"spearman": mean_finite(patients, f"{left}_{right}_spearman"),
                                          "top10pct_disagreement": mean_finite(patients, f"{left}_{right}_top10pct_disagreement")}
                      for left, right in (("b0", "prq"), ("b0", "bcr"), ("prq", "bcr"))},
        "shift_distribution": {name: float(np.quantile(patients.oracle_logit_shift, value)) for name, value in (("q10", 0.1), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q90", 0.9))},
        "shift_patient_outcomes": {"improved": int((patients.oracle_shift_macro_f1 > patients.b0_macro_f1 + 1e-12).sum()),
                                   "degraded": int((patients.oracle_shift_macro_f1 < patients.b0_macro_f1 - 1e-12).sum()),
                                   "unchanged": int((abs(patients.oracle_shift_macro_f1 - patients.b0_macro_f1) <= 1e-12).sum())},
        "provenance": {"b0_ledger_sha256": digest(args.b0_ledger), "cdel_ledger_sha256": digest(args.cdel_ledger),
                       "split_manifest_sha256": digest(args.split_manifest), "feature_manifest_sha256": digest(args.feature_manifest)},
    }
    summary["oracle_shift"]["delta_vs_b0_macro_f1"] = summary["oracle_shift"]["macro_f1"] - b0_macro
    summary["oracle_true_k"]["delta_vs_b0_macro_f1"] = summary["oracle_true_k"]["macro_f1"] - b0_macro
    # Gate declared before examining these outcomes. No oracle metric becomes deployable.
    summary["constructive_gate"] = {
        "rule": "Proceed only if oracle shift or true-K gains >= 0.01 Macro-F1, or BCR ranking AUPRC exceeds B0 by >= 0.005.",
        "pass": bool(max(summary["oracle_shift"]["delta_vs_b0_macro_f1"], summary["oracle_true_k"]["delta_vs_b0_macro_f1"]) >= 0.01
                     or summary["ranking"]["bcr"]["ez_auprc"] - summary["ranking"]["b0"]["ez_auprc"] >= 0.005),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(args.output_dir / "DIAGNOSTICS_PATIENT_PRIVATE.csv", index=False)
    (args.output_dir / "DIAGNOSTICS_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
