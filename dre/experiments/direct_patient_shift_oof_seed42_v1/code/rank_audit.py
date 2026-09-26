"""Retrospective B0/BCR rank-space oracle audit; no deployable rank model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


GRID = np.round(np.arange(0, 1.0001, 0.1), 1)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percent_rank(score: np.ndarray) -> np.ndarray:
    return rankdata(score, method="average") / (len(score) + 1)


def rank_metrics(y_nez: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    ez = (y_nez == 0).astype(int)
    if not 0 < ez.sum() < len(ez):
        return {name: float("nan") for name in ("ez_auprc", "ez_auroc", "ez_mrr", "top1_is_ez", "true_k_recall")}
    order = np.argsort(-scores, kind="mergesort")
    hits = np.flatnonzero(ez[order] == 1)
    k = int(ez.sum())
    return {"ez_auprc": float(average_precision_score(ez, scores)),
            "ez_auroc": float(roc_auc_score(ez, scores)),
            "ez_mrr": float(1 / (hits[0] + 1)),
            "top1_is_ez": float(ez[order[0]]),
            "true_k_recall": float(ez[order[:k]].sum() / k)}


def threshold_oracle_macro(y_nez: np.ndarray, score_ez: np.ndarray) -> float:
    """Best Macro-F1 over all distinct score-tie respecting top-k decision regions."""
    order = np.argsort(-score_ez, kind="mergesort")
    sorted_scores = score_ez[order]
    ez = (y_nez[order] == 0).astype(int)
    ends = np.concatenate(([0], np.flatnonzero(np.diff(sorted_scores) != 0) + 1, [len(ez)]))
    tp_prefix = np.concatenate(([0], np.cumsum(ez)))
    tp = tp_prefix[ends]
    fp = ends - tp
    total_ez = int(ez.sum())
    total_nez = len(ez) - total_ez
    fn = total_ez - tp
    tn = total_nez - fp
    ez_denominator = 2 * tp + fp + fn
    nez_denominator = 2 * tn + fp + fn
    ez_f1 = np.divide(2 * tp, ez_denominator, out=np.zeros_like(tp, dtype=float), where=ez_denominator > 0)
    nez_f1 = np.divide(2 * tn, nez_denominator, out=np.zeros_like(tn, dtype=float), where=nez_denominator > 0)
    return float(np.max((ez_f1 + nez_f1) / 2))


def average(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.nanmean(values))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("b0-ledger", "cdel-ledger", "protocol-lock", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    lock = json.loads(a.protocol_lock.read_text(encoding="utf-8"))
    if lock["rank_audit"]["lambda_grid"] != GRID.tolist() or lock["seed"] != 42:
        raise RuntimeError("Rank diagnostic protocol lock changed")
    b0 = pd.read_csv(a.b0_ledger)
    bcr = pd.read_csv(a.cdel_ledger)
    keys = ["subject_id", "outer_fold", "channel_name"]
    if b0[keys].duplicated().any() or bcr[keys].duplicated().any():
        raise RuntimeError("Duplicate B0/BCR channel identity")
    compare = bcr[keys + ["label_nez", "v3_score_ez"]].rename(columns={
        "label_nez": "bcr_label_nez", "v3_score_ez": "bcr_score_ez"})
    aligned = b0.merge(compare, on=keys, how="inner", validate="one_to_one")
    if len(aligned) != len(b0) or len(aligned) != len(bcr) or len(aligned) != 7635:
        raise RuntimeError("B0/BCR channel sets do not exactly align")
    if not np.array_equal(aligned.label_nez.to_numpy(int), aligned.bcr_label_nez.to_numpy(int)):
        raise RuntimeError("B0/BCR labels differ")
    if aligned.subject_id.nunique() != 80 or sorted(aligned.outer_fold.unique()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected 80 patients across five frozen folds")
    audit = {"status": "PASS", "patients": 80, "channels": 7635, "outer_folds": 5,
             "key_alignment": "EXACT_ONE_TO_ONE", "label_alignment": "EXACT",
             "b0_ledger_sha256": digest(a.b0_ledger), "cdel_ledger_sha256": digest(a.cdel_ledger),
             "protocol_lock_sha256": digest(a.protocol_lock),
             "historical_outer_previously_viewed": True, "rank_oracles_deployable": False}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "RANK_ALIGNMENT_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    private = []
    patient_data = []
    for (fold, patient), group in aligned.groupby(["outer_fold", "subject_id"], sort=True):
        y = group.label_nez.to_numpy(int)
        base = group.score_ez_probability.to_numpy(float)
        other = group.bcr_score_ez.to_numpy(float)
        r0, r1 = percent_rank(base), percent_rank(other)
        thresholds = group.selected_threshold.to_numpy(float)
        if np.ptp(thresholds) > 1e-12:
            raise RuntimeError("B0 threshold changed within patient")
        threshold_ez = math.log(float(thresholds[0]) / (1 - float(thresholds[0]))) * -1
        logit_ez = np.log(np.clip(base, 1e-7, 1 - 1e-7) / np.clip(1 - base, 1e-7, 1))
        patient_data.append({"fold": int(fold), "patient_id": patient, "y": y, "r0": r0, "r1": r1,
                             "logit_ez": logit_ez, "global_threshold_ez": threshold_ez})
        for lam in GRID:
            mixed = (1 - lam) * r0 + lam * r1
            private.append({"outer_fold": int(fold), "patient_id": patient, "lambda": float(lam),
                            **rank_metrics(y, mixed), "oracle_macro_f1": threshold_oracle_macro(y, mixed)})
    patient_frame = pd.DataFrame(private)
    if len(patient_frame) != 880:
        raise RuntimeError("Expected 80 patients times 11 lambda values")
    patient_frame.to_csv(a.output_dir / "RANK_MIX_PATIENT_PRIVATE.csv", index=False)
    grid = []
    for lam, group in patient_frame.groupby("lambda", sort=True):
        grid.append({"lambda": float(lam), "patients": len(group),
                     **{key: float(group[key].mean()) for key in ("ez_auprc", "ez_auroc", "ez_mrr", "top1_is_ez",
                                                                 "true_k_recall", "oracle_macro_f1")}})
    grid_frame = pd.DataFrame(grid).sort_values("lambda")
    grid_frame.to_csv(a.output_dir / "RANK_MIX_GRID.csv", index=False)
    winner = grid_frame.sort_values(["oracle_macro_f1", "lambda"], ascending=[False, True]).iloc[0]
    best_lambda = float(winner["lambda"])
    patient_best = patient_frame.groupby("patient_id").oracle_macro_f1.max().mean()
    baseline = grid_frame.iloc[0]
    uncertainty_rows = []
    for fraction in (0.1, 0.2, 0.3):
        for source in ("B0", "BCR", "GLOBAL_ORACLE_MIX"):
            measurements = []
            for item in patient_data:
                count = max(2, math.ceil(fraction * len(item["y"])))
                hard = np.argsort(np.abs(item["logit_ez"] - item["global_threshold_ez"]), kind="mergesort")[:count]
                y = item["y"][hard]
                values = (item["r0"] if source == "B0" else item["r1"] if source == "BCR"
                          else (1 - best_lambda) * item["r0"] + best_lambda * item["r1"])[hard]
                metrics = rank_metrics(y, values)
                if np.isfinite(metrics["ez_auprc"]):
                    measurements.append(metrics)
            uncertainty_rows.append({"hardest_channel_fraction": fraction, "score": source,
                                     "eligible_patients_with_both_classes": len(measurements),
                                     **{key: average(measurements, key) if measurements else float("nan")
                                        for key in ("ez_auprc", "ez_auroc", "ez_mrr", "top1_is_ez", "true_k_recall")}})
    pd.DataFrame(uncertainty_rows).to_csv(a.output_dir / "RANK_BOUNDARY_DIAGNOSTICS.csv", index=False)
    consistent = (winner.ez_auprc > baseline.ez_auprc + 1e-12 and winner.ez_mrr > baseline.ez_mrr + 1e-12
                  and winner.top1_is_ez >= baseline.top1_is_ez - 1e-12)
    ceiling = winner.oracle_macro_f1 >= 0.72 or winner.oracle_macro_f1 - baseline.oracle_macro_f1 >= 0.015
    decision = bool(ceiling and consistent)
    report = f"""# Retrospective B0/BCR ranking headroom audit\n\nThis is an oracle-only analysis on historical outer outcomes, not a deployable model or a trained distillation student. Alignment: 80 patients, 7,635 exact patient/fold/channel keys, no label mismatch.\n\n- B0 rank-space patient-threshold oracle Macro-F1: `{baseline.oracle_macro_f1:.6f}`.\n- Best **global-lambda oracle**: `lambda={best_lambda:.1f}`, Macro-F1 `{winner.oracle_macro_f1:.6f}` (delta `{winner.oracle_macro_f1-baseline.oracle_macro_f1:+.6f}`). Lambda was selected using outer labels and is not deployable.\n- Best **patient-lambda oracle**: Macro-F1 `{patient_best:.6f}`. Each patient's lambda was selected using that patient's labels; it is a stronger non-deployable bound.\n- Global mix ranking vs B0: EZ-AUPRC `{winner.ez_auprc:.6f}` vs `{baseline.ez_auprc:.6f}`; EZ-MRR `{winner.ez_mrr:.6f}` vs `{baseline.ez_mrr:.6f}`; Top-1 EZ `{winner.top1_is_ez:.4f}` vs `{baseline.top1_is_ez:.4f}`.\n- Predeclared ceiling condition (>=0.72 or >=+0.015) met: `{ceiling}`; ranking consistency (AUPRC and MRR higher, Top-1 nonnegative) met: `{consistent}`.\n\n`RANK_DISTILLATION_WORTHWHILE = {'YES' if decision else 'NO'}` under the locked diagnostic rule. Even YES would justify only a future separately preregistered distillation test, not a deployable 0.70 claim. The boundary diagnostics restrict to the fixed hardest 10/20/30% by distance to the frozen global B0 logit threshold; rows exclude subsets lacking either class and report eligible counts.\n"""
    (a.output_dir / "RANK_HEADROOM_REPORT.md").write_text(report, encoding="utf-8")
    summary = {"best_global_lambda": best_lambda, "b0_oracle_macro_f1": float(baseline.oracle_macro_f1),
               "global_mix_oracle_macro_f1": float(winner.oracle_macro_f1),
               "patient_lambda_oracle_macro_f1": float(patient_best),
               "rank_distillation_worthwhile": decision}
    print("RANK_AUDIT_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
