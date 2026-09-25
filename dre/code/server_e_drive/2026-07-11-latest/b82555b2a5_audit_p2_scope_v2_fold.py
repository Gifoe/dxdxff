"""Independent formal-mask recomputation for one completed SCOPE-v2 fold."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from neuroez_c.p2_scope_v2_decoder import FORMAL_PREDICTION_SOURCE, PREDICTED_K_DECISION_RULE


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", required=True); parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args(); root = Path(args.run_dir); channels = pd.read_csv(root / "p2_scope_v2_channel_predictions.csv")
    channels = channels[channels.outer_fold == args.fold].copy(); summary = pd.read_csv(root / f"fold_{args.fold}" / "fold_summary.csv").iloc[0]
    if channels.empty: raise RuntimeError("No channel predictions found for requested fold")
    if channels.decision_rule.nunique() != 1 or channels.decision_rule.iloc[0] != PREDICTED_K_DECISION_RULE: raise RuntimeError("Formal decision rule is not predicted_k_topk")
    if channels.formal_prediction_source.nunique() != 1 or channels.formal_prediction_source.iloc[0] != FORMAL_PREDICTION_SOURCE: raise RuntimeError("Formal prediction source is invalid")
    if channels.true_count_used_for_prediction.astype(bool).any(): raise RuntimeError("True count was used for formal prediction")
    patient_rows = []
    for subject, group in channels.groupby("subject_id", sort=True):
        if int(group.predicted_ez.sum()) != int(group.scope_predicted_k.iloc[0]): raise RuntimeError(f"Predicted K mask mismatch for {subject}")
        y, p = group.label_nez.to_numpy(int), group.predicted_nez.to_numpy(int)
        patient_rows.append({"macro_f1": f1_score(y, p, average="macro", labels=[0, 1], zero_division=0), "ez_f1": f1_score(1 - y, 1 - p, pos_label=1, zero_division=0), "nez_f1": f1_score(y, p, pos_label=1, zero_division=0), "balanced_accuracy": balanced_accuracy_score(y, p)})
    observed = {"patient_macro_f1": sum(row["macro_f1"] for row in patient_rows) / len(patient_rows), "patient_macro_ez_f1": sum(row["ez_f1"] for row in patient_rows) / len(patient_rows), "patient_macro_nez_f1": sum(row["nez_f1"] for row in patient_rows) / len(patient_rows), "patient_macro_balanced_accuracy": sum(row["balanced_accuracy"] for row in patient_rows) / len(patient_rows)}
    for name, value in observed.items():
        if abs(float(summary[name]) - value) > 1e-10: raise RuntimeError(f"Offline recomputation mismatch: {name}")
    if not math.isnan(float(summary.classification_threshold)) or str(summary.threshold_source) != "not_used": raise RuntimeError("Formal predicted-K result fell back to a threshold")
    print(f"passed fold={args.fold} patients={len(patient_rows)} formal_macro_f1={observed['patient_macro_f1']:.12f}")


if __name__ == "__main__": main()
