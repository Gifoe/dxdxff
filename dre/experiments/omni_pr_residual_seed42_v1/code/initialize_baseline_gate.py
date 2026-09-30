"""Validate the existing frozen baseline replay before residual development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--segment-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != lock["frozen_checkpoint_sha256"] or \
            source.get("checkpoint_sha256") != lock["frozen_checkpoint_sha256"] or \
            source.get("status") != "PASS":
        raise RuntimeError("Frozen baseline provenance mismatch")
    frame = pd.read_csv(args.segment_predictions)
    probability = np.clip(frame.normal_prob.to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    frame["normal_logit"] = np.log(probability / (1 - probability))
    grouped = frame.groupby(["edf", "channel"], sort=False).agg(
        normal_label=("normal_label", "first"),
        normal_prob=("normal_prob", "mean"),
        normal_logit=("normal_logit", "mean"),
        patient=("patient", "first")).reset_index()
    scored = grouped[grouped.normal_label.isin([0, 1])]
    y = 1 - scored.normal_label.to_numpy(dtype=np.int8)
    official = float(roc_auc_score(y, 1 - scored.normal_prob.to_numpy(dtype=float)))
    mean_logit = float(roc_auc_score(y, 1 / (1 + np.exp(scored.normal_logit.to_numpy(dtype=float)))))
    audit = {
        "status": "PASS" if abs(official - lock["expected_baseline_auroc"]) < lock["baseline_absolute_tolerance"] else "BASELINE_REPLAY_FAILED",
        "reused_existing_frozen_predictions": True,
        "source_audit_sha256": sha256(args.source_audit),
        "segment_predictions_sha256": sha256(args.segment_predictions),
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_sigmoid_then_mean_auroc": official,
        "expected_auroc": lock["expected_baseline_auroc"],
        "absolute_error": abs(official - lock["expected_baseline_auroc"]),
        "mean_logit_then_sigmoid_diagnostic_auroc": mean_logit,
        "aggregation_conflict_delta": mean_logit - official,
        "labeled_edf_channel_pairs": len(scored),
        "patients": int(scored.patient.nunique()),
        "normal_pairs": int((y == 0).sum()),
        "pathological_pairs": int((y == 1).sum()),
        "test_predictions_adjusted": False,
        "test_used_for_residual_selection": False,
    }
    atomic_json(args.output, audit)
    print(json.dumps(audit, sort_keys=True))
    if audit["status"] != "PASS":
        raise RuntimeError("BASELINE_REPLAY_FAILED")


if __name__ == "__main__":
    main()
