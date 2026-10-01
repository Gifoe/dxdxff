from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit_core import atomic_json, sha256, sigmoid


def train5_rows(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            offsets = np.asarray(z["segment_offsets"], dtype=np.int64)
            flat = 1.0 - sigmoid(np.asarray(z["segment_logits"], dtype=np.float64))
            for i, (channel, y) in enumerate(zip(z["channel_names"].astype(str), z["pathological_labels"])):
                rows.append({"patient": str(z["patient"]), "edf": str(z["edf"]), "channel": channel,
                             "y": int(y), "score": float(flat[offsets[i]:offsets[i + 1]].mean()),
                             "clips": int(offsets[i + 1] - offsets[i])})
    return pd.DataFrame(rows).query("y in [0, 1]").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--train5", type=Path, required=True)
    parser.add_argument("--test-channels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    args = parser.parse_args()
    lock_path = args.experiment / "PROTOCOL_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != lock["frozen_checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint SHA-256 mismatch")
    if sha256(args.official_cnn) != lock["official_cnn_source_sha256"]:
        raise RuntimeError("Official CNN source SHA-256 mismatch")
    test = pd.read_csv(args.test_channels)
    test_auc = float(roc_auc_score(test.y_true, test.pathological_score))
    test_error = abs(test_auc - lock["expected_test_auroc"])
    test_payload = {
        "status": "PASS" if test_error < lock["replay_absolute_tolerance"] else "BASELINE_REPLAY_FAILED",
        "expected_auroc": lock["expected_test_auroc"], "observed_auroc": test_auc,
        "absolute_error": test_error, "tolerance": lock["replay_absolute_tolerance"],
        "labeled_edf_channel_pairs": int(len(test)), "checkpoint_sha256": sha256(args.checkpoint),
        "official_cnn_source_sha256": sha256(args.official_cnn),
        "frozen_channel_predictions_sha256": sha256(args.test_channels),
    }
    atomic_json(args.experiment / "BASELINE_REPLAY_AUDIT.json", test_payload)
    if test_payload["status"] != "PASS":
        raise RuntimeError("BASELINE_REPLAY_FAILED")
    train = train5_rows(args.train5)
    train_auc = float(roc_auc_score(train.y, train.score))
    train_error = abs(train_auc - lock["expected_train5_auroc"])
    payload = {
        "status": "PASS" if train_error < lock["replay_absolute_tolerance"] else "TRAIN5_REPLAY_FAILED",
        "expected_train5_auroc": lock["expected_train5_auroc"], "observed_train5_auroc": train_auc,
        "absolute_error": train_error, "tolerance": lock["replay_absolute_tolerance"],
        "patients": int(train.patient.nunique()), "edfs": int(train.edf.nunique()),
        "labeled_edf_channel_pairs": int(len(train)), "segments": int(train.clips.sum()),
        "p_n_eq_5": float((train.clips == 5).mean()),
    }
    atomic_json(args.experiment / "TRAIN5_REPLAY_AUDIT.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("TRAIN5_REPLAY_FAILED")
    print(json.dumps({"test": test_payload, "train5": payload}, indent=2))


if __name__ == "__main__":
    main()
