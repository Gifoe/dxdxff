"""Recreate and freeze original test predictions once, with hard equality gates.

The previous official CNN evaluator wrote only aggregate metrics. This
authorized replay uses its identical source/checkpoint/preprocessing and
test grouping, then refuses to persist any prediction if aggregate F1/AUC
or the selected Youden threshold differs from that completed evaluation.
Patient/EDF/channel records are private server-only artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"Frozen input SHA mismatch: {path.name}: {got}")


def close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise RuntimeError(f"Original-result reproduction mismatch for {name}: {actual} vs {expected}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--source-result", type=Path, required=True)
    p.add_argument("--source-protocol", type=Path, required=True)
    p.add_argument("--training-audit", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--original-code", type=Path, required=True)
    p.add_argument("--test-features", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    args = p.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require_hash(args.source_result, lock["source_result_sha256"])
    require_hash(args.source_protocol, lock["source_protocol_sha256"])
    require_hash(args.checkpoint, lock["frozen_checkpoint_sha256"])
    require_hash(args.official_cnn, lock["official_cnn_source_sha256"])
    old = json.loads(args.source_result.read_text(encoding="utf-8"))
    training = json.loads(args.training_audit.read_text(encoding="utf-8"))
    extraction = json.loads((args.test_features / "EXTRACTION_TEST_AUDIT.json").read_text(encoding="utf-8"))
    for record in (old, training, extraction):
        if record["protocol_sha256"] != lock["source_protocol_sha256"]:
            raise RuntimeError("Source protocol differs among frozen artifacts")
    if not training["completed"] or not extraction["completed"]:
        raise RuntimeError("Source training or test extraction incomplete")
    if training["checkpoint_sha256"] != lock["frozen_checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint differs from frozen run")
    if old["labeled_edf_channel_pairs"] != lock["expected_labeled_pairs"]:
        raise RuntimeError("Frozen test cohort size mismatch")

    sys.path.insert(0, str(args.original_code))
    from train_official_cnn import official_module  # noqa: E402

    # Do not alter cuDNN flags: the completed evaluator used process defaults.
    source = official_module(args.official_cnn)
    device = torch.device("cuda")
    model = source.NeuralCNN(in_channels=1, outputs=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)["model_state_dict"])
    model.eval()
    prep = source.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)

    rows: list[tuple[str, str, str, int, float]] = []
    files = sorted((args.test_features / "test").rglob("*.npz"))
    if len(files) != extraction["npz_files"]:
        raise RuntimeError("Test NPZ file count changed")
    for ordinal, file in enumerate(files, start=1):
        with np.load(file, allow_pickle=False) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            names = np.asarray(archive["name"]).astype(str)
            labels = np.asarray(archive["labels"], dtype=np.int8)
            edf = str(archive["edf_name"])
            patient = str(archive["patient"])
        probs: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(data), 32):
                x = torch.from_numpy(data[start:start + 32]).to(device)
                image = prep(x)
                logits = model(image).squeeze(1)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        if len(probs) != len(data):
            raise RuntimeError("Incomplete deterministic inference")
        rows.extend((edf, patient, name, int(label), float(prob))
                    for name, label, prob in zip(names, labels, probs))
        print(json.dumps({"test_edf_ordinal": ordinal, "samples": len(data)}), flush=True)

    segments = pd.DataFrame(rows, columns=["edf", "patient", "channel", "normal_label", "normal_prob"])
    if len(segments) != extraction["samples"]:
        raise RuntimeError("Test clip count changed")
    # Match original evaluate_official_cnn.py: group only by (edf, channel),
    # preserve encounter order, average sigmoid normal probabilities.
    grouped = segments.groupby(["edf", "channel"], sort=False).agg(
        normal_label=("normal_label", "first"),
        label_unique=("normal_label", "nunique"),
        normal_prob=("normal_prob", "mean"),
        patient=("patient", "first"),
        patient_unique=("patient", "nunique"),
        clips=("normal_prob", "size")).reset_index()
    if not bool(((grouped.label_unique == 1) & (grouped.patient_unique == 1)).all()):
        raise RuntimeError("Test label or patient conflict within an EDF-channel")
    scored = grouped.loc[grouped.normal_label.isin([0, 1])].copy().reset_index(drop=True)
    y = 1 - scored.normal_label.to_numpy(dtype=int)
    s = 1 - scored.normal_prob.to_numpy(dtype=float)
    if len(y) != lock["expected_labeled_pairs"] or int((y == 0).sum()) != lock["expected_normal_pairs"] or int((y == 1).sum()) != lock["expected_pathological_pairs"]:
        raise RuntimeError("Test labeled-pair counts changed")
    fpr, tpr, thresholds = roc_curve(y, s)
    youden = float(thresholds[np.argmax(tpr - fpr)])
    f1_youden = float(f1_score(y, s >= youden, average="macro"))
    f1_half = float(f1_score(y, s >= 0.5, average="macro"))
    auroc = float(roc_auc_score(y, s))
    tol = float(lock["comparison_absolute_tolerance"])
    for name, actual, expected in (
        ("test-Youden threshold", youden, old["test_derived_youden_threshold_reproduction_only"]),
        ("test-Youden Macro-F1", f1_youden, old["reproduced_macro_f1_official_test_youden"]),
        ("fixed-half Macro-F1", f1_half, old["reproduced_macro_f1_fixed_half_diagnostic"]),
        ("channel AUROC", auroc, old["reproduced_channel_auroc"]),
    ):
        close(name, actual, expected, tol)

    args.private_output.mkdir(parents=True, exist_ok=True)
    segment_path = args.private_output / "frozen_segment_predictions.csv.gz"
    channel_path = args.private_output / "frozen_channel_predictions.csv"
    if segment_path.exists() or channel_path.exists():
        raise RuntimeError("Private prediction files already exist; refusing overwrite")
    segment_tmp = segment_path.with_suffix(segment_path.suffix + ".partial")
    channel_tmp = channel_path.with_suffix(channel_path.suffix + ".partial")
    segments.to_csv(segment_tmp, index=False, float_format="%.17g", compression="gzip")
    pd.DataFrame({
        "edf": scored.edf, "patient": scored.patient, "channel": scored.channel,
        "y_true": y, "pathological_score": s, "clips": scored.clips,
    }).to_csv(channel_tmp, index=False, float_format="%.17g")
    os.replace(segment_tmp, segment_path)
    os.replace(channel_tmp, channel_path)
    audit = {
        "status": "PASS", "source_result_sha256": sha256(args.source_result),
        "checkpoint_sha256": sha256(args.checkpoint), "source_protocol_sha256": sha256(args.source_protocol),
        "segment_rows": len(segments), "labeled_edf_channel_pairs": len(scored),
        "normal_pairs": int((y == 0).sum()), "pathological_pairs": int((y == 1).sum()),
        "youden_threshold": youden, "macro_f1_youden": f1_youden,
        "macro_f1_fixed_half": f1_half, "auroc": auroc,
        "patient_clusters": int(scored.patient.nunique()),
        "segment_predictions_sha256": sha256(segment_path),
        "channel_predictions_sha256": sha256(channel_path),
        "test_predictions_adjusted": False,
        "individual_predictions_kept_private": True,
    }
    (args.private_output / "FROZEN_PREDICTION_REPLAY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "segment_rows", "labeled_edf_channel_pairs", "auroc", "macro_f1_youden", "macro_f1_fixed_half")}), flush=True)


if __name__ == "__main__":
    main()
