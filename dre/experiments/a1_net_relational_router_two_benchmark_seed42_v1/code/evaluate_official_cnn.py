"""One frozen official CNN Task-2 test pass, reproduction metrics only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from train_official_cnn import PINNED_CNN_SHA256, official_module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    lock_sha = digest(args.protocol)
    training = json.loads((args.training / "TRAINING_AUDIT.json").read_text())
    extraction = json.loads((args.features / "EXTRACTION_TEST_AUDIT.json").read_text())
    if not training["completed"] or not extraction["completed"]:
        raise RuntimeError("Training/test extraction incomplete")
    if training["protocol_sha256"] != lock_sha or extraction["protocol_sha256"] != lock_sha:
        raise RuntimeError("Protocol hash mismatch before official test")
    checkpoint = args.training / "best_model.pt"
    if digest(checkpoint) != training["checkpoint_sha256"]:
        raise RuntimeError("Best checkpoint changed after training freeze")
    source = official_module(args.official_cnn)
    device = torch.device("cuda")
    model = source.NeuralCNN(in_channels=1, outputs=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=False)["model_state_dict"])
    model.eval()
    prep = source.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)
    rows = []
    for ordinal, file in enumerate(sorted((args.features / "test").rglob("*.npz")), start=1):
        with np.load(file, allow_pickle=False) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            names = np.asarray(archive["name"]).astype(str)
            labels = np.asarray(archive["labels"], dtype=np.int8)
            edf = str(archive["edf_name"])
        probs = []
        with torch.inference_mode():
            for start in range(0, len(data), 32):
                x = torch.from_numpy(data[start:start + 32]).to(device)
                image = prep(x)
                logits = model(image).squeeze(1)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        if len(probs) != len(data):
            raise RuntimeError("Incomplete official CNN test inference")
        for name, label, prob in zip(names, labels, probs):
            rows.append((edf, name, int(label), float(prob)))
        print(json.dumps({"test_edf_ordinal": ordinal, "samples": len(data)}), flush=True)
    if not rows:
        raise RuntimeError("No official test inference rows")
    frame = pd.DataFrame(rows, columns=["edf", "channel", "normal_label", "normal_prob"])
    grouped = frame.groupby(["edf", "channel"], sort=False).agg(
        normal_label=("normal_label", "first"),
        label_unique=("normal_label", "nunique"),
        normal_prob=("normal_prob", "mean"),
        clips=("normal_prob", "size")).reset_index()
    if not bool((grouped["label_unique"] == 1).all()):
        raise RuntimeError("Test label conflict within an EDF-channel")
    scored = grouped.loc[grouped["normal_label"].isin([0, 1])]
    if len(scored) != 8104:
        raise RuntimeError(f"Expected 8104 official labeled EDF-channels, got {len(scored)}")
    y = 1 - scored["normal_label"].to_numpy(dtype=int)
    s = 1 - scored["normal_prob"].to_numpy(dtype=float)
    if len(np.unique(y)) != 2 or not np.isfinite(s).all():
        raise RuntimeError("Invalid official test scores")
    fpr, tpr, thresholds = roc_curve(y, s)
    youden = float(thresholds[np.argmax(tpr - fpr)])
    pred_youden = (s >= youden).astype(int)
    pred_half = (s >= 0.5).astype(int)
    result = {
        "status": "COMPLETE", "classification_unit": "official labeled EDF-channel pair",
        "labeled_edf_channel_pairs": len(scored),
        "normal_pairs": int((y == 0).sum()),
        "pathological_pairs": int((y == 1).sum()),
        "published_macro_f1": 0.6469, "published_channel_auroc": 0.8061,
        "reproduced_macro_f1_official_test_youden": float(f1_score(y, pred_youden, average="macro")),
        "reproduced_macro_f1_fixed_half_diagnostic": float(f1_score(y, pred_half, average="macro")),
        "reproduced_channel_auroc": float(roc_auc_score(y, s)),
        "test_derived_youden_threshold_reproduction_only": youden,
        "test_threshold_used_for_a1_net": False,
        "protocol_sha256": lock_sha,
        "source_cnn_sha256": PINNED_CNN_SHA256,
        "model_checkpoint_sha256": training["checkpoint_sha256"],
        "selected_train_epoch": training["selected_epoch"],
        "exploratory_repeated_test_benchmark": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
