"""Hash matched B0/PC checkpoints and train-side thresholds before test I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from train_rawcnn import digest, save_json


def selected_validation_file(work: Path, selection: dict, model: str):
    if model == "B0":
        return work / "B0_RawCNN" / f"validation_epoch_{selection['epoch']:02d}_private.json"
    stage = model
    return work / f"stage_{stage}_validation_epoch_{selection['epoch']:02d}_private.json"


def validation_threshold(path: Path):
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    rows = snapshot["private_patient_scores"]
    labels = np.concatenate([np.asarray(row["labels"], dtype=np.int8) for row in rows.values()])
    scores = np.concatenate([np.asarray(row["scores"], dtype=np.float64) for row in rows.values()])
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise RuntimeError("Invalid train-side Omni threshold pool")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    chosen = max((round(float(f1_score(labels, scores >= threshold,
                                       average="macro", zero_division=0)), 12),
                  -abs(float(threshold) - 0.5), -float(threshold), float(threshold))
                 for threshold in candidates)
    return {"threshold": chosen[-1], "validation_macro_f1": chosen[0],
            "pooled_edf_channel_units": int(len(labels)),
            "candidate_unique_scores": int(len(candidates)),
            "source": "frozen_inner_validation_only"}


def checked_checkpoint(path: Path, protocol_sha: str, benchmark: str, fold: int):
    if not path.is_file():
        raise RuntimeError(f"Missing selected checkpoint {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["protocol_sha256"] != protocol_sha or \
            state["benchmark"] != benchmark or state["fold"] != fold or \
            state["test_accessed"]:
        raise RuntimeError("Selected checkpoint provenance differs")
    return {"private_path": str(path), "sha256": digest(path),
            "selected_epoch": state["selected"]["epoch"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol_sha = digest(args.protocol)
    frozen = {"status": "FROZEN_BEFORE_OFFICIAL_TEST",
              "protocol_sha256": protocol_sha,
              "model_frozen_before_final_test": True,
              "final_test_accessed": False,
              "test_used_for_tuning": False,
              "historical_a1_0_746382_is_development_vloo_not_outer_test": True,
              "benchmark_checkpoints": {}}
    for benchmark, folds in (("ictal", range(1, 6)), ("omni", (1,))):
        for fold in folds:
            work = args.runtime / benchmark / f"fold{fold}"
            b0 = json.loads((work / "B0_RawCNN" / "RAW_B0_SELECTION.json").read_text(encoding="utf-8"))
            pc = json.loads((work / "PC_VALIDATION_SELECTION.json").read_text(encoding="utf-8"))
            if b0["status"] != "TRAIN_VALIDATION_COMPLETE" or \
                    pc["status"] != "TRAIN_VALIDATION_COMPLETE" or \
                    b0["protocol_sha256"] != protocol_sha or \
                    pc["protocol_sha256"] != protocol_sha or \
                    b0["test_accessed"] or pc["test_accessed"]:
                raise RuntimeError("Incomplete train-side checkpoint selection")
            selected = {}
            b0_best = work / "B0_RawCNN" / "selected_best.pt"
            pc_best = Path(pc["final"]["checkpoint"])
            if pc["final"]["stage"] not in ("B", "C"):
                raise RuntimeError("PC-CNN candidate must not silently become B0")
            for label, path, stage, selection in (
                ("RawCNN", b0_best, "B0", b0["selected"]),
                ("PC-CNN", pc_best, pc["final"]["stage"], pc["final"]["selected"]),
            ):
                item = checked_checkpoint(path, protocol_sha, benchmark, fold)
                if item["selected_epoch"] != selection["epoch"]:
                    raise RuntimeError("Selected epoch differs from checkpoint")
                if benchmark == "omni":
                    item["validation_threshold"] = validation_threshold(
                        selected_validation_file(work, selection, stage))
                else:
                    item["classification_threshold"] = 0.5
                    item["threshold_note"] = "fixed_0.5; ranking metrics are primary"
                if label == "PC-CNN":
                    norm = work / "descriptor_normalization.json"
                    item["train_fit_descriptor_norm_sha256"] = digest(norm)
                item["stage"] = stage
                selected[label] = item
            frozen["benchmark_checkpoints"][f"{benchmark}/fold{fold}"] = selected
    save_json(args.output, frozen)
    print(json.dumps({"status": frozen["status"],
                      "benchmark_fold_count": len(frozen["benchmark_checkpoints"]),
                      "model_count": sum(len(value) for value in frozen["benchmark_checkpoints"].values()),
                      "final_test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
