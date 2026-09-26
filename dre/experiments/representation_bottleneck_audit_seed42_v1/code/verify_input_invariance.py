"""Independently verify recruitment inputs are bitwise identical across A1 epochs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from reproduce_source import RUNTIME, ensure_source

EXPERIMENT = Path(__file__).resolve().parents[1]


def input_digest(fold: int, epoch: int) -> str:
    path = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}" / "representation.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for role in ("fit", "validation"):
        for row in payload[role]:
            digest.update(role.encode())
            digest.update(row["subject_id"].encode())
            for seizure in row["seizures"]:
                for name in ("x", "centers", "channel_mask", "window_mask"):
                    array = np.ascontiguousarray(seizure[name])
                    digest.update(str(array.shape).encode())
                    digest.update(str(array.dtype).encode())
                    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    ensure_source()
    folds = []
    for fold in range(1, 6):
        hashes = [input_digest(fold, epoch) for epoch in range(1, 31)]
        folds.append({"fold": fold, "epochs": 30, "all_FIT_validation_inputs_bitwise_identical": len(set(hashes)) == 1})
        print(f"[INPUT] fold={fold} identical={folds[-1]['all_FIT_validation_inputs_bitwise_identical']}", flush=True)
    passed = all(row["all_FIT_validation_inputs_bitwise_identical"] for row in folds)
    output = {"pass": passed, "folds": folds, "cells_checked": 150,
              "fields": ["normalized_four_view_x", "relative_window_centers", "channel_mask", "window_mask"],
              "recruitment_features_reusable_per_fold": passed, "outer_test_accessed": False}
    (EXPERIMENT / "INPUT_INVARIANCE_AUDIT.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Input epoch-invariance failed; fold-level recruitment feature cache is invalid")


if __name__ == "__main__":
    main()
