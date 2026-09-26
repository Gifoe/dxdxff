"""Independently verify recruitment inputs are bitwise identical across A1 epochs."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from reproduce_source import RUNTIME, ensure_source

EXPERIMENT = Path(__file__).resolve().parents[1]


def input_digest(fold: int, epoch: int) -> str:
    path = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}" / "representation.pt"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Unreadable representation cache: fold={fold}, epoch={epoch}, path={path}") from exc
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-fold", type=int, default=1)
    options = parser.parse_args()
    ensure_source()
    folds = []
    unreadable = []
    for fold in range(options.start_fold, 6):
        hashes = []
        for epoch in range(1, 31):
            try:
                hashes.append(input_digest(fold, epoch))
            except Exception as exc:
                unreadable.append({"fold": fold, "epoch": epoch})
                print(f"[UNREADABLE] fold={fold} epoch={epoch}: {exc}", flush=True)
        folds.append({"fold": fold, "epochs": len(hashes), "all_FIT_validation_inputs_bitwise_identical": len(hashes) == 30 and len(set(hashes)) == 1})
        print(f"[INPUT] fold={fold} identical={folds[-1]['all_FIT_validation_inputs_bitwise_identical']}", flush=True)
    if unreadable:
        raise RuntimeError(f"Unreadable frozen representation cells: {unreadable}")
    passed = len(folds) == 5 and all(row["all_FIT_validation_inputs_bitwise_identical"] for row in folds)
    output = {"pass": passed, "folds": folds, "cells_checked": 150,
              "fields": ["normalized_four_view_x", "relative_window_centers", "channel_mask", "window_mask"],
              "recruitment_features_reusable_per_fold": passed, "outer_test_accessed": False}
    (EXPERIMENT / "INPUT_INVARIANCE_AUDIT.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Input epoch-invariance failed; fold-level recruitment feature cache is invalid")


if __name__ == "__main__":
    main()
