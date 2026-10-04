#!/usr/bin/env python3
"""Hash-lock all completed fold checkpoints before the sole OOF read."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from modeling import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_sha = sha256(args.protocol)
    entries = []
    initial_by_fold = {}
    for fold in range(1, 6):
        for variant in ("baseline", "plugin"):
            root = args.runtime / "checkpoints" / f"fold_{fold}" / variant
            complete = root / "TRAINING_COMPLETE.json"; checkpoint = root / "last.pt"
            if not complete.is_file() or not checkpoint.is_file():
                raise RuntimeError("Incomplete training checkpoint set")
            completed = json.loads(complete.read_text(encoding="utf-8"))
            if completed["checkpoint_sha256"] != sha256(checkpoint) or completed["epochs"] != 30:
                raise RuntimeError("Training completion hash/epoch mismatch")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["protocol_sha256"] != protocol_sha or state["fold"] != fold or state["variant"] != variant or state["epoch"] != 30:
                raise RuntimeError("Checkpoint protocol mismatch")
            initial = state["initial_raw_state_sha256"]
            prior = initial_by_fold.setdefault(fold, initial)
            if prior != initial:
                raise RuntimeError("Baseline/plugin raw initialization mismatch")
            entries.append({"fold": fold, "variant": variant, "checkpoint_sha256": sha256(checkpoint),
                            "initial_raw_state_sha256": initial})
    lock = {"status": "FROZEN_BEFORE_OOF", "protocol_sha256": protocol_sha,
            "models": entries, "test_or_oof_scores_read_before_lock": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
