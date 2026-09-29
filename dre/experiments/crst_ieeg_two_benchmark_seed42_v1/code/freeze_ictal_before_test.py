"""Freeze all ten fold/model checkpoints before accessing ictal outer test."""

import argparse
import json
from pathlib import Path

import torch

from train_crst import json_atomic, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol_sha, train_sha = sha(args.protocol), sha(args.training_lock)
    models = {}
    for fold in range(1, 6):
        for variant in ("CRST-0", "CRST-FULL"):
            key = f"fold{fold}/{variant}"
            work = args.runtime / "ictal" / f"fold{fold}" / variant
            complete = work / "supervised_complete.json"
            path = work / "supervised_best.pt"
            if not complete.is_file() or not path.is_file():
                raise RuntimeError(f"Ictal validation selection incomplete: {key}")
            marker = json.loads(complete.read_text(encoding="utf-8"))
            state = torch.load(path, map_location="cpu", weights_only=False)
            if (marker["best_checkpoint_sha256"] != sha(path) or
                    state["protocol_sha"] != protocol_sha or
                    state["train_sha"] != train_sha or
                    state["fold"] != fold or state["variant"] != variant or
                    marker["selected_epoch"] != state["epoch"] or
                    marker["test_accessed"]):
                raise RuntimeError(f"Ictal selected checkpoint mismatch: {key}")
            models[key] = {"checkpoint_path": str(path.resolve()),
                           "checkpoint_sha256": sha(path),
                           "selected_epoch": state["epoch"],
                           "frozen_threshold": state["threshold"]}
    frozen = {"status": "FROZEN_BEFORE_ICTAL_OUTER_TEST", "seed": 42,
              "protocol_sha256": protocol_sha, "training_lock_sha256": train_sha,
              "models": models, "outer_test_accessed_before_freeze": False,
              "test_used_for_checkpoint_or_threshold_selection": False}
    if args.output.is_file():
        if json.loads(args.output.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Existing ictal model freeze cannot be replaced")
        print(json.dumps({"status": "ICTAL_FREEZE_REUSED", "sha256": sha(args.output)}))
        return
    json_atomic(args.output, frozen)
    print(json.dumps({"status": "FROZEN_BEFORE_ICTAL_OUTER_TEST",
                      "sha256": sha(args.output), "checkpoints": len(models)}))


if __name__ == "__main__":
    main()
