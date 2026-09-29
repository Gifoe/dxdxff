"""Write and validate the immutable pre-test Omni model/threshold identity."""

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
    audit = json.loads((args.runtime / "OMNI_TRAIN_CACHE_AUDIT.json").read_text())
    if not audit["pass"] or audit["test_split_accessed"]:
        raise RuntimeError("Train cache audit not clean")
    protocol_sha, train_sha = sha(args.protocol), sha(args.training_lock)
    models = {}
    for variant in ("CRST-0", "CRST-FULL"):
        path = args.runtime / "omni" / "final_refit" / variant / "frozen_model.pt"
        complete = args.runtime / "omni" / "final_refit" / variant / "refit_complete.json"
        if not path.is_file() or not complete.is_file():
            raise RuntimeError(f"{variant} final refit not complete")
        marker = json.loads(complete.read_text(encoding="utf-8"))
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (sha(path) != marker["checkpoint_sha256"] or
                state["protocol_sha"] != protocol_sha or
                state["train_sha"] != train_sha or
                state["threshold"] != marker["frozen_threshold"] or
                state["epoch"] != marker["selected_epoch"] or
                state["test_accessed_during_refit"]):
            raise RuntimeError(f"{variant} freeze mismatch")
        models[variant] = {"checkpoint_path": str(path.resolve()),
                           "checkpoint_sha256": sha(path),
                           "selection_checkpoint_sha256": state["selection_sha"],
                           "selected_epoch": state["epoch"],
                           "threshold": state["threshold"],
                           "ssl_checkpoint_sha256": state["ssl_sha"]}
    frozen = {"status": "FROZEN_BEFORE_OFFICIAL_TEST", "seed": 42,
              "architecture": "CRSTiEEG", "protocol_sha256": protocol_sha,
              "training_lock_sha256": train_sha,
              "train_cache_audit_sha256": sha(args.runtime / "OMNI_TRAIN_CACHE_AUDIT.json"),
              "models": models, "official_test_accessed_before_freeze": False,
              "test_used_for_selection": False}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous != frozen:
            raise RuntimeError("An existing model freeze cannot be replaced")
        print(json.dumps({"status": "FROZEN_REUSED", "sha256": sha(args.output)}))
        return
    json_atomic(args.output, frozen)
    print(json.dumps({"status": "FROZEN_BEFORE_OFFICIAL_TEST",
                      "sha256": sha(args.output), "models": list(models)}))


if __name__ == "__main__":
    main()
