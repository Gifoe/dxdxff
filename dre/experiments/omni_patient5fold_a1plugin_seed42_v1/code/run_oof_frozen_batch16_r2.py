#!/usr/bin/env python3
"""Run exactly one frozen OOF evaluation with hash-verified patient resume."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from modeling import sha256


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    known, remainder = parser.parse_known_args()

    frozen = json.loads(known.freeze.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_OOF":
        raise RuntimeError("OOF freeze is not finalized")
    if frozen.get("protocol_sha256") != sha256(known.protocol):
        raise RuntimeError("OOF freeze protocol mismatch")
    expected = {(item["fold"], item["variant"]): item["checkpoint_sha256"] for item in frozen["models"]}
    if len(expected) != 10:
        raise RuntimeError("OOF freeze does not contain ten unique models")
    for (fold, variant), expected_hash in expected.items():
        checkpoint = known.runtime / "checkpoints" / f"fold_{fold}" / variant / "last.pt"
        if not checkpoint.is_file() or sha256(checkpoint) != expected_hash:
            raise RuntimeError(f"Checkpoint changed after OOF freeze: fold={fold}, variant={variant}")

    # The evaluator refuses unbound caches.  Each cache is tied to this freeze,
    # protocol and model hash, so a native-process restart cannot mix results.
    os.environ["OMNI_OOF_FREEZE_SHA"] = sha256(known.freeze)
    import evaluate_oof_r2 as evaluate_oof
    from train_fold_r3 import patient_forward

    evaluate_oof.patient_forward = patient_forward
    sys.argv = [sys.argv[0], "--runtime", str(known.runtime), "--protocol", str(known.protocol), *remainder]
    evaluate_oof.main()


if __name__ == "__main__":
    main()
