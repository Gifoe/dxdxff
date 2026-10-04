#!/usr/bin/env python3
"""Verify the OOF freeze then dispatch the only patient-level OOF evaluation."""
from __future__ import annotations

import argparse
import json
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
    if frozen["status"] != "FROZEN_BEFORE_OOF" or frozen["protocol_sha256"] != sha256(known.protocol):
        raise RuntimeError("OOF freeze protocol mismatch")
    expected = {(item["fold"], item["variant"]): item["checkpoint_sha256"] for item in frozen["models"]}
    if len(expected) != 10:
        raise RuntimeError("OOF freeze does not contain ten unique models")
    for (fold, variant), expected_hash in expected.items():
        actual = known.runtime / "checkpoints" / f"fold_{fold}" / variant / "last.pt"
        if not actual.is_file() or sha256(actual) != expected_hash:
            raise RuntimeError("Checkpoint changed after OOF freeze")
    # Import after the immutable checks.  This changes only evaluation batching:
    # test-time BatchNorm uses frozen running statistics.
    import evaluate_oof
    from train_fold_r3 import patient_forward
    evaluate_oof.patient_forward = patient_forward
    sys.argv = [sys.argv[0], "--runtime", str(known.runtime), "--protocol", str(known.protocol), *remainder]
    evaluate_oof.main()


if __name__ == "__main__":
    main()
