#!/usr/bin/env python3
"""Emit a non-identifying five-fold training-completion audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_sha = __import__("hashlib").sha256(args.protocol.read_bytes()).hexdigest()
    rows_by_variant: dict[str, list[dict]] = {"baseline": [], "plugin": []}
    for fold in range(1, 6):
        for variant in rows_by_variant:
            record = json.loads((args.runtime / "checkpoints" / f"fold_{fold}" / variant / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
            if record.get("status") != "COMPLETE" or record.get("fold") != fold or record.get("variant") != variant:
                raise RuntimeError("incomplete or mismatched training record")
            if record.get("epochs") != protocol["epochs"] or record.get("protocol_sha256") != protocol_sha:
                raise RuntimeError("training record differs from protocol lock")
            if record.get("test_labels_or_predictions_accessed") is not False:
                raise RuntimeError("test access declared during optimization")
            rows_by_variant[variant].append({
                "fold": fold, "variant": variant, "status": "COMPLETE", "epochs": record["epochs"],
                "test_labels_or_predictions_accessed": False,
                "initial_raw_state_sha256": record["initial_raw_state_sha256"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "effective_plugin_gate": record["final_effective_plugin_gate"],
            })
    atomic_csv(args.output / "BASELINE_TRAINING_AUDIT.csv", rows_by_variant["baseline"])
    atomic_csv(args.output / "PLUGIN_TRAINING_AUDIT.csv", rows_by_variant["plugin"])


if __name__ == "__main__":
    main()
