"""Append-only run registry; failures are recorded, never converted to success."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


FIELDS = ["experiment_type", "model", "seed", "outer_fold", "held_out_center", "status", "start_time", "end_time", "elapsed_seconds", "command", "config_path", "config_sha256", "cohort_sha256", "fold_manifest_sha256", "checkpoint_path", "checkpoint_sha256", "validation_ledger_path", "test_ledger_path", "log_path", "error_message"]


def append_registry(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.is_file()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
