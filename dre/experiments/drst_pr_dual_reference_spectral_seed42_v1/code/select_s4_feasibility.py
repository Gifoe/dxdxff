"""Freeze five FIT-meta-selected S4 checkpoints under amendment 02."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_s4_feasibility import AMENDMENT_SHA, LOCK_SHA, LR, VARIANT, WD
from stage0_source import file_sha, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--amendment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if file_sha(a.lock) != LOCK_SHA or file_sha(a.amendment) != AMENDMENT_SHA:
        raise RuntimeError("Feasibility protocol identity mismatch")
    status = json.loads((a.runtime / "S4_FEASIBILITY_STATUS.json").read_text(encoding="utf-8"))
    if not status["complete"] or status["completed_folds"] != 5 or status["amendment_sha256"] != AMENDMENT_SHA:
        raise RuntimeError("All five S4 source-only cells must finish first")
    rows = []
    for fold in range(1, 6):
        cell = a.runtime / VARIANT / f"fold_{fold}" / "lr_0.0003_wd_0.001"
        summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
        checkpoint = cell / "selected_best_private.pt"
        if (not summary["complete"] or summary["lock_sha256"] != LOCK_SHA or
                summary["fold"] != fold or summary["variant"] != VARIANT or
                summary["lr"] != LR or summary["weight_decay"] != WD or not checkpoint.exists()):
            raise RuntimeError("Selected S4 cell identity mismatch")
        rows.append({"variant": VARIANT, "fold": fold, "checkpoint": str(checkpoint),
                     "checkpoint_sha256": file_sha(checkpoint), "selected_epoch": summary["best_epoch"],
                     "fit_meta_patient_equal_ez_ap": summary["best_ap"], "epochs_run": summary["epochs_run"],
                     "learning_rate": LR, "weight_decay": WD,
                     "selection_scope": "FIT_ONLY", "target_labels_indexed": False})
    private = a.runtime / "S4_FEASIBILITY_SELECTION_PRIVATE.json"
    write_json(private, {"lock_sha256": LOCK_SHA, "amendment_sha256": AMENDMENT_SHA,
                         "selected_checkpoints": rows, "target_labels_indexed": False})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    public_rows = [{k: v for k, v in row.items() if k != "checkpoint"} for row in rows]
    with a.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    print("S4_FIT_CHECKPOINT_SELECTION_PASS 5/5", flush=True)


if __name__ == "__main__":
    main()
