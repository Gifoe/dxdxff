"""Select 20 spectral checkpoints strictly from FIT-meta AP, before target scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stage0_source import file_sha, write_json
from run_training_grid import VARIANTS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    lock_sha = file_sha(a.lock)
    status = json.loads((a.runtime / "TRAINING_GRID_STATUS.json").read_text(encoding="utf-8"))
    if not status["complete"] or status["completed_cells"] != 80 or status["lock_sha256"] != lock_sha:
        raise RuntimeError("Complete locked FIT-only grid required")
    rows = []
    selected_private = []
    for variant in VARIANTS:
        for fold in range(1, 6):
            candidates = []
            for lr in (1e-4, 3e-4):
                for wd in (1e-4, 1e-3):
                    folder = a.runtime / variant / f"fold_{fold}" / f"lr_{lr:g}_wd_{wd:g}"
                    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                    checkpoint = folder / "selected_best_private.pt"
                    if not summary["complete"] or summary["lock_sha256"] != lock_sha or not checkpoint.is_file():
                        raise RuntimeError("Missing locked FIT-only cell")
                    candidates.append((summary["best_ap"], lr, wd, summary["best_epoch"], checkpoint,
                                       summary["epochs_run"], summary["parameter_count"]))
            # Max AP, then lower LR, higher WD, earlier epoch: exactly the lock.
            winner = sorted(candidates, key=lambda row: (-row[0], row[1], -row[2], row[3]))[0]
            ap, lr, wd, epoch, checkpoint, epochs_run, parameters = winner
            selected_private.append({"variant": variant, "fold": fold, "checkpoint": str(checkpoint),
                                     "checkpoint_sha256": file_sha(checkpoint), "lock_sha256": lock_sha,
                                     "fit_meta_ap": ap, "selected_epoch": epoch,
                                     "lr": lr, "weight_decay": wd})
            rows.append({"variant": variant, "fold": fold, "selected_epoch": epoch,
                         "fit_meta_patient_equal_ez_ap": ap, "learning_rate": lr,
                         "weight_decay": wd, "epochs_run": epochs_run,
                         "trainable_parameters": parameters,
                         "checkpoint_sha256": file_sha(checkpoint),
                         "selection_scope": "FIT_ONLY", "target_labels_accessed": False})
    write_json(a.runtime / "FIT_SELECTION_PRIVATE.json", {"lock_sha256": lock_sha,
               "selection_complete_before_target_scoring": True, "rows": selected_private})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("FIT_SELECTION_PASS 20/20", flush=True)


if __name__ == "__main__":
    main()
