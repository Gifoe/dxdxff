"""Run matched D0+ first, then all prespecified FIT-only KD candidates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from student_core import LOCK_SHA, VARIANTS, atomic_json


def selection(runtime: Path, variant: str, fold: int) -> dict:
    path = runtime / "private" / variant / f"fold_{fold}" / "FIT_SELECTION.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["lock_sha"] != LOCK_SHA or value["variant"] != variant or value["fold"] != fold:
        raise RuntimeError("FIT selection identity mismatch")
    return value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    primary = VARIANTS[:5]
    # Complete all five matched hard-label controls before any KD Student.
    for variant in primary:
        for fold in range(1, 6):
            path = a.runtime / "private" / variant / f"fold_{fold}" / "FIT_SELECTION.json"
            if path.is_file():
                selection(a.runtime, variant, fold)
                print(f"[SKIP] {variant} fold={fold}", flush=True)
                continue
            command = [sys.executable, str(Path(__file__).with_name("train_student.py")),
                       "--variant", variant, "--fold", str(fold), "--runtime", str(a.runtime)]
            stdout = a.runtime / f"{variant}_fold{fold}.log"
            stderr = a.runtime / f"{variant}_fold{fold}.err"
            print(f"[START] {variant} fold={fold}", flush=True)
            with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
                result = subprocess.run(command, stdout=out, stderr=err, check=False)
            if result.returncode != 0 or not path.is_file():
                raise RuntimeError(f"FIT Student cell failed: {variant} fold={fold} exit={result.returncode}")
            selection(a.runtime, variant, fold)
            print(f"[DONE] {variant} fold={fold}", flush=True)
    control = [selection(a.runtime, VARIANTS[0], fold) for fold in range(1, 6)]
    d0_mean = sum(row["selected"]["best_meta"]["ap"] for row in control) / 5
    pair_mean = sum(selection(a.runtime, VARIANTS[1], fold)["selected"]["best_meta"]["ap"]
                    for fold in range(1, 6)) / 5
    list_mean = sum(selection(a.runtime, VARIANTS[2], fold)["selected"]["best_meta"]["ap"]
                    for fold in range(1, 6)) / 5
    d4_eligible = max(pair_mean, list_mean) > d0_mean
    gate = {"lock_sha": LOCK_SHA, "fit_meta_D0_mean_ap": d0_mean,
            "fit_meta_D1_mean_ap": pair_mean, "fit_meta_D2_mean_ap": list_mean,
            "D4_FIT_ELIGIBLE": d4_eligible, "target_outcomes_accessed": False}
    atomic_json(a.output / "D4_FIT_ELIGIBILITY.json", gate)
    if d4_eligible:
        variant = VARIANTS[5]
        for fold in range(1, 6):
            path = a.runtime / "private" / variant / f"fold_{fold}" / "FIT_SELECTION.json"
            if path.is_file():
                selection(a.runtime, variant, fold)
                continue
            command = [sys.executable, str(Path(__file__).with_name("train_student.py")),
                       "--variant", variant, "--fold", str(fold), "--runtime", str(a.runtime)]
            stdout = a.runtime / f"{variant}_fold{fold}.log"
            stderr = a.runtime / f"{variant}_fold{fold}.err"
            print(f"[START] {variant} fold={fold}", flush=True)
            with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
                result = subprocess.run(command, stdout=out, stderr=err, check=False)
            if result.returncode != 0 or not path.is_file():
                raise RuntimeError(f"FIT D4 cell failed: fold={fold} exit={result.returncode}")
            selection(a.runtime, variant, fold)
            print(f"[DONE] {variant} fold={fold}", flush=True)
    variants = list(primary) + ([VARIANTS[5]] if d4_eligible else [])
    rows = []
    for variant in variants:
        for fold in range(1, 6):
            chosen = selection(a.runtime, variant, fold)["selected"]
            rows.append({"variant": variant, "fold": fold,
                         "tau": chosen["tau"], "lambda": chosen["lambda"],
                         "selected_epoch": chosen["best_epoch"],
                         "fit_meta_ap": chosen["best_meta"]["ap"],
                         "fit_meta_mrr": chosen["best_meta"]["mrr"],
                         "fit_meta_top1": chosen["best_meta"]["top1"],
                         "student_train_only": True, "target_selection_used": False})
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "FIT_KD_HYPERPARAM_SELECTION.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (a.output / "D0_CONTINUED_CONTROL.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([r for r in rows if r["variant"] == VARIANTS[0]])
    atomic_json(a.output / "FIT_SELECTION_FREEZE_AUDIT.json",
                {"lock_sha": LOCK_SHA, "n_variants": len(variants), "n_fold_models": len(rows),
                 "all_models_fit_selected_before_target_outcomes": True,
                 "outer_test_accessed": False, "D4_FIT_ELIGIBLE": d4_eligible})
    print(f"FIT_ALL_COMPLETE variants={len(variants)} models={len(rows)} D4={d4_eligible}", flush=True)


if __name__ == "__main__":
    main()
