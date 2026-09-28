"""Five-fold, single-configuration S4 feasibility; no target outcomes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from stage0_source import file_sha, write_json


LOCK_SHA = "d89fd83a847fc1d217a4d5556862b5660b9f45dbcab7e29a8bd1b08fd53fa1eb"
AMENDMENT_SHA = "7cd2ec5d054baf853a3a72ce5dc79ef456b4dd0d1047ff4d871650e02a506345"
VARIANT = "S4_DRST_PR"
LR = 3e-4
WD = 1e-3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--amendment", type=Path, required=True)
    a = p.parse_args()
    if file_sha(a.lock) != LOCK_SHA or file_sha(a.amendment) != AMENDMENT_SHA:
        raise RuntimeError("S4 feasibility lock/amendment mismatch")
    amendment = json.loads(a.amendment.read_text(encoding="utf-8"))
    if amendment["first_round_train"]["variant"] != VARIANT or amendment["first_round_train"]["learning_rate"] != LR or amendment["first_round_train"]["weight_decay"] != WD:
        raise RuntimeError("Fixed feasibility configuration mismatch")
    sanity = a.runtime / VARIANT / "fold_1" / "lr_0.0001_wd_0.0001" / "overfit_sanity" / "OVERFIT_SANITY_AUDIT.json"
    if not sanity.exists() or not json.loads(sanity.read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("Mandatory source-only S4 overfit check missing")
    for fold in range(1, 6):
        moments = json.loads((a.runtime / f"fold_{fold}" / "FIT_GLOBAL_SPECTRAL_NORMALIZER.json").read_text(encoding="utf-8"))
        if not moments["fit_only"] or moments["lock_sha256"] != LOCK_SHA:
            raise RuntimeError(f"FIT-only moments missing fold={fold}")
    started = time.time()
    for fold in range(1, 6):
        cell = a.runtime / VARIANT / f"fold_{fold}" / "lr_0.0003_wd_0.001"
        cell.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).with_name("train_cell.py")),
                   "--fold", str(fold), "--variant", VARIANT, "--lr", str(LR), "--wd", str(WD),
                   "--raw-cache", str(a.raw_cache), "--runtime", str(a.runtime), "--lock", str(a.lock)]
        for attempt in range(4):
            with (cell / "train.log").open("a", encoding="utf-8") as out, (cell / "train.err").open("a", encoding="utf-8") as err:
                status = subprocess.run(command, stdout=out, stderr=err, env=os.environ.copy(), check=False).returncode
            if status != 3221225477:
                break
            if not (cell / "resume_private.pt").is_file() or attempt == 3:
                break
            print(f"ACCESS_VIOLATION_RESUME fold={fold} attempt={attempt+1}", flush=True)
        if status != 0:
            write_json(a.runtime / "S4_FEASIBILITY_STATUS.json", {"complete": False, "completed_folds": fold-1,
                       "failed_fold": fold, "exit_code": status, "lock_sha256": LOCK_SHA,
                       "amendment_sha256": AMENDMENT_SHA, "target_outcomes_accessed": False})
            raise RuntimeError(f"S4 feasibility fold {fold} failed with exit={status}")
        summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
        if (not summary["complete"] or summary["variant"] != VARIANT or
                summary["fold"] != fold or summary["lr"] != LR or summary["weight_decay"] != WD):
            raise RuntimeError("Completed fold identity mismatch")
        write_json(a.runtime / "S4_FEASIBILITY_STATUS.json", {"complete": fold == 5, "completed_folds": fold,
                   "expected_folds": 5, "last_fold": fold, "elapsed_seconds": time.time()-started,
                   "lock_sha256": LOCK_SHA, "amendment_sha256": AMENDMENT_SHA,
                   "target_outcomes_accessed": False})
        print(f"S4_FEASIBILITY {fold}/5 epoch={summary['best_epoch']} FIT_META_AP={summary['best_ap']:.6f}", flush=True)


if __name__ == "__main__":
    main()
