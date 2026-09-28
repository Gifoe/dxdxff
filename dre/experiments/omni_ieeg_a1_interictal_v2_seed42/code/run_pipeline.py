"""Resume-only v2 sequence. Train/freeze before any official-test read.

The four official-train feature workers are started separately and are never
replaced by this runner. All large artifacts stay in private server runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from train_v2 import sha256


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
AUDIT = ROOT / "audit"
RUNTIME = Path("F:/Omni-iEEG/a1_interictal_v2_seed42_runtime")
OUTPUT = ROOT / "outputs"
COHORT = AUDIT / "OFFICIAL_COHORT_AUDIT.csv"
SPLIT = AUDIT / "TRAIN_VAL_SPLIT.csv"
PROTOCOL = CODE / "PROTOCOL_LOCK.json"
FREEZE = OUTPUT / "TEST_SCORE_FREEZE_AUDIT.json"
THRESHOLD = OUTPUT / "FROZEN_THRESHOLD.json"
PYTHON = Path(sys.executable)


def update(phase: str, **fields):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    value = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), **fields}
    target = RUNTIME / "PIPELINE_STATUS.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, target)
    print(value, flush=True)


def feature_files(rows: pd.DataFrame, root: Path) -> list[Path]:
    return [root / Path(row.edf).with_suffix(".npz")
            for row in rows.itertuples(index=False) if row.official_labeled_channels > 0]


def complete(files: list[Path], protocol_sha: str) -> int:
    done = 0
    for path in files:
        marker = path.with_suffix(".json")
        if not path.is_file() or not marker.is_file():
            continue
        details = json.loads(marker.read_text(encoding="utf-8"))
        if details["protocol_sha256"] != protocol_sha or path.stat().st_size != details["bytes"]:
            raise RuntimeError(f"Feature marker/protocol mismatch: {path}")
        done += 1
    return done


def execute(name: str, arguments: list[str]):
    with (RUNTIME / f"{name}.log").open("a", encoding="utf-8") as stdout, \
            (RUNTIME / f"{name}.err").open("a", encoding="utf-8") as stderr:
        result = subprocess.run([str(PYTHON), *arguments], stdout=stdout, stderr=stderr,
                                cwd=CODE, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed: exit {result.returncode}; inspect private logs")


def main():
    if not (COHORT.is_file() and SPLIT.is_file() and PROTOCOL.is_file()):
        raise RuntimeError("Frozen v2 train-side audit/protocol absent")
    protocol_sha = sha256(PROTOCOL)
    cohort = pd.read_csv(COHORT)
    train = cohort.loc[cohort["official_split"] == "train"]
    test = cohort.loc[cohort["official_split"] == "test"]
    if (train["official_labeled_channels"] > 0).sum() != 296 or \
            (test["official_labeled_channels"] > 0).sum() != 174 or \
            train.loc[train["official_labeled_channels"] > 0, "patient"].nunique() != 141:
        raise RuntimeError("Official v2 cohort counts differ from frozen metadata audit")
    train_root = RUNTIME / "features_train"
    train_files = feature_files(train, train_root)
    while True:
        done = complete(train_files, protocol_sha)
        update("WAIT_TRAIN_FEATURES", complete=done, required=len(train_files))
        if done == len(train_files):
            break
        time.sleep(120)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not FREEZE.exists():
        update("TRAINING_AP_SELECTION_THRESHOLD_AND_REFIT")
        execute("train_v2", [str(CODE / "train_v2.py"),
                             "--cohort", str(COHORT), "--train-val-split", str(SPLIT),
                             "--features", str(train_root),
                             "--runtime", str(RUNTIME / "training"),
                             "--output", str(OUTPUT), "--protocol", str(PROTOCOL)])
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not frozen.get("model_frozen_before_official_test") or \
            not frozen.get("threshold_frozen_before_official_test") or \
            frozen["protocol_sha256"] != protocol_sha or \
            frozen["threshold_json_sha256"] != sha256(THRESHOLD):
        raise RuntimeError("Frozen v2 checkpoint/threshold invalid")
    if not (OUTPUT / "V1_OFFICIAL_COHORT_RESCORE.json").exists():
        update("D1_V1_OFFICIAL_LABEL_OVERLAP_RESCORE")
        v1_predictions = Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v1/outputs/CHANNEL_PREDICTIONS.csv")
        execute("d1_v1_rescore", [str(CODE / "rescore_v1.py"),
                                  "--cohort", str(COHORT),
                                  "--v1-predictions", str(v1_predictions),
                                  "--source", "F:/Omni-iEEG/data",
                                  "--cache", "F:/Omni-iEEG/signal_cache",
                                  "--freeze", str(FREEZE),
                                  "--output", str(OUTPUT / "V1_OFFICIAL_COHORT_RESCORE.json")])
    test_root = RUNTIME / "features_test"
    test_files = feature_files(test, test_root)
    if complete(test_files, protocol_sha) < len(test_files):
        update("BUILD_OFFICIAL_TEST_FEATURES", complete=complete(test_files, protocol_sha),
               required=len(test_files))
        procs, streams = [], []
        try:
            for shard in range(4):
                stdout = (RUNTIME / f"test_features_shard{shard}.log").open("a", encoding="utf-8")
                stderr = (RUNTIME / f"test_features_shard{shard}.err").open("a", encoding="utf-8")
                streams.extend([stdout, stderr])
                cmd = [str(PYTHON), str(CODE / "extract_cross_channel_features.py"),
                       "--split", "test", "--cohort", str(COHORT),
                       "--output", str(test_root), "--protocol", str(PROTOCOL),
                       "--test-freeze", str(FREEZE),
                       "--num-shards", "4", "--shard-index", str(shard)]
                procs.append(subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=CODE))
            codes = [proc.wait() for proc in procs]
        finally:
            for stream in streams:
                stream.close()
        if any(codes):
            raise RuntimeError(f"Official test feature shard failure: {codes}")
    if complete(test_files, protocol_sha) != len(test_files):
        raise RuntimeError("Official test feature cache incomplete")
    shared = ["--cohort", str(COHORT), "--features", str(test_root),
              "--runtime", str(RUNTIME / "training"), "--output", str(OUTPUT),
              "--protocol", str(PROTOCOL), "--freeze", str(FREEZE),
              "--threshold", str(THRESHOLD)]
    if not (OUTPUT / "CHANNEL_PREDICTIONS.csv").exists():
        update("OFFICIAL_TEST_INFERENCE", required_patients=96)
        execute("official_test_inference", [str(CODE / "evaluate_v2.py"),
                                            "--phase", "inference", *shared])
    if not (OUTPUT / "PATIENT_CLUSTER_BOOTSTRAP.csv").exists():
        update("OFFICIAL_TEST_SUMMARY")
        execute("official_test_summary", [str(CODE / "evaluate_v2.py"),
                                          "--phase", "summary", *shared])
    update("COMPLETE", output=str(OUTPUT))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        update("FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
