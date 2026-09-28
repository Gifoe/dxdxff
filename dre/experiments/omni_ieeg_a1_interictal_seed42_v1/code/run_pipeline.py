"""Resume-only runner for the frozen A1-Omni sequence on the original server.

It waits for the already-running train feature shards; it never starts or
replaces them. Official test feature construction is gated by the checkpoint
freeze. All large artifacts remain outside Git.
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


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
AUDIT = ROOT / "audit"
RUNTIME = Path("F:/Omni-iEEG/a1_interictal_seed42_runtime")
OUTPUT = ROOT / "outputs"
COHORT = AUDIT / "OMNI_COHORT_AUDIT.csv"
SPLIT = AUDIT / "TRAIN_VAL_SPLIT.csv"
PROTOCOL = CODE / "PROTOCOL_LOCK.json"
FREEZE = OUTPUT / "TEST_SCORE_FREEZE_AUDIT.json"
PYTHON = Path(sys.executable)


def update(phase: str, **fields):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    payload = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), **fields}
    target = RUNTIME / "PIPELINE_STATUS.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, target)
    print(payload, flush=True)


def feature_files(rows: pd.DataFrame, root: Path) -> list[Path]:
    return [root / Path(row.edf).with_suffix(".npz")
            for row in rows.itertuples(index=False) if row.valid_channels > 0]


def execute(name: str, args: list[str]):
    with (RUNTIME / f"{name}.log").open("a", encoding="utf-8") as stdout, \
            (RUNTIME / f"{name}.err").open("a", encoding="utf-8") as stderr:
        result = subprocess.run([str(PYTHON), *args], stdout=stdout, stderr=stderr,
                                cwd=CODE, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed: exit {result.returncode}")


def main():
    if not (COHORT.is_file() and SPLIT.is_file() and PROTOCOL.is_file()):
        raise RuntimeError("Frozen protocol or train-side audits absent")
    cohort = pd.read_csv(COHORT)
    train = cohort.loc[cohort["official_split"] == "train"]
    test = cohort.loc[cohort["official_split"] == "test"]
    if (train["valid_channels"] > 0).sum() != 149 or (test["valid_channels"] > 0).sum() != 102:
        raise RuntimeError("Supervised cohort counts differ from locked audit")
    train_root = RUNTIME / "features_train"
    train_files = feature_files(train, train_root)
    while True:
        done = sum(path.is_file() and path.with_suffix(".json").is_file() for path in train_files)
        update("WAIT_TRAIN_FEATURES", complete=int(done), required=len(train_files))
        if done == len(train_files):
            break
        time.sleep(120)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not FREEZE.exists():
        update("TRAINING_INNER_AND_FINAL")
        execute("train_a1", [str(CODE / "train_a1_omni.py"),
                             "--cohort", str(COHORT), "--train-val-split", str(SPLIT),
                             "--features", str(train_root), "--runtime", str(RUNTIME / "training"),
                             "--output", str(OUTPUT), "--protocol", str(PROTOCOL)])
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not freeze.get("model_frozen_before_official_test"):
        raise RuntimeError("Test score freeze not valid")
    test_root = RUNTIME / "features_test"
    test_files = feature_files(test, test_root)
    if not all(path.is_file() and path.with_suffix(".json").is_file() for path in test_files):
        update("BUILD_OFFICIAL_TEST_FEATURES", required=len(test_files))
        procs = []
        streams = []
        try:
            for shard in range(3):
                stdout = (RUNTIME / f"test_features_shard{shard}.log").open("a", encoding="utf-8")
                stderr = (RUNTIME / f"test_features_shard{shard}.err").open("a", encoding="utf-8")
                streams.extend([stdout, stderr])
                cmd = [str(PYTHON), str(CODE / "extract_60s_a1_features.py"),
                       "--split", "test", "--cohort", str(COHORT),
                       "--output", str(test_root), "--protocol", str(PROTOCOL),
                       "--test-freeze", str(FREEZE), "--num-shards", "3", "--shard-index", str(shard)]
                procs.append(subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=CODE))
            codes = [proc.wait() for proc in procs]
        finally:
            for stream in streams:
                stream.close()
        if any(codes):
            raise RuntimeError(f"Official test feature shard failures: {codes}")
    if not all(path.is_file() and path.with_suffix(".json").is_file() for path in test_files):
        raise RuntimeError("Official test feature coverage incomplete")
    arguments = ["--cohort", str(COHORT), "--features", str(test_root),
                 "--runtime", str(RUNTIME / "training"), "--output", str(OUTPUT),
                 "--protocol", str(PROTOCOL), "--freeze", str(FREEZE)]
    if not (OUTPUT / "CHANNEL_PREDICTIONS.csv").is_file():
        update("OFFICIAL_TEST_INFERENCE", required_patients=94)
        execute("official_test_inference", [str(CODE / "evaluate_official_test.py"),
                                            "--phase", "inference", *arguments])
    if not (OUTPUT / "PATIENT_CLUSTER_BOOTSTRAP.csv").is_file():
        update("OFFICIAL_TEST_SUMMARY")
        execute("official_test_summary", [str(CODE / "evaluate_official_test.py"),
                                          "--phase", "summary", *arguments])
    update("COMPLETE", output=str(OUTPUT))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        update("FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
