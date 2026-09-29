"""Sequential gate: train extraction -> CNN freeze -> test extraction -> score.

The standalone training extractor shards are started separately to permit
explicit monitoring; this watcher does not start duplicate train jobs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("E:/DRE-nips/new-pipeline/7-11/a1_net_relational_router_two_benchmark_seed42_v1")
RUN = Path("F:/Omni-iEEG/a1_net_seed42_runtime")
SPLIT = Path("F:/Omni-iEEG/signal_cache/derivatives/datasplit/final_split.csv")
SOURCE = Path("F:/Omni-iEEG/data")
CACHE = Path("F:/Omni-iEEG/signal_cache")
CODE = ROOT / "code"
LOCK = ROOT / "PROTOCOL_LOCK.json"
TRAIN = RUN / "features_train_full"
TEST = RUN / "features_test_full"
OFFICIAL = RUN / "official_source/cnn.py"
TRAINING = RUN / "official_cnn_training"
STATUS = RUN / "GATE_STATUS.json"


def status(stage: str, **kwargs):
    payload = {"stage": stage, "utc_epoch": time.time(), **kwargs}
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATUS)
    print(json.dumps(payload), flush=True)


def run(name: str, args: list[str]):
    with (RUN / f"{name}.log").open("a", encoding="utf-8") as out, \
         (RUN / f"{name}.err").open("a", encoding="utf-8") as err:
        code = subprocess.call([sys.executable, *args], stdout=out, stderr=err)
    if code != 0:
        raise RuntimeError(f"{name} exit code {code}; inspect private log/err")


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    protocol_sha = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    criterion = json.loads((ROOT / "GATE_CRITERION.json").read_text(encoding="utf-8"))
    criterion_sha = hashlib.sha256((ROOT / "GATE_CRITERION.json").read_bytes()).hexdigest()
    status("WAITING_TRAIN_EXTRACTION", protocol_sha256=protocol_sha,
           gate_criterion_sha256=criterion_sha)
    shard_files = [TRAIN / f"EXTRACTION_TRAIN_SHARD{i}OF4.json" for i in range(4)]
    while not all(file.is_file() for file in shard_files):
        time.sleep(60)
    status("VERIFYING_TRAIN_EXTRACTION")
    run("finalize_train", [str(CODE / "finalize_waveform_extraction.py"),
                           "--split", "train", "--official-split", str(SPLIT),
                           "--output", str(TRAIN), "--protocol", str(LOCK),
                           "--num-shards", "4"])
    status("TRAINING_OFFICIAL_CNN")
    run("train_official_cnn", [str(CODE / "train_official_cnn.py"),
                               "--official-cnn", str(OFFICIAL),
                               "--features", str(TRAIN), "--protocol", str(LOCK),
                               "--runtime", str(TRAINING)])
    summary = json.loads((TRAINING / "TRAINING_AUDIT.json").read_text())
    if not summary["completed"] or summary["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Official CNN training freeze incomplete")
    status("EXTRACTING_OFFICIAL_TEST", checkpoint_sha256=summary["checkpoint_sha256"])
    processes = []
    handles = []
    try:
        for i in range(4):
            out = (RUN / f"extract_test_shard{i}.log").open("a", encoding="utf-8")
            err = (RUN / f"extract_test_shard{i}.err").open("a", encoding="utf-8")
            handles.extend([out, err])
            process = subprocess.Popen([sys.executable,
                str(CODE / "prepare_official_cnn_waveforms.py"),
                "--split", "test", "--official-split", str(SPLIT),
                "--source", str(SOURCE), "--cache", str(CACHE),
                "--output", str(TEST), "--protocol", str(LOCK),
                "--num-shards", "4", "--shard-index", str(i)], stdout=out, stderr=err)
            processes.append(process)
        codes = [process.wait() for process in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"Test extraction shard exits: {codes}")
    finally:
        for handle in handles:
            handle.close()
    status("VERIFYING_TEST_EXTRACTION")
    run("finalize_test", [str(CODE / "finalize_waveform_extraction.py"),
                          "--split", "test", "--official-split", str(SPLIT),
                          "--output", str(TEST), "--protocol", str(LOCK),
                          "--num-shards", "4"])
    status("EVALUATING_FROZEN_OFFICIAL_CNN")
    result_file = RUN / "OMNI_CNN_REPRODUCTION_RESULTS.json"
    run("evaluate_official_cnn", [str(CODE / "evaluate_official_cnn.py"),
                                   "--official-cnn", str(OFFICIAL),
                                   "--features", str(TEST), "--protocol", str(LOCK),
                                   "--training", str(TRAINING),
                                   "--output", str(result_file)])
    result = json.loads(result_file.read_text())
    gates = criterion["pass_if_both"]
    passed = (
        result["reproduced_macro_f1_official_test_youden"] >=
        gates["macro_f1_official_test_youden_at_least"] and
        result["reproduced_channel_auroc"] >= gates["channel_auroc_at_least"])
    terminal = "RAW_ENCODER_REPRODUCED" if passed else "RAW_ENCODER_REPRODUCTION_FAILED"
    status(terminal, passed=passed, criterion_sha256=criterion_sha,
           result_sha256=hashlib.sha256(result_file.read_bytes()).hexdigest(),
           result={"macro_f1": result["reproduced_macro_f1_official_test_youden"],
                   "auroc": result["reproduced_channel_auroc"]})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("ENGINEERING_FAILURE", error_type=type(exc).__name__, error=str(exc))
        raise
