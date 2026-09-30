"""Outcome-free CPU inference throughput measurement on one real patient."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import psutil
import torch

from run_scm import SCMEZ, load_scaler, prepare_patient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        fit = [row["subject_id"] for row in csv.DictReader(stream)
               if int(row["outer_fold"]) == 1 and row["split_role"] == "fit"]
    scaler = load_scaler(args.runtime / "folds/fold1/FIT_SCALER_PRIVATE.npz")
    candidates = [(prepare_patient(args.cache, patient, scaler, fold=1), patient) for patient in fit]
    item, _ = max(candidates, key=lambda pair: pair[0]["n_channels"])
    state = torch.load(args.runtime / "folds/fold1/checkpoint_epoch_01.pt",
                       map_location="cpu", weights_only=False)
    model = SCMEZ(); model.load_state_dict(state["model"], strict=True); model.eval()
    with torch.inference_mode():
        for _ in range(10): model(item["matrices"], item["channel_index"], item["n_channels"])
        started = time.perf_counter()
        for _ in range(100): model(item["matrices"], item["channel_index"], item["n_channels"])
        elapsed = time.perf_counter() - started
    value = {"device": "CPU", "gpu_memory_bytes": 0, "repetitions": 100,
             "real_patient_channels": item["n_channels"],
             "real_patient_record_channel_matrices": int(len(item["matrices"])),
             "elapsed_seconds": elapsed, "patient_passes_per_second": 100.0 / elapsed,
             "channel_scores_per_second": 100.0 * item["n_channels"] / elapsed,
             "process_rss_bytes_after_loaded_benchmark": psutil.Process().memory_info().rss,
             "peak_process_rss_bytes": None, "outer_test_accessed": False}
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2), flush=True)


if __name__ == "__main__":
    main()
