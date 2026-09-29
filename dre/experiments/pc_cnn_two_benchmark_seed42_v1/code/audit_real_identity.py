"""Real TRAIN-sample RawCNN/PC-CNN identity; never opens validation/test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from official_spectrum import OfficialSpectrum, load_official_module
from patient_bank import IctalBank
from pc_cnn import PCCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--ictal-cache", type=Path, required=True)
    parser.add_argument("--ictal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load_official_module(args.official_cnn)
    bank = IctalBank(args.ictal_cache, args.ictal_manifest)
    fit_patient = bank.folds[1]["fit"][0]
    record = next(bank.records(fit_patient))
    selected = np.flatnonzero(record["channel_mask"])[:4]
    if len(selected) < 2:
        raise RuntimeError("Real fit record needs two channels for identity audit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = module.NeuralCNN(in_channels=1, outputs=1).to(device).eval()
    pc = PCCNN(raw).to(device).eval()
    waves = torch.from_numpy(record["waveforms"][selected][None]).to(device)
    desc = torch.from_numpy(record["descriptors"][selected][None]).to(device)
    available = torch.from_numpy(record["descriptor_mask"][selected][None].astype(np.float32)).to(device)
    valid = torch.ones(1, len(selected), dtype=torch.bool, device=device)
    preprocessor = OfficialSpectrum(module, channel_chunk=2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        base = pc.forward_record(waves, 250.0, desc, valid, available,
                                 preprocessor, physiology=False, context=False,
                                 channel_chunk=2)
        enhanced = pc.forward_record(waves, 250.0, desc, valid, available,
                                     preprocessor, channel_chunk=2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    difference = float(torch.max(torch.abs(base - enhanced)))
    mismatch = int(torch.sum((base > 0) != (enhanced > 0)))
    audit = {"status": "PASS" if difference < 1e-6 and mismatch == 0 else "FAIL",
             "benchmark": "Ictal", "source_role": "fit_only",
             "real_sample_channels": len(selected), "logit_max_abs_difference": difference,
             "prediction_mismatch": mismatch, "seconds_two_forward_passes": time.perf_counter() - start,
             "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
             "gate_effective_initial": {key: float(value.detach()) for key, value in pc.alphas().items()},
             "test_accessed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
