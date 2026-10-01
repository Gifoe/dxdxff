from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from audit_core import atomic_json
from extract_train_full import model_and_preprocessor, physical_signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--train-waveforms", type=Path, required=True)
    parser.add_argument("--train5", type=Path, required=True)
    parser.add_argument("--signal-cache", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    source_path = sorted(args.train_waveforms.glob("*.npz"))[0]
    with np.load(source_path, allow_pickle=False) as z:
        edf, channel = str(z["edf"]), str(z["channel_names"][0])
        channel_names = z["channel_names"].astype(str)
        starts = np.asarray(z["starts"], dtype=np.int64)
        cached_waves = np.asarray(z["waveforms"][:, 0, :], dtype=np.float32)
        historical_batch = np.asarray(z["waveforms"], dtype=np.float32).reshape(-1, 60000)[:8]
    h5_path = args.signal_cache / Path(edf).with_suffix(".edf.h5")
    with h5py.File(h5_path, "r") as h5:
        metadata = json.loads(h5["metadata_json"][()].decode("utf-8"))
        headers = metadata["signal_headers"]
        index = {str(header["label"]): i for i, header in enumerate(headers)}
        reconstructed = physical_signal(h5, index[channel], headers[index[channel]])
    reconstructed_waves = np.stack([reconstructed[int(s):int(s) + 60000] for s in starts])
    waveform_error = float(np.max(np.abs(reconstructed_waves - cached_waves)))
    expected = None
    for path in args.train5.glob("*.npz"):
        with np.load(path, allow_pickle=False) as z:
            if str(z["edf"]) != edf: continue
            offsets = np.asarray(z["segment_offsets"], dtype=np.int64)
            stored_names = list(z["channel_names"].astype(str))
            flat = np.asarray(z["segment_logits"], dtype=np.float64)
            # Historical extraction flattened [clip, channel] before inference
            # and later regrouped logits by channel. The first inference batch
            # is therefore clip 0 for channels 0..7.
            expected = np.asarray([flat[offsets[stored_names.index(name)]] for name in channel_names[:8]])
            break
    if expected is None: raise RuntimeError("Matching frozen TRAIN prediction not found")
    device = torch.device("cuda")
    model, prep = model_and_preprocessor(args.official_cnn, args.checkpoint, device)
    with torch.inference_mode():
        observed = model(prep(torch.from_numpy(historical_batch).to(device))).squeeze(1).cpu().numpy().astype(np.float64)
    logit_error = float(np.max(np.abs(observed - expected)))
    logit_tolerance = 5e-3
    passed = waveform_error <= 1e-5 and logit_error <= logit_tolerance
    payload = {"status": "PASS_WITH_CUDA_NUMERICAL_TOLERANCE" if passed else "FAIL",
               "edf_hash": __import__("hashlib").sha256(edf.encode()).hexdigest()[:16],
               "channel_hash": __import__("hashlib").sha256(channel.encode()).hexdigest()[:16],
               "clips_checked": int(len(starts)), "logits_checked": int(len(expected)),
               "historical_inference_batch_size": 8, "waveform_max_abs_error": waveform_error,
               "normal_logit_max_abs_error": logit_error,
               "pathological_probability_error_upper_bound": 0.25 * logit_error,
               "waveform_tolerance": 1e-5, "logit_tolerance": logit_tolerance,
               "interpretation": "Waveform replay is bitwise exact; the frozen CUDA forward is accepted within a declared numerical-runtime tolerance."}
    atomic_json(args.experiment / "PIPELINE_IDENTITY_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if not passed: raise RuntimeError("Frozen pipeline identity failed")


if __name__ == "__main__":
    main()
