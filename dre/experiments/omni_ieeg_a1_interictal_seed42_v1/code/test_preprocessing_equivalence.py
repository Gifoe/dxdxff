"""Numerical check of per-channel versus official MNE RawArray preprocessing."""

import json
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd


def main():
    cohort = pd.read_csv(Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v1/audit/OMNI_COHORT_AUDIT.csv"))
    row = cohort.loc[cohort["official_split"] == "train"].iloc[0]
    path = Path("F:/Omni-iEEG/signal_cache") / Path(row.edf).with_suffix(".edf.h5")
    with h5py.File(path, "r") as h5:
        meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
        header = meta["signal_headers"][0]
        n = int(63 * float(header["sample_frequency"]))
        digital = np.asarray(h5["digital/ch0000"][:n], dtype=np.float64)
    rate = float(header["sample_frequency"])
    physical = (digital - float(header["digital_min"])) * (
        (float(header["physical_max"]) - float(header["physical_min"])) /
        (float(header["digital_max"]) - float(header["digital_min"]))) + float(header["physical_min"])
    direct = mne.filter.notch_filter(physical, Fs=rate, freqs=[60],
                                     notch_widths=2, n_jobs=1, verbose=False)
    direct = mne.filter.resample(direct, up=300, down=rate, npad="auto",
                                 n_jobs=1, verbose=False)
    info = mne.create_info(["check"], sfreq=rate, ch_types="seeg")
    raw = mne.io.RawArray((physical / 1e6)[None, :], info, verbose=False)
    raw.notch_filter(60, n_jobs=1, notch_widths=2, verbose=False)
    raw.resample(300, n_jobs=1, verbose=False)
    official_style = raw.get_data()[0] * 1e6
    if direct.shape != official_style.shape:
        raise RuntimeError(f"Resample lengths differ: {direct.shape}, {official_style.shape}")
    error = float(np.max(np.abs(direct - official_style)))
    tolerance = 1e-5
    if error > tolerance:
        raise RuntimeError(f"Channel-separable preprocessing differs: max error={error}")
    print(json.dumps({"pass": True, "max_abs_difference_uV": error,
                      "samples": len(direct), "rate_hz": rate,
                      "comparison": "direct notch/resample vs MNE RawArray full-record API"}))


if __name__ == "__main__":
    main()
