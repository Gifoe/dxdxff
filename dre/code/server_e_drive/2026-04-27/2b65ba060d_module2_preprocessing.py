from __future__ import annotations

import re

import mne
import numpy as np
import pandas as pd
from scipy import signal


def normalize_channel_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)

    channel_name = name.upper().strip()
    if channel_name.startswith("EEG "):
        channel_name = channel_name[4:].strip()
    elif channel_name.startswith("POL "):
        channel_name = channel_name[4:].strip()

    for suffix in ["-REF", " REF", "_REF"]:
        if channel_name.endswith(suffix):
            channel_name = channel_name[: -len(suffix)].strip()

    channel_name = channel_name.replace(" ", "")
    match = re.match(r"^([A-Z]+)0*(\d+)$", channel_name)
    if match:
        channel_name = f"{match.group(1)}{match.group(2)}"
    return channel_name


def load_and_preprocess_edf(
    edf_path: str,
    channels_path: str,
    target_sfreq: float = 512.0,
    bandpass_low: float = 1.0,
    bandpass_high: float = 150.0,
):
    if float(target_sfreq) < float(bandpass_high) * 2.0:
        raise ValueError(
            f"target_sfreq={target_sfreq} Hz is too low for a {bandpass_high} Hz spectral ceiling."
        )

    channels_df = pd.read_csv(channels_path, sep="\t")
    channels_df["name_norm"] = channels_df["name"].apply(normalize_channel_name)
    if "type" in channels_df.columns:
        channels_df["type_upper"] = channels_df["type"].astype(str).str.upper()
    else:
        channels_df["type_upper"] = "SEEG"
    if "status" in channels_df.columns:
        status_series = channels_df["status"]
    else:
        status_series = pd.Series("good", index=channels_df.index)
    channels_df["status_lower"] = status_series.astype(str).str.lower()
    if "good" in channels_df.columns:
        good_series = pd.to_numeric(channels_df["good"], errors="coerce").fillna(0).astype(int)
        good_mask = good_series.eq(1)
    else:
        good_mask = channels_df["status_lower"] != "bad"

    valid_mask = channels_df["type_upper"].isin(["ECOG", "SEEG", "STEREOEEG"]) & good_mask
    valid_channels_norm = channels_df.loc[valid_mask, "name_norm"].tolist()

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    original_to_norm = {}
    for channel_name in raw.ch_names:
        original_to_norm[channel_name] = normalize_channel_name(channel_name)

    raw.rename_channels(original_to_norm)
    existing_valid_norm = [channel for channel in valid_channels_norm if channel in raw.ch_names]

    seen = set()
    picked_channels = []
    for channel_name in existing_valid_norm:
        if channel_name not in seen:
            picked_channels.append(channel_name)
            seen.add(channel_name)

    raw.pick(picked_channels)
    if len(raw.ch_names) == 0:
        raise ValueError(f"No valid intracranial channels found in {edf_path}.")

    freqs = np.arange(60.0, raw.info["sfreq"] / 2.0, 60.0)
    if len(freqs) > 0:
        raw.notch_filter(freqs=freqs, verbose="ERROR")

    raw_sfreq = float(raw.info["sfreq"])
    raw_nyquist = raw_sfreq / 2.0
    h_freq = min(float(bandpass_high), raw_nyquist - 0.1)
    if h_freq <= float(bandpass_low):
        raise ValueError(
            f"Invalid bandpass for {edf_path}: low={bandpass_low} Hz, high={h_freq:.2f} Hz."
        )

    raw.filter(l_freq=float(bandpass_low), h_freq=h_freq, verbose="ERROR")
    if float(raw.info["sfreq"]) != float(target_sfreq):
        raw.resample(float(target_sfreq))

    data = raw.get_data().astype(np.float32, copy=False)
    data = signal.detrend(data, axis=-1, type="linear").astype(np.float32, copy=False)
    data -= np.median(data, axis=1, keepdims=True).astype(np.float32, copy=False)
    return raw, picked_channels, data, original_to_norm


__all__ = ["load_and_preprocess_edf", "normalize_channel_name"]
