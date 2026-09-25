from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import scipy.io as sio


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
            channel_name = channel_name[:-len(suffix)].strip()

    channel_name = channel_name.replace(" ", "").replace("'", "")
    match = re.match(r"^([A-Z]+)0*(\d+)$", channel_name)
    if match:
        channel_name = f"{match.group(1)}{match.group(2)}"
    return channel_name


def _coerce_channel_name(value) -> str:
    if isinstance(value, np.ndarray):
        return "".join(str(item) for item in value.flatten()).strip()
    return str(value).strip()


def _maybe_channel_count(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return None

    if not np.isfinite(numeric):
        return None

    numeric_int = int(round(numeric))
    if numeric_int <= 0 or abs(numeric - numeric_int) > 1e-6:
        return None
    return numeric_int


def _synthetic_channel_names(n_channels: int) -> List[str]:
    return [f"C{channel_idx}" for channel_idx in range(1, int(n_channels) + 1)]


def _extract_channel_names(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, bytes):
        return [value.decode(errors="ignore").strip()]
    if isinstance(value, str):
        return [value.strip()]

    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[0] == 1 and value.shape[1] > 1:
            names: List[str] = []
            for item in value[0]:
                names.extend(_extract_channel_names(item))
            return [name for name in names if name]

        if value.ndim == 2 and value.shape[1] == 1 and value.shape[0] > 1:
            names = []
            for item in value[:, 0]:
                names.extend(_extract_channel_names(item))
            return [name for name in names if name]

        if value.dtype == object:
            names = []
            for item in value.ravel():
                names.extend(_extract_channel_names(item))
            return [name for name in names if name]

        if value.dtype.kind in {"U", "S"}:
            flat = value.ravel()
            if value.ndim == 0:
                return [str(value.item()).strip()]
            if value.ndim == 1 and all(len(str(item)) == 1 for item in flat):
                return ["".join(str(item) for item in flat).strip()]
            if value.ndim >= 2 and all(len(str(item)) == 1 for item in flat):
                return ["".join(str(item) for item in row).strip() for row in value]
            return [str(item).strip() for item in flat if str(item).strip()]

        return [_coerce_channel_name(item) for item in value.ravel()]

    if isinstance(value, (list, tuple)):
        if value and all(
            isinstance(item, (str, bytes, np.str_, np.bytes_)) and len(str(item)) == 1
            for item in value
        ):
            return [
                "".join(
                    item.decode(errors="ignore") if isinstance(item, bytes) else str(item)
                    for item in value
                ).strip()
            ]

        names = []
        for item in value:
            names.extend(_extract_channel_names(item))
        return [name for name in names if name]

    return [str(value).strip()]


def _resolve_channel_names(value: Any, expected_n_channels: Optional[int] = None) -> List[str]:
    channel_count = _maybe_channel_count(value)
    if channel_count is not None:
        return _synthetic_channel_names(channel_count)

    channel_names = _extract_channel_names(value)
    if (
        expected_n_channels is not None
        and len(channel_names) == 1
        and expected_n_channels > 1
    ):
        fallback_count = _maybe_channel_count(channel_names[0])
        if fallback_count == expected_n_channels:
            return _synthetic_channel_names(fallback_count)

    return channel_names


def _load_mat_signal_payload(mat_data: dict, mat_path: str) -> Tuple[np.ndarray, float, List[str]]:
    eeg_data = mat_data.get("eeg_data")
    info = mat_data.get("info")

    if eeg_data is not None and info is not None:
        if not isinstance(eeg_data, np.ndarray) or eeg_data.ndim != 2:
            raise ValueError(
                f"eeg_data must be a 2D array for {mat_path}, got shape {getattr(eeg_data, 'shape', None)}."
            )

        n_channels = getattr(info, "n_channels", None)
        if n_channels is not None and eeg_data.shape[0] != int(n_channels):
            if eeg_data.shape[1] == int(n_channels):
                eeg_data = eeg_data.T
            else:
                raise ValueError(
                    f"eeg_data shape {eeg_data.shape} does not match info.n_channels={n_channels} for {mat_path}."
                )

        sfreq = float(getattr(info, "sfreq"))
        ch_names_raw = _resolve_channel_names(
            getattr(info, "ch_names"),
            expected_n_channels=int(eeg_data.shape[0]),
        )
    else:
        eeg_data = mat_data.get("data")
        ch_names = mat_data.get("ch_names")
        sfreq_value = mat_data.get("sfreq")
        if eeg_data is None or ch_names is None or sfreq_value is None:
            raise ValueError(
                f"Missing compatible MAT signal keys in {mat_path}. "
                "Expected either (eeg_data, info) or (data, ch_names, sfreq)."
            )
        if not isinstance(eeg_data, np.ndarray) or eeg_data.ndim != 2:
            raise ValueError(
                f"data must be a 2D array for {mat_path}, got shape {getattr(eeg_data, 'shape', None)}."
            )

        ch_names_raw = _resolve_channel_names(
            ch_names,
            expected_n_channels=int(min(eeg_data.shape)) if eeg_data.ndim == 2 else 0,
        )
        n_channels = len(ch_names_raw)
        if eeg_data.shape[0] != n_channels:
            if eeg_data.shape[1] == n_channels:
                eeg_data = eeg_data.T
            else:
                raise ValueError(
                    f"data shape {eeg_data.shape} does not match channel count {n_channels} for {mat_path}."
                )

        sfreq = float(np.asarray(sfreq_value).squeeze())

    if len(ch_names_raw) != eeg_data.shape[0]:
        raise ValueError(
            f"Channel-name count {len(ch_names_raw)} does not match eeg_data channels {eeg_data.shape[0]} for {mat_path}."
        )

    return eeg_data.astype(np.float32, copy=False), sfreq, ch_names_raw


def inspect_mat_file(mat_path) -> Tuple[np.ndarray, float, List[str]]:
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return _load_mat_signal_payload(mat_data, str(mat_path))


def _resolve_keep_indices(
    channel_names_norm: Sequence[str],
    keep_channel_names_norm: Optional[Iterable[str]],
    keep_channel_indices: Optional[Iterable[int]],
    resolved_channel_names_norm: Optional[Sequence[str]],
) -> Tuple[List[int], List[str]]:
    if keep_channel_indices is not None:
        keep_indices = [int(index) for index in keep_channel_indices]
        if resolved_channel_names_norm is not None:
            picked_channels_norm = [str(name) for name in resolved_channel_names_norm]
            if len(picked_channels_norm) != len(keep_indices):
                raise ValueError("resolved_channel_names_norm must match keep_channel_indices length.")
        else:
            picked_channels_norm = [channel_names_norm[idx] for idx in keep_indices]
        return keep_indices, picked_channels_norm

    keep_set = None if keep_channel_names_norm is None else set(keep_channel_names_norm)
    keep_indices: List[int] = []
    picked_channels_norm: List[str] = []
    seen = set()

    for idx, channel_name_norm in enumerate(channel_names_norm):
        if keep_set is not None and channel_name_norm not in keep_set:
            continue
        if channel_name_norm in seen:
            continue
        seen.add(channel_name_norm)
        keep_indices.append(idx)
        picked_channels_norm.append(channel_name_norm)

    return keep_indices, picked_channels_norm


def load_and_preprocess_mat(
    mat_path,
    target_sfreq=512.0,
    bandpass_low=1.0,
    bandpass_high=250.0,
    keep_channel_names_norm=None,
    keep_channel_indices=None,
    resolved_channel_names_norm=None,
    eeg_data=None,
    sfreq=None,
    ch_names_raw=None,
):
    if float(target_sfreq) < float(bandpass_high) * 2.0:
        raise ValueError(
            f"target_sfreq={target_sfreq} Hz is too low for a {bandpass_high} Hz spectral ceiling."
        )

    if eeg_data is None or sfreq is None or ch_names_raw is None:
        eeg_data, sfreq, ch_names_raw = inspect_mat_file(mat_path)
    else:
        eeg_data = np.asarray(eeg_data, dtype=np.float32)
        sfreq = float(sfreq)
        ch_names_raw = [str(name) for name in ch_names_raw]

    channel_names_norm = [normalize_channel_name(name) for name in ch_names_raw]
    keep_indices, picked_channels_norm = _resolve_keep_indices(
        channel_names_norm,
        keep_channel_names_norm,
        keep_channel_indices,
        resolved_channel_names_norm,
    )
    if not keep_indices:
        raise ValueError(f"No valid channels remain after MAT channel filtering: {mat_path}")

    filtered_data = eeg_data[np.asarray(keep_indices, dtype=np.int64)]
    info = mne.create_info(
        ch_names=picked_channels_norm,
        sfreq=sfreq,
        ch_types=["seeg"] * len(picked_channels_norm),
    )
    raw = mne.io.RawArray(filtered_data, info, verbose="ERROR")

    raw_nyquist = float(raw.info["sfreq"]) / 2.0
    max_notch_freq = max(0.0, raw_nyquist - 5.0)
    freqs = np.arange(50.0, max_notch_freq, 50.0)
    if len(freqs) > 0:
        raw.notch_filter(freqs=freqs, verbose="ERROR")

    raw_sfreq = float(raw.info["sfreq"])
    if raw_nyquist < float(bandpass_high):
        print(
            f"WARNING: {mat_path} original sampling rate {raw_sfreq:.2f} Hz cannot faithfully preserve "
            f"frequencies up to {bandpass_high:.1f} Hz before resampling."
        )

    h_freq = min(float(bandpass_high), raw_nyquist - 0.1)
    if h_freq <= float(bandpass_low):
        raise ValueError(
            f"Invalid bandpass for {mat_path}: low={bandpass_low} Hz, high={h_freq:.2f} Hz."
        )

    raw.filter(l_freq=float(bandpass_low), h_freq=h_freq, verbose="ERROR")
    if float(raw.info["sfreq"]) != float(target_sfreq):
        raw.resample(target_sfreq)

    data = raw.get_data().astype(np.float32, copy=False)
    data = data - np.median(data, axis=0, keepdims=True)
    return raw, picked_channels_norm, data, ch_names_raw


__all__ = ["inspect_mat_file", "load_and_preprocess_mat", "normalize_channel_name"]
