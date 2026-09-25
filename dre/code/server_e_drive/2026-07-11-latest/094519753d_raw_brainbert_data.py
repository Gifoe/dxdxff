from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data_factory import build_outer_splits
from ez_dataset import build_or_load_run_records, flatten_window_samples


@dataclass(frozen=True)
class SpectrogramPatchBatch:
    patches: np.ndarray
    time_ids: np.ndarray
    freq_ids: np.ndarray
    time_patch_count: int
    freq_patch_count: int
    duration_sec: float | None


@dataclass(frozen=True)
class RawRecordChannelItem:
    subject_id: str
    run_id: str
    sample_id: str
    channel_name: str
    channel_index: int
    raw: np.ndarray
    sfreq: float
    duration_sec: float | None


def normalize_channel_name(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("EEG"):
        text = text[3:].strip()
    text = re.sub(r"[\s_\-]+", "", text)
    match = re.match(r"^([A-Z]+)0*([0-9]+)$", text)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return text


def parse_contact_topology(channel_name: Any) -> tuple[str | None, int | None]:
    norm = normalize_channel_name(channel_name)
    match = re.match(r"^([A-Z]+)([0-9]+)$", norm)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def read_cache_samples(
    cache_path: str | Path,
    *,
    outcome_subset: str,
    include_raw_waveform: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    args = SimpleNamespace(
        window_cache_path=str(cache_path),
        sample_cache_path=None,
        outcome_subset=outcome_subset,
        drop_high_ez_fraction_lzu=False,
        output_dir=None,
    )
    run_records, patient_index = build_or_load_run_records(args)
    samples = flatten_window_samples(run_records, include_raw_waveform=include_raw_waveform)
    return samples, patient_index


def load_subject_splits(
    feature_cache_path: str | Path,
    *,
    split_strategy: str,
    n_splits: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    success_args = SimpleNamespace(
        window_cache_path=str(feature_cache_path),
        sample_cache_path=None,
        outcome_subset="success",
        drop_high_ez_fraction_lzu=False,
        output_dir=None,
    )
    _, success_index = build_or_load_run_records(success_args)
    splits = build_outer_splits(
        success_index,
        split_strategy=split_strategy,
        n_splits=int(n_splits),
        random_seed=int(random_seed),
    )

    sf_args = SimpleNamespace(
        window_cache_path=str(feature_cache_path),
        sample_cache_path=None,
        outcome_subset="success_failure",
        drop_high_ez_fraction_lzu=False,
        output_dir=None,
    )
    _, success_failure_index = build_or_load_run_records(sf_args)
    failure_subjects = sorted(
        str(sid) for sid, meta in success_failure_index.items() if str(meta.get("outcome_group")) == "failure"
    )
    return splits, failure_subjects


def subject_sets_for_fold(
    splits: Sequence[dict[str, Any]],
    *,
    fold_idx: int,
    failure_subjects: Sequence[str],
    ssl_subject_mode: str = "success_train_only",
) -> dict[str, list[str]]:
    selected = None
    for split in splits:
        if int(split.get("fold_idx")) == int(fold_idx):
            selected = split
            break
    if selected is None:
        raise ValueError(f"fold_idx={fold_idx} not present in available splits {[s.get('fold_idx') for s in splits]}")
    success_train = sorted(str(sid) for sid in selected["train_subjects"])
    success_test = sorted(str(sid) for sid in selected["test_subjects"])
    failure_ssl: list[str] = []
    if str(ssl_subject_mode) == "success_train_plus_all_failure":
        failure_ssl = sorted(str(sid) for sid in failure_subjects)
    ssl_subjects = sorted(set(success_train).union(failure_ssl))
    leakage = sorted(set(success_test).intersection(ssl_subjects))
    return {
        "success_train_subjects": success_train,
        "success_test_subjects": success_test,
        "failure_ssl_subjects": failure_ssl,
        "ssl_subjects": ssl_subjects,
        "leakage_success_test_subjects_in_ssl": leakage,
    }


def _resample_raw(raw: np.ndarray, sfreq: float, target_sfreq: float) -> np.ndarray:
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid source sfreq={sfreq!r}")
    if abs(float(sfreq) - float(target_sfreq)) < 1e-6:
        return raw.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        source = int(round(float(sfreq)))
        target = int(round(float(target_sfreq)))
        gcd = math.gcd(source, target)
        return resample_poly(raw, target // gcd, source // gcd).astype(np.float32, copy=False)
    except Exception:
        tensor = torch.as_tensor(raw, dtype=torch.float32).view(1, 1, -1)
        n_samples = max(1, int(round(raw.shape[-1] * float(target_sfreq) / float(sfreq))))
        return (
            torch.nn.functional.interpolate(tensor, size=n_samples, mode="linear", align_corners=False)
            .view(-1)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )


def build_log_spectrogram(
    raw: np.ndarray,
    *,
    sfreq: float,
    resample_sfreq: float,
    n_fft: int,
    hop_length: int,
    freq_min: float,
    freq_max: float,
) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw_arr.size < max(int(n_fft), 2):
        return np.zeros((0, 0), dtype=np.float32)
    raw_arr = _resample_raw(raw_arr, float(sfreq), float(resample_sfreq))
    raw_arr = raw_arr.astype(np.float32, copy=False)
    raw_arr = raw_arr - float(np.mean(raw_arr))
    std = float(np.std(raw_arr))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    raw_arr = raw_arr / std

    wave = torch.as_tensor(raw_arr, dtype=torch.float32)
    window = torch.hann_window(int(n_fft), dtype=torch.float32)
    stft = torch.stft(
        wave,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(n_fft),
        window=window,
        center=True,
        return_complex=True,
    )
    power = stft.real.square() + stft.imag.square()
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(resample_sfreq))
    freq_mask = (freqs >= float(freq_min)) & (freqs <= min(float(freq_max), float(resample_sfreq) / 2.0))
    if not np.any(freq_mask):
        return np.zeros((0, 0), dtype=np.float32)
    log_power = torch.log1p(power[freq_mask, :]).transpose(0, 1).cpu().numpy().astype(np.float32, copy=False)
    return log_power


def build_spectrogram_patches(
    raw: np.ndarray,
    *,
    sfreq: float,
    resample_sfreq: float,
    n_fft: int,
    hop_length: int,
    freq_min: float,
    freq_max: float,
    patch_time: int,
    patch_freq: int,
    mean: float,
    std: float,
    duration_sec: float | None = None,
) -> SpectrogramPatchBatch:
    spec = build_log_spectrogram(
        raw,
        sfreq=sfreq,
        resample_sfreq=resample_sfreq,
        n_fft=n_fft,
        hop_length=hop_length,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    if spec.size == 0:
        empty = np.zeros((0, int(patch_time) * int(patch_freq)), dtype=np.float32)
        return SpectrogramPatchBatch(empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), 0, 0, duration_sec)
    spec = (spec - float(mean)) / max(float(std), 1e-6)
    t_crop = (spec.shape[0] // int(patch_time)) * int(patch_time)
    f_crop = (spec.shape[1] // int(patch_freq)) * int(patch_freq)
    if t_crop <= 0 or f_crop <= 0:
        empty = np.zeros((0, int(patch_time) * int(patch_freq)), dtype=np.float32)
        return SpectrogramPatchBatch(empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), 0, 0, duration_sec)
    spec = spec[:t_crop, :f_crop]
    t_blocks = t_crop // int(patch_time)
    f_blocks = f_crop // int(patch_freq)
    patches = (
        spec.reshape(t_blocks, int(patch_time), f_blocks, int(patch_freq))
        .transpose(0, 2, 1, 3)
        .reshape(t_blocks * f_blocks, int(patch_time) * int(patch_freq))
        .astype(np.float32, copy=False)
    )
    time_ids = np.repeat(np.arange(t_blocks, dtype=np.int64), f_blocks)
    freq_ids = np.tile(np.arange(f_blocks, dtype=np.int64), t_blocks)
    return SpectrogramPatchBatch(patches, time_ids, freq_ids, t_blocks, f_blocks, duration_sec)


def collect_raw_record_channel_items(
    samples: Iterable[dict[str, Any]],
    *,
    subject_ids: Sequence[str] | None = None,
) -> tuple[list[RawRecordChannelItem], Counter[str]]:
    selected = None if subject_ids is None else {str(sid) for sid in subject_ids}
    items: list[RawRecordChannelItem] = []
    skip_reasons: Counter[str] = Counter()
    for sample in samples:
        subject_id = str(sample.get("subject_id"))
        if selected is not None and subject_id not in selected:
            continue
        if "raw_waveform" not in sample:
            skip_reasons["missing_raw_waveform"] += 1
            continue
        raw = np.asarray(sample["raw_waveform"], dtype=np.float32)
        if raw.ndim != 2:
            skip_reasons["raw_waveform_not_2d"] += 1
            continue
        sfreq = float(sample.get("raw_temporal_sfreq", 0.0) or 0.0)
        if not np.isfinite(sfreq) or sfreq <= 0:
            skip_reasons["invalid_raw_temporal_sfreq"] += raw.shape[0]
            continue
        channel_names = list(sample.get("channel_names_norm") or [])
        if len(channel_names) != raw.shape[0]:
            channel_names = [f"ch{idx}" for idx in range(raw.shape[0])]
        duration = sample.get("raw_temporal_duration_sec")
        duration_sec = None
        try:
            duration_sec = float(duration)
        except (TypeError, ValueError):
            duration_sec = None
        for channel_idx, channel_name in enumerate(channel_names):
            items.append(
                RawRecordChannelItem(
                    subject_id=subject_id,
                    run_id=str(sample.get("run_id", "")),
                    sample_id=str(sample.get("sample_id", "")),
                    channel_name=str(channel_name),
                    channel_index=int(channel_idx),
                    raw=raw[channel_idx],
                    sfreq=sfreq,
                    duration_sec=duration_sec,
                )
            )
    return items, skip_reasons


def audit_raw_cache_onset_timing(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    onset_centered_keys = (
        "raw_temporal_onset_centered",
        "raw_window_onset_centered",
        "raw_waveform_onset_centered",
        "onset_centered",
        "raw_temporal_zero_is_onset",
    )
    onset_reference_keys = (
        "raw_temporal_reference",
        "raw_waveform_reference",
        "temporal_reference",
        "window_reference",
    )
    checked = 0
    verified = 0
    evidence: Counter[str] = Counter()
    for sample in samples:
        if "raw_waveform" not in sample:
            continue
        checked += 1
        explicit = any(bool(sample.get(key)) for key in onset_centered_keys)
        references = {str(sample.get(key, "")).strip().lower() for key in onset_reference_keys}
        reference_ok = any(value in {"onset", "seizure_onset", "seizure_onset_centered", "ictal_onset"} for value in references)
        if explicit or reference_ok:
            verified += 1
            for key in onset_centered_keys:
                if bool(sample.get(key)):
                    evidence[key] += 1
            for key in onset_reference_keys:
                value = str(sample.get(key, "")).strip().lower()
                if value:
                    evidence[f"{key}={value}"] += 1
    onset_verified = checked > 0 and verified == checked
    return {
        "raw_cache_items_checked_for_onset_timing": int(checked),
        "raw_cache_items_with_onset_timing_evidence": int(verified),
        "onset_timing_verified": bool(onset_verified),
        "rawbb_temporal_pooling_basis": "verified_onset_reference" if onset_verified else "record_midpoint_fallback",
        "rawbb_preictal_onset_names_are_verified": bool(onset_verified),
        "onset_timing_evidence": dict(evidence),
    }


def estimate_spectrogram_normalizer(
    items: Sequence[RawRecordChannelItem],
    *,
    resample_sfreq: float,
    n_fft: int,
    hop_length: int,
    freq_min: float,
    freq_max: float,
    max_items: int = 0,
) -> dict[str, Any]:
    count = 0
    total = 0.0
    total_sq = 0.0
    used_items = 0
    max_time_frames = 0
    max_freq_bins = 0
    selected_items = list(items[: int(max_items)]) if int(max_items or 0) > 0 else list(items)
    for item in selected_items:
        spec = build_log_spectrogram(
            item.raw,
            sfreq=item.sfreq,
            resample_sfreq=resample_sfreq,
            n_fft=n_fft,
            hop_length=hop_length,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        if spec.size == 0:
            continue
        used_items += 1
        max_time_frames = max(max_time_frames, int(spec.shape[0]))
        max_freq_bins = max(max_freq_bins, int(spec.shape[1]))
        values = spec.reshape(-1).astype(np.float64)
        count += int(values.size)
        total += float(values.sum())
        total_sq += float(np.square(values).sum())
    if count <= 0:
        raise ValueError("No usable spectrogram values found for Raw-BrainBERT normalization.")
    mean = total / count
    var = max(total_sq / count - mean * mean, 1e-12)
    return {
        "mean": float(mean),
        "std": float(math.sqrt(var)),
        "num_values": int(count),
        "num_items_used": int(used_items),
        "max_time_frames": int(max_time_frames),
        "max_freq_bins": int(max_freq_bins),
    }


class RawBrainBERTPatchDataset(Dataset):
    def __init__(
        self,
        items: Sequence[RawRecordChannelItem],
        preproc: dict[str, Any],
        *,
        max_patches_per_item: int = 0,
        random_seed: int = 42,
    ) -> None:
        self.items = list(items)
        self.preproc = dict(preproc)
        self.max_patches_per_item = int(max_patches_per_item or 0)
        self.random_seed = int(random_seed)
        self.patch_count_before_clip_by_index: list[int | None] = [None] * len(self.items)
        self.patch_count_after_clip_by_index: list[int | None] = [None] * len(self.items)

    @property
    def num_items_clipped(self) -> int:
        return int(
            sum(
                1
                for before, after in zip(self.patch_count_before_clip_by_index, self.patch_count_after_clip_by_index)
                if before is not None and after is not None and int(after) < int(before)
            )
        )

    def __len__(self) -> int:
        return len(self.items)

    def _clip_patches(self, patches: SpectrogramPatchBatch, index: int) -> SpectrogramPatchBatch:
        n_patches = int(patches.patches.shape[0])
        if self.max_patches_per_item <= 0 or n_patches <= self.max_patches_per_item:
            return patches
        keep = max(1, int(self.max_patches_per_item))
        rng = np.random.default_rng(self.random_seed + int(index))
        # Prefer a contiguous patch window so local time-frequency structure is preserved.
        if n_patches > keep:
            start = int(rng.integers(0, n_patches - keep + 1))
            indices = np.arange(start, start + keep, dtype=np.int64)
        else:
            indices = np.sort(rng.choice(n_patches, size=keep, replace=False)).astype(np.int64)
        return SpectrogramPatchBatch(
            patches=patches.patches[indices],
            time_ids=patches.time_ids[indices],
            freq_ids=patches.freq_ids[indices],
            time_patch_count=patches.time_patch_count,
            freq_patch_count=patches.freq_patch_count,
            duration_sec=patches.duration_sec,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        patches = build_spectrogram_patches(
            item.raw,
            sfreq=item.sfreq,
            resample_sfreq=float(self.preproc["resample_sfreq"]),
            n_fft=int(self.preproc["n_fft"]),
            hop_length=int(self.preproc["hop_length"]),
            freq_min=float(self.preproc["freq_min"]),
            freq_max=float(self.preproc["freq_max"]),
            patch_time=int(self.preproc["patch_time"]),
            patch_freq=int(self.preproc["patch_freq"]),
            mean=float(self.preproc["mean"]),
            std=float(self.preproc["std"]),
            duration_sec=item.duration_sec,
        )
        before = int(patches.patches.shape[0])
        patches = self._clip_patches(patches, index)
        after = int(patches.patches.shape[0])
        self.patch_count_before_clip_by_index[index] = before
        self.patch_count_after_clip_by_index[index] = after
        return {
            "item": item,
            "patches": patches,
            "patch_count_before_clip": before,
            "patch_count_after_clip": after,
        }


def collate_patch_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid_entries = [entry for entry in batch if entry["patches"].patches.shape[0] > 0]
    if not valid_entries:
        return {
            "patches": torch.zeros((0, 0, 0), dtype=torch.float32),
            "time_ids": torch.zeros((0, 0), dtype=torch.long),
            "freq_ids": torch.zeros((0, 0), dtype=torch.long),
            "valid_mask": torch.zeros((0, 0), dtype=torch.bool),
            "items": [],
        }
    max_len = max(entry["patches"].patches.shape[0] for entry in valid_entries)
    patch_dim = valid_entries[0]["patches"].patches.shape[1]
    patches = np.zeros((len(valid_entries), max_len, patch_dim), dtype=np.float32)
    time_ids = np.zeros((len(valid_entries), max_len), dtype=np.int64)
    freq_ids = np.zeros((len(valid_entries), max_len), dtype=np.int64)
    valid_mask = np.zeros((len(valid_entries), max_len), dtype=bool)
    items: list[RawRecordChannelItem] = []
    for row_idx, entry in enumerate(valid_entries):
        patch_batch = entry["patches"]
        n = patch_batch.patches.shape[0]
        patches[row_idx, :n, :] = patch_batch.patches
        time_ids[row_idx, :n] = patch_batch.time_ids
        freq_ids[row_idx, :n] = patch_batch.freq_ids
        valid_mask[row_idx, :n] = True
        items.append(entry["item"])
    return {
        "patches": torch.as_tensor(patches, dtype=torch.float32),
        "time_ids": torch.as_tensor(time_ids, dtype=torch.long),
        "freq_ids": torch.as_tensor(freq_ids, dtype=torch.long),
        "valid_mask": torch.as_tensor(valid_mask, dtype=torch.bool),
        "items": items,
    }


def pool_hidden_states(
    hidden: np.ndarray,
    *,
    time_ids: np.ndarray,
    duration_sec: float | None,
    resample_sfreq: float,
    hop_length: int,
    patch_time: int,
) -> dict[str, np.ndarray]:
    if hidden.size == 0:
        raise ValueError("Cannot pool empty hidden states.")
    time_ids = np.asarray(time_ids, dtype=np.int64)
    emb_all = hidden.mean(axis=0)
    unique_time_ids = np.unique(time_ids)
    early_limit = max(1, int(math.ceil(len(unique_time_ids) * 0.25)))
    early_times = set(unique_time_ids[:early_limit].tolist())
    emb_early = hidden[np.asarray([tid in early_times for tid in time_ids])].mean(axis=0)

    if duration_sec is not None and np.isfinite(duration_sec) and duration_sec > 0:
        patch_centers = (time_ids.astype(np.float32) * float(patch_time) + float(patch_time) / 2.0) * (
            float(hop_length) / float(resample_sfreq)
        )
        rel_sec = patch_centers - float(duration_sec) / 2.0
        preictal_mask = rel_sec <= 0.0
        onset_mask = (rel_sec >= 0.0) & (rel_sec <= 10.0)
    else:
        midpoint = np.median(unique_time_ids) if unique_time_ids.size else 0
        preictal_mask = time_ids <= midpoint
        onset_mask = (time_ids > midpoint) & (time_ids <= midpoint + max(1, int(round(10.0 * resample_sfreq / hop_length / patch_time))))
    if not np.any(preictal_mask):
        preictal_mask = time_ids <= np.median(unique_time_ids)
    if not np.any(onset_mask):
        center = np.median(unique_time_ids) if unique_time_ids.size else 0
        onset_mask = time_ids >= center
    emb_preictal = hidden[preictal_mask].mean(axis=0) if np.any(preictal_mask) else emb_all
    emb_onset = hidden[onset_mask].mean(axis=0) if np.any(onset_mask) else emb_all
    return {
        "all": emb_all.astype(np.float32, copy=False),
        "preictal": emb_preictal.astype(np.float32, copy=False),
        "onset": emb_onset.astype(np.float32, copy=False),
        "early": emb_early.astype(np.float32, copy=False),
    }


def aggregate_record_embeddings(rows: Sequence[dict[str, Any]], *, fold_idx: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["subject_id"]), str(row["channel_name"]))].append(row)
    output: list[dict[str, Any]] = []
    for (subject_id, channel_name), records in sorted(grouped.items()):
        all_stack = np.vstack([np.asarray(record["emb_all"], dtype=np.float32) for record in records])
        base: dict[str, Any] = {
            "fold_idx": int(fold_idx),
            "subject_id": subject_id,
            "channel_name": channel_name,
            "n_records": int(len(records)),
        }
        for mode in ("all", "preictal", "onset", "early"):
            stack = np.vstack([np.asarray(record[f"emb_{mode}"], dtype=np.float32) for record in records])
            mean_vec = stack.mean(axis=0)
            for idx, value in enumerate(mean_vec):
                base[f"rawbb_{mode}_{idx}"] = float(value)
        for prefix, vec in (("rawbb_std", all_stack.std(axis=0)), ("rawbb_max", all_stack.max(axis=0))):
            for idx, value in enumerate(vec):
                base[f"{prefix}_{idx}"] = float(value)
        output.append(base)
    return output


def write_dataframe_with_parquet_fallback(df: Any, path_without_suffix: str | Path) -> Path:
    import pandas as pd

    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_embedding_table(path_base: str | Path) -> Any:
    import pandas as pd

    base = Path(path_base)
    candidates = [base]
    if base.suffix == "":
        candidates = [base.with_suffix(".parquet"), base.with_suffix(".csv")]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    raise FileNotFoundError(f"No embedding table found for {base}")


def save_preproc(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        pickle.dump(payload, fout)


def load_preproc(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fin:
        return pickle.load(fin)


__all__ = [
    "RawBrainBERTPatchDataset",
    "RawRecordChannelItem",
    "SpectrogramPatchBatch",
    "aggregate_record_embeddings",
    "audit_raw_cache_onset_timing",
    "build_log_spectrogram",
    "build_spectrogram_patches",
    "collate_patch_batch",
    "collect_raw_record_channel_items",
    "estimate_spectrogram_normalizer",
    "load_embedding_table",
    "load_preproc",
    "load_subject_splits",
    "normalize_channel_name",
    "parse_contact_topology",
    "pool_hidden_states",
    "read_cache_samples",
    "save_preproc",
    "subject_sets_for_fold",
    "write_dataframe_with_parquet_fallback",
]
