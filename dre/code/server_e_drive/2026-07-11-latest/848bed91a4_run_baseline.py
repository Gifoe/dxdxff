#!/usr/bin/env python3
"""Standalone Task 1 raw-signal baseline for the frozen 80-patient protocol.

The implementation is intentionally self-contained so it can be copied to
``/root/autodl-tmp/remote_task1_omni_baselines`` and run without patching the
legacy repository runner. Labels are NEZ=1 and EZ=0.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# The repository threshold utility is the canonical Task 1 definition: it
# selects a global fold threshold by patient-equal macro-F1 on validation data.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from task1_baselines.thresholds import select_patient_macro_threshold


@dataclass(frozen=True)
class Config:
    sfreq: float = 200.0
    window_sec: float = 4.0
    bandpass_low: float = 0.5
    bandpass_high: float = 80.0
    notch: float = 50.0
    clip: float = 8.0


class OmniRawBaseline(nn.Module):
    """Compact Omni-SEEGNet-style multiscale Conv1D encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 24, k, stride=s, padding=k // 2), nn.BatchNorm1d(24), nn.GELU())
            for k, s in ((51, 4), (17, 2), (3, 1))
        ])
        self.mix = nn.Sequential(nn.Conv1d(72, 96, 1), nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(0.5))
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.rnn = nn.LSTM(96, 64, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, 800):
            raise ValueError(f"expected [B,1,800], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("input contains NaN or Inf")
        parts = [branch(x) for branch in self.branches]
        width = min(part.shape[-1] for part in parts)
        z = self.pool(self.mix(torch.cat([part[..., :width] for part in parts], dim=1))).transpose(1, 2)
        z, _ = self.rnn(z)
        weights = torch.softmax(self.attn(z).squeeze(-1), dim=1)
        return self.head(torch.sum(z * weights.unsqueeze(-1), dim=1)).squeeze(-1)


class OmniTimeConvCNN(nn.Module):
    """Time-frequency Conv2D baseline inspired by Omni-iEEG cnn.py."""
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(128), persistent=False)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(1, 5), padding=(1, 2)), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 2), stride=(1, 2), padding=(1, 1)), nn.GELU(),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, 800):
            raise ValueError(f"expected [B,1,800], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("input contains NaN or Inf")
        spec = torch.stft(x.squeeze(1), n_fft=128, hop_length=20, window=self.window.to(x), return_complex=True).abs()
        spec = torch.log1p(spec).unsqueeze(1)
        spec = (spec - spec.mean((2, 3), keepdim=True)) / spec.std((2, 3), keepdim=True).clamp_min(1e-6)
        return self.head(self.encoder(self.temporal(spec)).flatten(1)).squeeze(-1)


class OmniCLAP(nn.Module):
    """Audio branch of the public CLAP model, adapted to 200 Hz / 4 s."""
    def __init__(self, model_id: str, *, local_files_only: bool, train_mode: str) -> None:
        super().__init__()
        try:
            from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor
            self.extractor = ClapFeatureExtractor.from_pretrained(model_id, local_files_only=local_files_only)
            self.extractor.sampling_rate = 200; self.extractor.chunk_length_s = 4; self.extractor.max_length_s = 4
            self.extractor.nb_max_samples = 800; self.extractor.n_mels = 64; self.extractor.f_min = 0; self.extractor.f_max = 100
            self.extractor.n_fft = 256; self.extractor.hop_length = 20
            self.clap = ClapAudioModelWithProjection.from_pretrained(model_id, local_files_only=local_files_only)
        except Exception as exc:
            raise RuntimeError(f"CLAP checkpoint unavailable ({model_id}). Download it first or allow network access.") from exc
        if train_mode not in {"frozen", "audio_finetune", "full"}:
            raise ValueError("clap_train_mode must be frozen, audio_finetune, or full")
        for name, parameter in self.clap.named_parameters():
            parameter.requires_grad = train_mode == "full" or (train_mode == "audio_finetune" and ("audio_model" in name or "audio_projection" in name))
        self.train_mode = train_mode
        self.head = nn.Linear(int(self.clap.config.projection_dim), 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_mode == "frozen":
            self.clap.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, 800):
            raise ValueError(f"expected [B,1,800], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("input contains NaN or Inf")
        features = self.extractor(x.squeeze(1).detach().cpu().numpy(), sampling_rate=200, return_tensors="pt", padding=True)
        features = {key: value.to(x.device) for key, value in features.items()}
        if self.train_mode == "frozen":
            with torch.no_grad():
                embeds = self.clap(**features).audio_embeds
        else:
            embeds = self.clap(**features).audio_embeds
        return self.head(embeds).squeeze(-1)

    @torch.no_grad()
    def encode_frozen(self, x: torch.Tensor) -> torch.Tensor:
        """Return cached audio embeddings; valid only for frozen CLAP."""
        if self.train_mode != "frozen":
            raise RuntimeError("embedding cache is valid only for frozen CLAP")
        self.clap.eval()
        features = self.extractor(x.squeeze(1).detach().cpu().numpy(), sampling_rate=200, return_tensors="pt", padding=True)
        features = {key: value.to(x.device) for key, value in features.items()}
        return self.clap(**features).audio_embeds


class EmbeddingHead(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x).squeeze(-1)


def build_model(name: str, *, clap_model_id: str, clap_local_files_only: bool, clap_train_mode: str) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "omni_seegnet": return OmniRawBaseline()
    if normalized == "omni_timeconv_cnn": return OmniTimeConvCNN()
    if normalized == "omni_clap": return OmniCLAP(clap_model_id, local_files_only=clap_local_files_only, train_mode=clap_train_mode)
    raise ValueError("model must be omni_seegnet, omni_timeconv_cnn, or omni_clap")


def _norm_name(value: Any) -> str:
    return "".join(str(value).strip().upper().replace("-", "").replace("_", "").split())


def _labels(record: dict[str, Any], patient: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    channels = [str(x) for x in record.get("channel_names_norm", [])]
    canonical, labels = patient.get("canonical_channels"), patient.get("labels")
    if canonical is not None and labels is not None:
        lookup = {_norm_name(c): int(float(y)) for c, y in zip(canonical, labels)}
        ez = np.asarray([lookup[_norm_name(c)] for c in channels], dtype=np.int64)
    else:
        ez = np.asarray(record["labels"], dtype=np.int64)
    if len(ez) != len(channels) or not set(ez.tolist()) <= {0, 1}:
        raise ValueError("channel labels are missing, unaligned, or non-binary")
    return channels, 1 - ez


def _preprocess(signal: np.ndarray, source_sfreq: float, cfg: Config) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    sos = butter(4, [cfg.bandpass_low, min(cfg.bandpass_high, source_sfreq / 2 - 0.5)], btype="bandpass", fs=source_sfreq, output="sos")
    x = sosfiltfilt(sos, x)
    b, a = iirnotch(cfg.notch, 30.0, fs=source_sfreq)
    x = filtfilt(b, a, x)
    if source_sfreq != cfg.sfreq:
        x = resample_poly(x, int(cfg.sfreq), int(source_sfreq))
    target = int(cfg.sfreq * cfg.window_sec)
    x = np.pad(x[:target], (0, max(0, target - len(x))))
    median = np.median(x); iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return np.clip(np.nan_to_num((x - median) / max(float(iqr), 1e-6)), -cfg.clip, cfg.clip).astype(np.float32)


def load_tokens(raw_cache: Path, subjects: set[str], cfg: Config, windows_per_run: int) -> tuple[np.ndarray, pd.DataFrame]:
    with raw_cache.open("rb") as handle:
        payload = pickle.load(handle)
    values: list[np.ndarray] = []; rows: list[dict[str, Any]] = []; observed: set[str] = set()
    for record in payload["run_records"]:
        subject = str(record.get("subject_id", ""))
        if subject not in subjects:
            continue
        sample = record["sample"]; raw = np.asarray(sample["raw_waveform"]); source_sfreq = float(sample["raw_temporal_sfreq"])
        channels, labels_nez = _labels(record, payload["patient_index"][subject])
        centers = np.asarray(sample["window_relative_centers_sec"], dtype=float)
        # Prefer peri-onset windows closest to zero; endpoint windows may lie
        # outside the audited valid raw interval after 4-second expansion.
        chosen = np.argsort(np.abs(centers))[:min(windows_per_run, len(centers))]
        origin = raw.shape[1] / 2.0; length = int(round(cfg.window_sec * source_sfreq))
        valid_start = int(sample["raw_valid_start_sample"]); valid_end = min(raw.shape[1], valid_start + int(sample["raw_valid_samples"]))
        for window_id in chosen:
            start = int(round(origin + centers[window_id] * source_sfreq - length / 2)); end = start + length
            if start < valid_start or end > valid_end:
                continue
            for channel_id, channel in enumerate(channels):
                values.append(_preprocess(raw[channel_id, start:end], source_sfreq, cfg))
                rows.append({"subject_id": subject, "center": subject.split(":", 1)[0], "seizure_id": str(record.get("run_id", "")), "channel_name": channel, "window_id": int(window_id), "label_nez": int(labels_nez[channel_id])})
        observed.add(subject)
    missing = sorted(subjects - observed)
    if missing:
        raise ValueError(f"raw cache missing subjects: {missing}")
    return np.stack(values), pd.DataFrame(rows)


def _model_input(values: torch.Tensor) -> torch.Tensor:
    return values.unsqueeze(1) if values.ndim == 2 and values.shape[-1] == 800 else values


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = _model_input(torch.from_numpy(x[start:start + batch_size])).to(device)
            output.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(output)


def fit_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    validation_x: np.ndarray,
    validation_rows: pd.DataFrame,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    amp: bool,
    checkpoint: Path,
    patience: int,
    min_epochs_before_early_stop: int,
    seed: int,
) -> tuple[nn.Module, Any, int, list[dict[str, float]]]:
    """Fit from source centers and choose a checkpoint on validation only."""
    model = model.to(device); optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    pos = max(int(y.sum()), 1); neg = max(int((1 - y).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))), batch_size=batch_size, shuffle=True, generator=generator, drop_last=isinstance(model, OmniCLAP))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True); bx, by = _model_input(bx).to(device), by.to(device)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                loss = loss_fn(model(bx), by)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        validation_channel = aggregate(validation_rows.reset_index(drop=True), predict(model, validation_x, device, batch_size))
        threshold = select_patient_macro_threshold(validation_channel, source="outer_validation_patient_macro_f1")
        key = (float(threshold.patient_macro_f1), float(threshold.patient_ez_f1), -epoch)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_patient_macro_f1": key[0], "validation_patient_ez_f1": key[1], "validation_threshold": float(threshold.threshold)})
        if best_key is None or key > best_key:
            best_key = key; best_state = copy.deepcopy(model.state_dict()); best_threshold = threshold; best_epoch = epoch; stale = 0
        else:
            stale += 1
        if epoch >= int(min_epochs_before_early_stop) and stale >= int(patience):
            break
    if best_state is None or best_threshold is None:
        raise RuntimeError("TimeConv-CNN did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "best_epoch": best_epoch, "selected_threshold": float(best_threshold.threshold), "threshold_source": best_threshold.source}, checkpoint)
    return model, best_threshold, best_epoch, history


def cache_clap_embeddings(model: OmniCLAP, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.to(device); model.eval(); chunks: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start:start + batch_size]).unsqueeze(1).to(device)
        chunks.append(model.encode_frozen(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def aggregate(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    table = rows.copy(); table["score_nez_probability"] = scores
    return table.groupby(["subject_id", "center", "channel_name", "label_nez"], as_index=False).agg(score_nez_probability=("score_nez_probability", "mean"), seizure_count=("seizure_id", "nunique"), window_count=("window_id", "count"))


def main() -> int:
    parser = argparse.ArgumentParser(); root = Path("/root/autodl-tmp")
    parser.add_argument("--raw-cache", type=Path, default=root / "raw/all_window_cache.pkl"); parser.add_argument("--cohort", type=Path, default=root / "csv/cohort_subjects.csv")
    parser.add_argument("--folds", type=Path, default=root / "csv/fixed_outer_folds_v3_qbc.csv"); parser.add_argument("--splits", type=Path, default=root / "csv/outer_fold_ledger.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/task1_omni_raw"); parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--model", choices=["omni_seegnet", "omni_timeconv_cnn", "omni_clap"], default="omni_seegnet")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--patience", type=int, default=8); parser.add_argument("--min-epochs-before-early-stop", type=int, default=18); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--windows-per-run", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cuda"); parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clap-model-id", default="laion/clap-htsat-fused"); parser.add_argument("--clap-local-files-only", action=argparse.BooleanOptionalAction, default=False); parser.add_argument("--clap-train-mode", choices=["frozen", "audio_finetune", "full"], default="frozen")
    parser.add_argument("--allow-legacy-derived-validation", action="store_true", help="Only for historical smoke runs with train/test-only ledgers.")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); started = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cohort = pd.read_csv(args.cohort); folds = pd.read_csv(args.folds); splits = pd.read_csv(args.splits).rename(columns={"fold_idx": "outer_fold", "split_role": "partition"})
    if cohort.subject_id.nunique() != 80 or folds.subject_id.nunique() != 80:
        raise ValueError("expected exactly 80 cohort/fold subjects")
    if "partition" not in splits.columns:
        raise ValueError("Split manifest must include partition or split_role.")
    current = splits[splits.outer_fold == args.outer_fold].copy(); current["partition"] = current.partition.astype(str).str.lower().replace({"train": "fit", "val": "validation"})
    if "validation" not in set(current.partition):
        if not args.allow_legacy_derived_validation:
            raise ValueError("An explicit validation partition is required; refusing to derive validation from another outer fold.")
        # The supplied ledger contains only outer train/test. Reserve the next
        # frozen outer-fold group from outer-train for validation; outer-test
        # is never used for epoch or threshold selection.
        next_fold = args.outer_fold % int(folds.outer_fold.max()) + 1
        validation_subjects = set(folds.loc[folds.outer_fold == next_fold, "subject_id"].astype(str))
        current.loc[current.subject_id.astype(str).isin(validation_subjects) & current.partition.eq("fit"), "partition"] = "validation"
    if set(current.partition) != {"fit", "validation", "test"} or current.subject_id.duplicated().any():
        raise ValueError("invalid fixed split manifest")
    values, rows = load_tokens(args.raw_cache, set(cohort.subject_id.astype(str)), Config(), args.windows_per_run)
    partition = rows[["subject_id"]].merge(current[["subject_id", "partition"]], on="subject_id", validate="many_to_one")["partition"].to_numpy()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    fit = partition == "fit"; val = partition == "validation"; test = partition == "test"
    backbone = build_model(args.model, clap_model_id=args.clap_model_id, clap_local_files_only=args.clap_local_files_only, clap_train_mode=args.clap_train_mode)
    model_values = values
    if isinstance(backbone, OmniCLAP) and args.clap_train_mode == "frozen":
        print("caching frozen CLAP audio embeddings once for this fold", flush=True)
        model_values = cache_clap_embeddings(backbone, values, device, args.batch_size)
        model = EmbeddingHead(int(model_values.shape[1]))
    else:
        model = backbone
    model, threshold_result, best_epoch, history = fit_model(
        model, model_values[fit], rows.loc[fit, "label_nez"].to_numpy(),
        model_values[val], rows.loc[val].reset_index(drop=True), device, args.epochs,
        args.batch_size, args.learning_rate, args.amp,
        args.output_dir / "checkpoints" / f"{args.model}_fold_{args.outer_fold}.pt",
        args.patience, args.min_epochs_before_early_stop, args.seed,
    )
    threshold = float(threshold_result.threshold)
    result = aggregate(rows.loc[test].reset_index(drop=True), predict(model, model_values[test], device, args.batch_size))
    result["model"] = args.model; result["seed"] = args.seed; result["outer_fold"] = args.outer_fold; result["selected_threshold"] = threshold; result["threshold_source"] = threshold_result.source; result["predicted_nez"] = result.score_nez_probability.ge(threshold).astype(int); result["predicted_ez"] = 1 - result.predicted_nez; result["clinical_true_nez"] = result.label_nez; result["clinical_true_ez"] = 1 - result.label_nez; result["cohort_name"] = "task1_reference_80"
    ledger = args.output_dir / f"oof_ledgers/{args.model}"; ledger.mkdir(parents=True, exist_ok=True); result.to_csv(ledger / f"seed_{args.seed}_channel_oof.csv", index=False)
    (args.output_dir / "logs").mkdir(exist_ok=True); pd.DataFrame(history).to_csv(args.output_dir / "logs" / f"{args.model}_training_history.csv", index=False)
    provenance = {"model_name": args.model, "upstream_repository": "Omni-iEEG/Omni-iEEG", "input_sfreq": 200, "window_sec": 4, "n_times": 800, "expected_subjects": 80, "outer_fold": args.outer_fold, "windows_per_run": args.windows_per_run, "parameter_count": sum(p.numel() for p in model.parameters()), "elapsed_sec": time.time() - started, "best_epoch": best_epoch, "threshold": {"value": threshold, "source": threshold_result.source, "patient_macro_f1": threshold_result.patient_macro_f1}, "config": vars(args)}
    (args.output_dir / "provenance").mkdir(exist_ok=True); (args.output_dir / "provenance" / f"{args.model}_architecture.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"oof_rows": len(result), "threshold": threshold, "device": str(device), "output": str(ledger)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
