from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuroez_c.raw_brainbert import RawBrainBERTEncoder, make_random_patch_mask, masked_patch_loss
from neuroez_c.raw_brainbert_data import (
    RawBrainBERTPatchDataset,
    collate_patch_batch,
    collect_raw_record_channel_items,
    estimate_spectrogram_normalizer,
    load_subject_splits,
    read_cache_samples,
    save_preproc,
    subject_sets_for_fold,
)


def _device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _summary(values: list[int | None]) -> dict[str, Any]:
    nums = [int(v) for v in values if v is not None]
    if not nums:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    arr = np.asarray(nums, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def pretrain_raw_brainbert_encoder(args: Any) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits, failure_subjects = load_subject_splits(
        args.feature_cache_path,
        split_strategy=args.split_strategy,
        n_splits=int(args.n_splits),
        random_seed=int(args.random_seed),
    )
    split_summary = subject_sets_for_fold(
        splits,
        fold_idx=int(args.fold_idx),
        failure_subjects=failure_subjects,
        ssl_subject_mode=str(args.ssl_subject_mode),
    )
    if split_summary["leakage_success_test_subjects_in_ssl"]:
        raise ValueError(
            "Raw-BrainBERT SSL leakage: success test subjects entered SSL train: "
            f"{split_summary['leakage_success_test_subjects_in_ssl']}"
        )
    samples, _ = read_cache_samples(args.raw_cache_path, outcome_subset="success_failure", include_raw_waveform=True)
    items, skip_reasons = collect_raw_record_channel_items(samples, subject_ids=split_summary["ssl_subjects"])
    if int(getattr(args, "max_train_items", 0) or 0) > 0:
        items = items[: int(args.max_train_items)]
    if not items:
        raise ValueError("No Raw-BrainBERT SSL record-channel items were available. Check raw_waveform in raw cache.")
    effective_freq_max = min(float(args.freq_max), float(args.resample_sfreq) / 2.0)

    normalizer = estimate_spectrogram_normalizer(
        items,
        resample_sfreq=float(args.resample_sfreq),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        freq_min=float(args.freq_min),
        freq_max=effective_freq_max,
        max_items=0,
    )
    preproc = {
        **normalizer,
        "raw_cache_path": str(args.raw_cache_path),
        "feature_cache_path": str(args.feature_cache_path),
        "fold_idx": int(args.fold_idx),
        "split_strategy": str(args.split_strategy),
        "n_splits": int(args.n_splits),
        "random_seed": int(args.random_seed),
        "resample_sfreq": float(args.resample_sfreq),
        "n_fft": int(args.n_fft),
        "hop_length": int(args.hop_length),
        "freq_min": float(args.freq_min),
        "requested_freq_max": float(args.freq_max),
        "freq_max": float(effective_freq_max),
        "effective_freq_max": float(effective_freq_max),
        "patch_time": int(args.patch_time),
        "patch_freq": int(args.patch_freq),
        "patch_dim": int(args.patch_time) * int(args.patch_freq),
        "max_time_patches": max(1, int(normalizer["max_time_frames"]) // int(args.patch_time) + 2),
        "max_freq_patches": max(1, int(normalizer["max_freq_bins"]) // int(args.patch_freq) + 2),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "ssl_subject_mode": str(args.ssl_subject_mode),
        "subject_split": split_summary,
    }
    preproc_path = output_dir / f"raw_brainbert_preproc_fold_{int(args.fold_idx)}.pkl"
    save_preproc(preproc_path, preproc)

    device = _device(args.device)
    model = RawBrainBERTEncoder(
        patch_dim=int(preproc["patch_dim"]),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        max_time_patches=int(preproc["max_time_patches"]),
        max_freq_patches=int(preproc["max_freq_patches"]),
    ).to(device)
    dataset = RawBrainBERTPatchDataset(
        items,
        preproc,
        max_patches_per_item=int(args.max_patches_per_item),
        random_seed=int(args.random_seed) + int(args.fold_idx) * 1000,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_patch_batch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_curve: list[dict[str, Any]] = []
    best_loss: float | None = None
    final_loss: float | None = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            if batch["patches"].numel() == 0:
                continue
            patches = batch["patches"].to(device)
            time_ids = batch["time_ids"].to(device)
            freq_ids = batch["freq_ids"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            mask = make_random_patch_mask(valid_mask, float(args.mask_ratio))
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(patches, time_ids, freq_ids, mask=mask, key_padding_mask=~valid_mask)
                loss = masked_patch_loss(out["reconstruction"], patches, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else None
        if final_loss is not None:
            best_loss = final_loss if best_loss is None else min(best_loss, final_loss)
        loss_curve.append({"epoch": epoch, "loss": final_loss})

    encoder_path = output_dir / f"raw_brainbert_encoder_fold_{int(args.fold_idx)}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "preproc_path": str(preproc_path),
        },
        encoder_path,
    )
    loss_curve_path = output_dir / f"raw_brainbert_loss_curve_fold_{int(args.fold_idx)}.csv"
    with loss_curve_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(loss_curve)
    audit = {
        "fold_idx": int(args.fold_idx),
        "ssl_subject_mode": str(args.ssl_subject_mode),
        "success_train_subjects": split_summary["success_train_subjects"],
        "success_test_subjects": split_summary["success_test_subjects"],
        "failure_ssl_subjects": split_summary["failure_ssl_subjects"],
        "leakage_success_test_subjects_in_ssl": split_summary["leakage_success_test_subjects_in_ssl"],
        "num_success_train_subjects": len(split_summary["success_train_subjects"]),
        "num_success_test_subjects": len(split_summary["success_test_subjects"]),
        "num_failure_ssl_subjects": len(split_summary["failure_ssl_subjects"]),
        "num_ssl_subjects": len(split_summary["ssl_subjects"]),
        "num_train_record_channels": int(len(items)),
        "raw_cache_path": str(args.raw_cache_path),
        "feature_cache_path": str(args.feature_cache_path),
        "resample_sfreq": float(args.resample_sfreq),
        "n_fft": int(args.n_fft),
        "hop_length": int(args.hop_length),
        "freq_min": float(args.freq_min),
        "freq_max": float(args.freq_max),
        "effective_freq_max": float(effective_freq_max),
        "patch_time": int(args.patch_time),
        "patch_freq": int(args.patch_freq),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "mask_ratio": float(args.mask_ratio),
        "epochs": int(args.epochs),
        "best_loss": best_loss,
        "final_loss": final_loss,
        "encoder_path": str(encoder_path),
        "preproc_path": str(preproc_path),
        "loss_curve_path": str(loss_curve_path),
        "max_patches_per_item": int(args.max_patches_per_item),
        "num_items_clipped": int(dataset.num_items_clipped),
        "patch_count_before_clip_summary": _summary(dataset.patch_count_before_clip_by_index),
        "patch_count_after_clip_summary": _summary(dataset.patch_count_after_clip_by_index),
        "skipped_items_count": int(sum(skip_reasons.values())),
        "skip_reasons": dict(skip_reasons),
    }
    audit_path = output_dir / f"raw_brainbert_pretrain_audit_fold_{int(args.fold_idx)}.json"
    with audit_path.open("w", encoding="utf-8") as fout:
        json.dump(_json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain fold-specific Raw-BrainBERT encoder for V3-HNC.")
    parser.add_argument("--raw-cache-path", required=True)
    parser.add_argument("--feature-cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--split-strategy", choices=["5fold"], default="5fold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ssl-subject-mode", choices=["success_train_only", "success_train_plus_all_failure"], default="success_train_only")
    parser.add_argument("--resample-sfreq", type=float, default=250.0)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--freq-min", type=float, default=1.0)
    parser.add_argument("--freq-max", type=float, default=125.0)
    parser.add_argument("--patch-time", type=int, default=4)
    parser.add_argument("--patch-freq", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-patches-per-item", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = pretrain_raw_brainbert_encoder(args)
    print(json.dumps(_json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
