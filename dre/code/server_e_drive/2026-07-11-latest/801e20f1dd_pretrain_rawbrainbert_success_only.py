from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import add_label_encoding_columns, apply_allowed_subject_filter, json_safe
from neuroez_c.raw_brainbert import RawBrainBERTEncoder, make_random_patch_mask, masked_patch_loss
from neuroez_c.raw_brainbert_data import (
    RawBrainBERTPatchDataset,
    audit_raw_cache_onset_timing,
    collate_patch_batch,
    collect_raw_record_channel_items,
    estimate_spectrogram_normalizer,
    load_subject_splits,
    read_cache_samples,
    save_preproc,
    subject_sets_for_fold,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain RawBrainBERT SSL on success-train subjects only.")
    parser.add_argument("--raw-cache-path", required=True)
    parser.add_argument("--feature-cache-path", required=True)
    parser.add_argument("--v3-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--split-strategy", default="5fold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default=None)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ssl-split-source", choices=["v3_ledger", "feature_cache"], default="v3_ledger")
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-patches-per-item", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(args.v3_ledger)
    ledger = add_label_encoding_columns(ledger, label_encoding_mode=args.label_encoding_mode)
    ledger, filter_audit = apply_allowed_subject_filter(
        ledger,
        allowed_subjects_ledger=args.allowed_subjects_ledger,
        allowed_subjects_file=args.allowed_subjects_file,
        require_n_patients=args.require_n_patients,
    )
    allowed_subjects = set(ledger["subject_id"].astype(str).unique())

    if args.ssl_split_source == "v3_ledger":
        # Use V3 OOF ledger for fold split — strict alignment
        fold_col = "fold_idx" if "fold_idx" in ledger.columns else "fold_id"
        all_ledger_subjects = sorted(ledger["subject_id"].astype(str).unique())
        v3_test_subjects = sorted(
            ledger[ledger[fold_col].astype(int) == int(args.fold_idx)]["subject_id"].astype(str).unique()
        )
        ssl_subjects = sorted(set(all_ledger_subjects) - set(v3_test_subjects))
        n_success_train_subjects = len(ssl_subjects)
        n_success_test_subjects = len(v3_test_subjects)
        leakage_v3 = sorted(set(v3_test_subjects) & set(ssl_subjects))
        leakage_success_test_subjects_in_ssl = leakage_v3
        if leakage_v3:
            raise ValueError(f"V3 test subjects leaked into SSL: {leakage_v3}")
        n_failure_subjects_seen = 0
        failure_used_in_ssl = False
        ssl_audit = {
            "ssl_split_source": "v3_ledger",
            "strict_v3_fold_alignment": True,
            "fold_idx": int(args.fold_idx),
            "n_all_ledger_subjects": len(all_ledger_subjects),
            "n_v3_test_subjects": n_success_test_subjects,
            "n_success_train_subjects": n_success_train_subjects,
            "n_success_test_subjects": n_success_test_subjects,
            "n_ssl_subjects": len(ssl_subjects),
            "n_failure_subjects_seen": n_failure_subjects_seen,
            "ledger_subjects_sha1": hashlib.sha1(",".join(sorted(all_ledger_subjects)).encode()).hexdigest(),
            "v3_test_subjects_sha1": hashlib.sha1(",".join(sorted(v3_test_subjects)).encode()).hexdigest(),
            "ssl_subjects_sha1": hashlib.sha1(",".join(sorted(ssl_subjects)).encode()).hexdigest(),
            "leakage_v3_test_subjects_in_ssl": leakage_v3,
            "leakage_success_test_subjects_in_ssl": leakage_success_test_subjects_in_ssl,
            "failure_used_in_ssl": failure_used_in_ssl,
            "allowed_subject_filter_used": bool(filter_audit.get("allowed_subject_filter_used", False)),
            "require_n_patients": int(args.require_n_patients or 0),
            "feature_cache_path": str(args.feature_cache_path),
            "v3_ledger": str(args.v3_ledger),
        }
    else:
        # Legacy: feature-cache-based split
        splits, failure_subjects = load_subject_splits(
            args.feature_cache_path, split_strategy=args.split_strategy,
            n_splits=args.n_splits, random_seed=args.random_seed,
        )
        subject_sets = subject_sets_for_fold(splits, fold_idx=args.fold_idx, failure_subjects=failure_subjects)
        ssl_subjects = sorted(set(subject_sets["ssl_subjects"]).intersection(allowed_subjects))
        leakage = sorted(set(subject_sets["success_test_subjects"]).intersection(ssl_subjects))
        if leakage:
            raise ValueError(f"success_test subjects leaked into SSL: {leakage}")
        failure_used = sorted(set(failure_subjects).intersection(ssl_subjects))
        if failure_used:
            raise ValueError(f"failure subjects leaked into SSL: {failure_used}")
        ssl_audit = {
            "ssl_split_source": "feature_cache",
            "strict_v3_fold_alignment": False,
            "fold_idx": int(args.fold_idx),
            "n_all_ledger_subjects": len(allowed_subjects),
            "n_ssl_subjects": len(ssl_subjects),
            "leakage_v3_test_subjects_in_ssl": leakage,
            "failure_used_in_ssl": len(failure_used) > 0,
            "feature_cache_path": str(args.feature_cache_path),
        }

    # Common audit variables — defined after both branches resolved
    n_success_train_subjects_common = ssl_audit.get("n_success_train_subjects", len(ssl_subjects))
    n_success_test_subjects_common = ssl_audit.get("n_success_test_subjects", 0)
    n_failure_subjects_seen_common = ssl_audit.get("n_failure_subjects_seen", 0)
    leakage_success_test_subjects_in_ssl_common = ssl_audit.get("leakage_success_test_subjects_in_ssl", [])
    failure_used_in_ssl_common = ssl_audit.get("failure_used_in_ssl", False)
    n_ssl_subjects_common = len(ssl_subjects)

    samples, _ = read_cache_samples(args.raw_cache_path, outcome_subset="success_failure", include_raw_waveform=True)
    ssl_samples = [sample for sample in samples if str(sample.get("subject_id")) in ssl_subjects]
    onset_timing_audit = audit_raw_cache_onset_timing(ssl_samples)
    missing_raw = sum(1 for sample in ssl_samples if "raw_waveform" not in sample)
    if missing_raw:
        raise ValueError(f"Selected SSL samples missing raw_waveform: {missing_raw}")
    items, skip_reasons = collect_raw_record_channel_items(samples, subject_ids=ssl_subjects)
    if args.max_train_items and args.max_train_items > 0:
        items = items[: int(args.max_train_items)]
    if not items:
        raise ValueError("No RawBrainBERT SSL record-channel items were available after success-train filtering.")

    # Hard check: items must only come from ssl_subjects
    item_subjects = set(str(getattr(item, "subject_id", "")) for item in items)
    extra = item_subjects - set(ssl_subjects)
    if extra:
        raise ValueError(f"RawBrainBERT items contain subjects outside SSL set: {sorted(extra)[:10]}")
    ssl_audit.update({
        "raw_cache_total_patients_loaded": len({str(s.get("subject_id")) for s in samples}),
        "rawbrainbert_item_subjects_count": len(item_subjects),
        "rawbrainbert_item_subjects_subset_of_ssl_subjects": item_subjects <= set(ssl_subjects),
    })

    norm = estimate_spectrogram_normalizer(
        items,
        resample_sfreq=args.resample_sfreq,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        max_items=args.max_train_items,
    )
    preproc = {
        **norm,
        "resample_sfreq": float(args.resample_sfreq),
        "n_fft": int(args.n_fft),
        "hop_length": int(args.hop_length),
        "freq_min": float(args.freq_min),
        "freq_max": float(args.freq_max),
        "patch_time": int(args.patch_time),
        "patch_freq": int(args.patch_freq),
    }
    dataset = RawBrainBERTPatchDataset(items, preproc, max_patches_per_item=args.max_patches_per_item, random_seed=args.random_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_patch_batch)
    device = _device(args.device)
    model = RawBrainBERTEncoder(
        patch_dim=int(args.patch_time) * int(args.patch_freq),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
    losses: list[dict[str, float | int]] = []
    best_loss = None
    for epoch in range(1, int(args.epochs) + 1):
        epoch_losses: list[float] = []
        model.train()
        for batch in loader:
            if not batch["items"]:
                continue
            patches = batch["patches"].to(device)
            time_ids = batch["time_ids"].to(device)
            freq_ids = batch["freq_ids"].to(device)
            valid = batch["valid_mask"].to(device)
            mask = make_random_patch_mask(valid, args.mask_ratio)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(args.amp) and device.type == "cuda"):
                out = model(patches, time_ids, freq_ids, mask=mask, key_padding_mask=~valid)
                loss = masked_patch_loss(out["reconstruction"], patches, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        best_loss = mean_loss if best_loss is None else min(float(best_loss), mean_loss)
        losses.append({"epoch": epoch, "loss": mean_loss})

    torch.save(
        {"model_config": model.config.to_dict(), "model_state_dict": model.state_dict()},
        output_dir / f"rawbrainbert_encoder_fold_{args.fold_idx}.pt",
    )
    save_preproc(output_dir / f"rawbrainbert_preproc_fold_{args.fold_idx}.pkl", preproc)
    pd.DataFrame(losses).to_csv(output_dir / f"rawbrainbert_loss_curve_fold_{args.fold_idx}.csv", index=False)
    audit = {
        **filter_audit,
        "fold_idx": int(args.fold_idx),
        "n_success_train_subjects": n_success_train_subjects_common,
        "n_success_test_subjects": n_success_test_subjects_common,
        "n_failure_subjects_seen": n_failure_subjects_seen_common,
        "failure_used_in_ssl": failure_used_in_ssl_common,
        "leakage_success_test_subjects_in_ssl": leakage_success_test_subjects_in_ssl_common,
        "n_ssl_subjects": n_ssl_subjects_common,
        "ssl_subjects_sha1": hashlib.sha1("\n".join(ssl_subjects).encode("utf-8")).hexdigest(),
        "n_record_channels": len(items),
        "raw_cache_path": str(args.raw_cache_path),
        "feature_cache_path": str(args.feature_cache_path),
        "v3_ledger": str(args.v3_ledger),
        "label_encoding_mode": str(ledger["label_encoding_mode"].iloc[0]) if "label_encoding_mode" in ledger.columns and not ledger.empty else None,
        "spectrogram_shape_stats": norm,
        "mask_ratio": float(args.mask_ratio),
        "best_loss": best_loss,
        "final_loss": losses[-1]["loss"] if losses else None,
        "skipped_items_count": int(sum(skip_reasons.values())),
        "skip_reasons": dict(skip_reasons),
        "normalizer_fit_subjects_count": len({item.subject_id for item in items}),
        "normalizer_fit_on_train_only": True,
        **onset_timing_audit,
        **ssl_audit,
    }
    with (output_dir / f"rawbrainbert_pretrain_audit_fold_{args.fold_idx}.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
