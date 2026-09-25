from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuroez_c.raw_brainbert import build_encoder_from_checkpoint_payload
from neuroez_c.raw_brainbert_data import (
    aggregate_record_embeddings,
    build_spectrogram_patches,
    collect_raw_record_channel_items,
    load_preproc,
    load_subject_splits,
    pool_hidden_states,
    read_cache_samples,
    subject_sets_for_fold,
    write_dataframe_with_parquet_fallback,
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


def extract_raw_brainbert_embeddings(args: Any) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_idx = int(args.fold_idx)
    encoder_dir = Path(args.encoder_dir)
    encoder_path = encoder_dir / f"raw_brainbert_encoder_fold_{fold_idx}.pt"
    preproc_path = encoder_dir / f"raw_brainbert_preproc_fold_{fold_idx}.pkl"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing Raw-BrainBERT encoder: {encoder_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"Missing Raw-BrainBERT preproc: {preproc_path}")

    preproc = load_preproc(preproc_path)
    device = _device(args.device)
    payload = torch.load(encoder_path, map_location=device)
    model = build_encoder_from_checkpoint_payload(payload).to(device)
    model.eval()

    splits, failure_subjects = load_subject_splits(
        args.feature_cache_path,
        split_strategy=args.split_strategy,
        n_splits=int(args.n_splits),
        random_seed=int(args.random_seed),
    )
    split_summary = subject_sets_for_fold(
        splits,
        fold_idx=fold_idx,
        failure_subjects=failure_subjects,
        ssl_subject_mode="success_train_only",
    )
    subject_ids = sorted(set(split_summary["success_train_subjects"]).union(split_summary["success_test_subjects"]))
    if bool(args.include_failure_embeddings):
        subject_ids = sorted(set(subject_ids).union(failure_subjects))

    samples, _ = read_cache_samples(args.raw_cache_path, outcome_subset="success_failure", include_raw_waveform=True)
    items, skip_reasons = collect_raw_record_channel_items(samples, subject_ids=subject_ids)
    record_rows: list[dict[str, Any]] = []
    skipped_items_count = int(sum(skip_reasons.values()))
    with torch.no_grad():
        for item in items:
            patch_batch = build_spectrogram_patches(
                item.raw,
                sfreq=item.sfreq,
                resample_sfreq=float(preproc["resample_sfreq"]),
                n_fft=int(preproc["n_fft"]),
                hop_length=int(preproc["hop_length"]),
                freq_min=float(preproc["freq_min"]),
                freq_max=float(preproc["freq_max"]),
                patch_time=int(preproc["patch_time"]),
                patch_freq=int(preproc["patch_freq"]),
                mean=float(preproc["mean"]),
                std=float(preproc["std"]),
                duration_sec=item.duration_sec,
            )
            if patch_batch.patches.shape[0] == 0:
                skipped_items_count += 1
                skip_reasons["empty_patch_sequence"] += 1
                continue
            patches = torch.as_tensor(patch_batch.patches[None, :, :], dtype=torch.float32, device=device)
            time_ids = torch.as_tensor(patch_batch.time_ids[None, :], dtype=torch.long, device=device)
            freq_ids = torch.as_tensor(patch_batch.freq_ids[None, :], dtype=torch.long, device=device)
            out = model(patches, time_ids, freq_ids, mask=None)
            hidden = out["hidden"][0, : patch_batch.patches.shape[0]].detach().cpu().numpy().astype(np.float32)
            pooled = pool_hidden_states(
                hidden,
                time_ids=patch_batch.time_ids,
                duration_sec=item.duration_sec,
                resample_sfreq=float(preproc["resample_sfreq"]),
                hop_length=int(preproc["hop_length"]),
                patch_time=int(preproc["patch_time"]),
            )
            record_rows.append(
                {
                    "subject_id": item.subject_id,
                    "run_id": item.run_id,
                    "sample_id": item.sample_id,
                    "channel_name": item.channel_name,
                    "emb_all": pooled["all"],
                    "emb_preictal": pooled["preictal"],
                    "emb_onset": pooled["onset"],
                    "emb_early": pooled["early"],
                }
            )
    if not record_rows:
        raise ValueError("No Raw-BrainBERT embeddings were extracted. Check raw cache and encoder/preproc files.")
    patient_channel_rows = aggregate_record_embeddings(record_rows, fold_idx=fold_idx)
    df = pd.DataFrame(patient_channel_rows)
    output_base = output_dir / f"rawbrainbert_patient_channel_embeddings_fold_{fold_idx}"
    output_path = write_dataframe_with_parquet_fallback(df, output_base)
    audit = {
        "fold_idx": fold_idx,
        "num_subjects_embedded": int(df["subject_id"].nunique()),
        "num_patient_channels": int(len(df)),
        "num_record_channels": int(len(record_rows)),
        "missing_raw_count": int(skip_reasons.get("missing_raw_waveform", 0)),
        "skipped_items_count": int(skipped_items_count),
        "skip_reasons": dict(skip_reasons),
        "embedding_dim": int(preproc["d_model"]),
        "pooling_modes": ["all", "preictal", "onset", "early", "std", "max"],
        "output_path": str(output_path),
        "subject_split": split_summary,
        "leakage_check": {
            "success_test_subjects_in_ssl_train": split_summary["leakage_success_test_subjects_in_ssl"],
            "include_failure_embeddings": bool(args.include_failure_embeddings),
        },
    }
    audit_path = output_dir / f"rawbrainbert_embedding_audit_fold_{fold_idx}.json"
    with audit_path.open("w", encoding="utf-8") as fout:
        json.dump(_json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Raw-BrainBERT patient-channel embeddings for V3-HNC.")
    parser.add_argument("--raw-cache-path", required=True)
    parser.add_argument("--feature-cache-path", required=True)
    parser.add_argument("--encoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--split-strategy", default="5fold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-failure-embeddings", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = extract_raw_brainbert_embeddings(args)
    print(json.dumps(_json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
