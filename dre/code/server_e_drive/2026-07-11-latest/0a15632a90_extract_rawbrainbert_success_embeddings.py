from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import add_label_encoding_columns, apply_allowed_subject_filter, json_safe
from neuroez_c.raw_brainbert import build_encoder_from_checkpoint_payload
from neuroez_c.raw_brainbert_data import (
    RawBrainBERTPatchDataset,
    aggregate_record_embeddings,
    audit_raw_cache_onset_timing,
    collate_patch_batch,
    collect_raw_record_channel_items,
    load_preproc,
    pool_hidden_states,
    read_cache_samples,
    write_dataframe_with_parquet_fallback,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract success-patient RawBrainBERT patient-channel embeddings.")
    parser.add_argument("--raw-cache-path", required=True)
    parser.add_argument("--v3-ledger", required=True)
    parser.add_argument("--encoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default=None)
    parser.add_argument("--include-failure-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
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
    success_subjects = sorted(ledger["subject_id"].astype(str).unique())
    outcome_subset = "success_failure" if bool(args.include_failure_embeddings) else "success"
    samples, _ = read_cache_samples(args.raw_cache_path, outcome_subset=outcome_subset, include_raw_waveform=True)
    selected_subjects = None if bool(args.include_failure_embeddings) else success_subjects
    selected_sample_set = samples if selected_subjects is None else [sample for sample in samples if str(sample.get("subject_id")) in set(selected_subjects)]
    onset_timing_audit = audit_raw_cache_onset_timing(selected_sample_set)
    items, skip_reasons = collect_raw_record_channel_items(samples, subject_ids=selected_subjects)
    if not items:
        raise ValueError("No raw record-channel items available for RawBrainBERT embedding extraction.")

    encoder_dir = Path(args.encoder_dir)
    checkpoint_path = encoder_dir / f"rawbrainbert_encoder_fold_{args.fold_idx}.pt"
    preproc_path = encoder_dir / f"rawbrainbert_preproc_fold_{args.fold_idx}.pkl"
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_encoder_from_checkpoint_payload(payload)
    preproc = load_preproc(preproc_path)
    device = _device(args.device)
    model.to(device)
    model.eval()
    dataset = RawBrainBERTPatchDataset(items, preproc, max_patches_per_item=args.max_patches_per_item)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_patch_batch)
    record_rows = []
    missing_raw_count = 0
    with torch.no_grad():
        for batch in loader:
            if not batch["items"]:
                missing_raw_count += 1
                continue
            patches = batch["patches"].to(device)
            time_ids = batch["time_ids"].to(device)
            freq_ids = batch["freq_ids"].to(device)
            valid = batch["valid_mask"].to(device)
            out = model(patches, time_ids, freq_ids, mask=None, key_padding_mask=~valid)
            hidden = out["hidden"].detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            time_np = time_ids.detach().cpu().numpy()
            for row_idx, item in enumerate(batch["items"]):
                n_valid = int(valid_np[row_idx].sum())
                if n_valid <= 0:
                    missing_raw_count += 1
                    continue
                pooled = pool_hidden_states(
                    hidden[row_idx, :n_valid, :],
                    time_ids=time_np[row_idx, :n_valid],
                    duration_sec=item.duration_sec,
                    resample_sfreq=float(preproc["resample_sfreq"]),
                    hop_length=int(preproc["hop_length"]),
                    patch_time=int(preproc["patch_time"]),
                )
                record_rows.append(
                    {
                        "subject_id": item.subject_id,
                        "channel_name": item.channel_name,
                        "run_id": item.run_id,
                        "emb_all": pooled["all"],
                        "emb_preictal": pooled["preictal"],
                        "emb_onset": pooled["onset"],
                        "emb_early": pooled["early"],
                    }
                )

    patient_rows = aggregate_record_embeddings(record_rows, fold_idx=args.fold_idx)
    df = pd.DataFrame(patient_rows)
    output_path = write_dataframe_with_parquet_fallback(
        df,
        output_dir / f"rawbrainbert_patient_channel_embeddings_fold_{args.fold_idx}",
    )
    audit = {
        **filter_audit,
        "fold_idx": int(args.fold_idx),
        "num_subjects": int(df["subject_id"].nunique()) if not df.empty else 0,
        "num_patient_channels": int(len(df)),
        "num_record_channels": int(len(record_rows)),
        "missing_raw_count": int(missing_raw_count + sum(skip_reasons.values())),
        "embedding_dim": int(len([col for col in df.columns if str(col).startswith("rawbb_all_")])) if not df.empty else 0,
        "failure_embeddings_included": bool(args.include_failure_embeddings),
        "label_encoding_mode": str(ledger["label_encoding_mode"].iloc[0]) if "label_encoding_mode" in ledger.columns and not ledger.empty else None,
        "leakage_check": {
            "default_excludes_failure_embeddings": not bool(args.include_failure_embeddings),
        },
        "output_path": str(output_path),
        "skip_reasons": dict(Counter(skip_reasons)),
        **onset_timing_audit,
    }
    with (output_dir / f"rawbrainbert_embedding_audit_fold_{args.fold_idx}.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(audit), fout, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
