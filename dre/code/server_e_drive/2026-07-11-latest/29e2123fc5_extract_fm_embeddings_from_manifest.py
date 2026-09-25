"""Extract frozen-FM window embeddings from a raw-window manifest.

`random_projection` is runnable debug only. `biot`, `cbramod`, and `labram`
must load a real external repo and checkpoint before any embeddings are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fm_baselines.cache_io import build_run_lookup, load_cache, load_raw_window_from_cache
from scripts.fm_baselines.extractors import build_extractor
from scripts.fm_baselines.preprocess import ensure_fixed_length, resample_1d_if_needed, robust_normalize_1d


def _embedding_row_offset(embeddings: List[np.ndarray]) -> int:
    return int(sum(part.shape[0] for part in embeddings))


def _manifest_rows_for_physical(physical_to_rows: Dict[str, List[Dict[str, Any]]], physical_id: str) -> List[Dict[str, Any]]:
    return physical_to_rows.get(str(physical_id), [])


def _flush_batch_safe(
    extractor: Any,
    batch: List[np.ndarray],
    batch_meta: List[Dict[str, Any]],
    batch_physical_ids: List[str],
    encoded_by_physical: Dict[str, np.ndarray],
    skipped: List[Dict[str, Any]],
    physical_to_rows: Dict[str, List[Dict[str, Any]]],
) -> None:
    if not batch:
        return
    try:
        encoded = np.asarray(extractor.encode_batch(np.stack(batch, axis=0)), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError(f"encoded shape {encoded.shape} does not match batch size {len(batch)}")
        for idx, physical_id in enumerate(batch_physical_ids):
            encoded_by_physical[str(physical_id)] = encoded[idx].astype(np.float32, copy=False)
    except Exception as exc:
        failed_phys_ids = set(str(item) for item in batch_physical_ids)
        for physical_id in failed_phys_ids:
            for meta_row in _manifest_rows_for_physical(physical_to_rows, physical_id):
                skipped.append({"row_id": meta_row.get("row_id", ""), "skip_reason": "batch encode failed: " + str(exc)})
    finally:
        batch.clear()
        batch_meta.clear()
        batch_physical_ids.clear()


def extract_fm_embeddings(
    *,
    window_cache_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    fm_model: str,
    target_sfreq: int,
    window_sec: float,
    batch_size: int,
    device: str,
    checkpoint_path: str | None = None,
    external_repo_path: str | None = None,
    extractor_kwargs: Mapping[str, Any] | None = None,
    allow_skipped_windows: bool = False,
    allow_empty_embeddings: bool = False,
    deduplicate_physical_windows: bool = True,
) -> Dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    cache = load_cache(window_cache_path)
    lookup = build_run_lookup(cache)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError("FM raw-window manifest is empty; cannot extract embeddings.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(fm_model, target_sfreq=int(target_sfreq), window_sec=float(window_sec), **dict(extractor_kwargs or {}))
    extractor.load(checkpoint_path, external_repo_path, device)
    target_len = int(round(float(target_sfreq) * float(window_sec)))
    dedup_enabled = bool(deduplicate_physical_windows) and "physical_window_id" in manifest.columns
    if dedup_enabled:
        encoding_manifest = manifest.drop_duplicates("physical_window_id", keep="first").reset_index(drop=True)
        physical_to_rows = {
            str(physical_id): [row.to_dict() for _, row in group.iterrows()]
            for physical_id, group in manifest.groupby("physical_window_id", sort=False)
        }
    else:
        manifest = manifest.copy()
        manifest["_synthetic_physical_window_id"] = [f"row:{idx}" for idx in range(len(manifest))]
        encoding_manifest = manifest
        physical_to_rows = {str(row["_synthetic_physical_window_id"]): [row.to_dict()] for _, row in manifest.iterrows()}
    embeddings_parts: List[np.ndarray] = []
    metadata_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    batch: List[np.ndarray] = []
    batch_meta: List[Dict[str, Any]] = []
    batch_physical_ids: List[str] = []
    encoded_by_physical: Dict[str, np.ndarray] = {}
    resampling_methods = set()

    for _, row in encoding_manifest.iterrows():
        physical_id = str(row["physical_window_id"] if dedup_enabled else row["_synthetic_physical_window_id"])
        try:
            raw, meta = load_raw_window_from_cache(lookup, row)
            raw = robust_normalize_1d(raw)
            raw, resampling_method = resample_1d_if_needed(raw, float(meta["sfreq"]), float(target_sfreq))
            raw = ensure_fixed_length(raw, target_len)
            resampling_methods.add(resampling_method)
            enriched = row.to_dict()
            enriched.update(
                {
                    "fm_model": str(fm_model),
                    "checkpoint_path": str(checkpoint_path or ""),
                    "external_repo_path": str(external_repo_path or ""),
                    "preprocessing_version": "median_iqr_clip8_resample_v1",
                }
            )
            batch.append(raw)
            batch_meta.append(enriched)
            batch_physical_ids.append(physical_id)
            if len(batch) >= int(batch_size):
                _flush_batch_safe(extractor, batch, batch_meta, batch_physical_ids, encoded_by_physical, skipped, physical_to_rows)
        except Exception as exc:
            for meta_row in _manifest_rows_for_physical(physical_to_rows, physical_id):
                skipped.append({"row_id": meta_row.get("row_id", ""), "skip_reason": str(exc)})

    _flush_batch_safe(extractor, batch, batch_meta, batch_physical_ids, encoded_by_physical, skipped, physical_to_rows)

    for _, row in manifest.iterrows():
        physical_id = str(row["physical_window_id"] if dedup_enabled else row["_synthetic_physical_window_id"])
        embedding = encoded_by_physical.get(physical_id)
        if embedding is None:
            continue
        out_row = row.to_dict()
        out_row.pop("_synthetic_physical_window_id", None)
        out_row.update(
            {
                "fm_model": str(fm_model),
                "checkpoint_path": str(checkpoint_path or ""),
                "external_repo_path": str(external_repo_path or ""),
                "preprocessing_version": "median_iqr_clip8_resample_v1",
                "embedding_row_idx": len(metadata_rows),
            }
        )
        embeddings_parts.append(embedding.reshape(1, -1).astype(np.float32, copy=False))
        metadata_rows.append(out_row)

    embeddings = np.vstack(embeddings_parts).astype(np.float32) if embeddings_parts else np.zeros((0, 0), dtype=np.float32)
    metadata_columns = [col for col in manifest.columns if col != "_synthetic_physical_window_id"] + [
        "fm_model",
        "checkpoint_path",
        "external_repo_path",
        "preprocessing_version",
        "embedding_row_idx",
    ]
    metadata = pd.DataFrame(metadata_rows, columns=metadata_columns)
    if embeddings.shape[0] != len(metadata_rows):
        raise RuntimeError(f"embedding/metadata row mismatch: {embeddings.shape[0]} embeddings vs {len(metadata_rows)} metadata rows")
    contiguous = (
        list(metadata["embedding_row_idx"].astype(int)) == list(range(len(metadata)))
        if not metadata.empty and "embedding_row_idx" in metadata.columns
        else embeddings.shape[0] == 0
    )
    skipped_df = pd.DataFrame(skipped, columns=["row_id", "skip_reason"])
    skipped_df.to_csv(out / "fm_window_skipped_windows.csv", index=False)
    skipped_by_reason = pd.Series([item["skip_reason"] for item in skipped]).value_counts().to_dict() if skipped else {}
    audit = {
        "fm_model": str(fm_model),
        "checkpoint_path": str(checkpoint_path or ""),
        "external_repo_path": str(external_repo_path or ""),
        "device": str(device),
        "target_sfreq": int(target_sfreq),
        "window_sec": float(window_sec),
        "batch_size": int(batch_size),
        "n_manifest_rows": int(len(manifest)),
        "n_embeddings_written": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else int(getattr(extractor, "embedding_dim", 0) or 0),
        "skipped_windows_count": int(len(skipped)),
        "skipped_windows_by_reason": skipped_by_reason,
        "allow_skipped_windows": bool(allow_skipped_windows),
        "allow_empty_embeddings": bool(allow_empty_embeddings),
        "deduplicate_physical_windows": bool(dedup_enabled),
        "n_unique_physical_windows": int(encoding_manifest.shape[0]),
        "encoder_forward_windows": int(len(encoded_by_physical)),
        "reused_embedding_rows": int(max(0, len(metadata_rows) - len(encoded_by_physical))),
        "metadata_embedding_alignment_check": {
            "metadata_rows": int(len(metadata_rows)),
            "embeddings_rows": int(embeddings.shape[0]),
            "metadata_rows_equals_embeddings_rows": bool(len(metadata_rows) == embeddings.shape[0]),
            "embedding_row_idx_contiguous": bool(contiguous),
            "pass": bool(len(metadata_rows) == embeddings.shape[0] and contiguous),
        },
        "preprocessing": {
            "median_iqr_normalization": True,
            "clip_range": [-8, 8],
            "resampling_method": sorted(resampling_methods),
        },
        "frozen_encoder": True,
        "fine_tuned": False,
        "adapter_tuned": False,
        "test_time_selection": False,
        "backbone_trainable_params": 0,
        "trainable_component": "logistic_l2_or_linear_svm_head_only",
        "seeg_adaptation": "single-channel SEEG contact mode; no scalp montage remapping",
        "warning": "FM is used only as a frozen comparison baseline, not as the proposed method.",
    }
    audit.update(getattr(extractor, "audit_info", {}))
    failure_reason = ""
    if embeddings.shape[0] == 0 and not allow_empty_embeddings:
        failure_reason = "No FM embeddings were written; inspect fm_window_skipped_windows.csv"
    elif skipped and not allow_skipped_windows:
        failure_reason = "FM embedding extraction skipped windows; inspect fm_window_skipped_windows.csv"
    audit["extraction_success"] = not bool(failure_reason)
    if failure_reason:
        audit["failure_reason"] = failure_reason
        (out / "fm_embedding_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        raise RuntimeError(failure_reason)
    np.save(out / "fm_window_embeddings.npy", embeddings)
    metadata.to_csv(out / "fm_window_metadata.csv", index=False)
    (out / "fm_embedding_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_cache_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fm_model", choices=["random_projection", "biot", "cbramod", "labram"], required=True)
    parser.add_argument("--external_repo_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--target_sfreq", type=int, default=200)
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow_skipped_windows", action="store_true")
    parser.add_argument("--allow_empty_embeddings", action="store_true")
    parser.add_argument("--deduplicate_physical_windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--biot_n_channels", default="auto")
    parser.add_argument("--biot_n_fft", type=int, default=200)
    parser.add_argument("--biot_hop_length", type=int, default=100)
    parser.add_argument("--biot_emb_size", type=int, default=256)
    parser.add_argument("--biot_heads", type=int, default=8)
    parser.add_argument("--biot_depth", type=int, default=4)
    parser.add_argument("--biot_n_channel_offset", type=int, default=0)
    parser.add_argument("--biot_strict_load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cbramod_points_per_patch", type=int, default=200)
    parser.add_argument("--cbramod_num_segments", default="auto")
    parser.add_argument("--cbramod_strict_load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--labram_model_name", default="labram_base_patch200_200")
    parser.add_argument("--labram_patch_size", type=int, default=200)
    parser.add_argument("--labram_num_channels", type=int, default=1)
    parser.add_argument("--labram_channel_order", default="single_seeg")
    parser.add_argument("--labram_strict_load", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    extract_fm_embeddings(
        window_cache_path=args.window_cache_path,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        fm_model=args.fm_model,
        external_repo_path=args.external_repo_path,
        checkpoint_path=args.checkpoint_path,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        batch_size=args.batch_size,
        device=args.device,
        extractor_kwargs={
            "biot_n_channels": args.biot_n_channels,
            "biot_n_fft": args.biot_n_fft,
            "biot_hop_length": args.biot_hop_length,
            "biot_emb_size": args.biot_emb_size,
            "biot_heads": args.biot_heads,
            "biot_depth": args.biot_depth,
            "biot_n_channel_offset": args.biot_n_channel_offset,
            "biot_strict_load": args.biot_strict_load,
            "cbramod_points_per_patch": args.cbramod_points_per_patch,
            "cbramod_num_segments": args.cbramod_num_segments,
            "cbramod_strict_load": args.cbramod_strict_load,
            "labram_model_name": args.labram_model_name,
            "labram_patch_size": args.labram_patch_size,
            "labram_num_channels": args.labram_num_channels,
            "labram_channel_order": args.labram_channel_order,
            "labram_strict_load": args.labram_strict_load,
        },
        allow_skipped_windows=args.allow_skipped_windows,
        allow_empty_embeddings=args.allow_empty_embeddings,
        deduplicate_physical_windows=args.deduplicate_physical_windows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
