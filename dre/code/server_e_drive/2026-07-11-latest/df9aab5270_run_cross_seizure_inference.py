"""Run frozen PRQ/BCR forward passes after an ID-only seizure subset is selected."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from .core import (
    CROSS_SEIZURE_INFERENCE_VERSION, FORMAL_THRESHOLD_SOURCE, SEEDS, build_subset_manifest, discover_checkpoint, discover_threshold_file,
    load_cache, load_fit_subjects_by_seed, load_model, load_subjects_and_folds_by_seed, make_prediction_rows, patient_seizure_map,
    prepare_output_root, read_thresholds, selected_samples, sha256_file, stable_hash, validate_prediction_frame, write_json,
    resolve_model_normalizer,
)


def _threshold_root(root: str | Path, seed: int, model: str) -> Path:
    root = Path(root) / f"seed_{seed}"
    return root / ("p2_q10" if model == "PRQ-Net" else "bcr_boundary_coverage")


def _completion_path(root: Path, seed: int, fold: int, mode: str, repeat_id: int) -> Path:
    return root / f"seed_{seed}" / f"fold_{fold}" / mode / f"repeat_{repeat_id}" / "completion.json"


def run(args: argparse.Namespace, *, mode: str) -> pd.DataFrame:
    if mode not in {"one", "two", "all"}:
        raise ValueError(f"Unsupported seizure mode {mode}")
    paths = prepare_output_root(args.output_root)
    subjects, fold_maps, manifest = load_subjects_and_folds_by_seed(
        args.protocol_root,
        args.seeds,
    )
    fit_maps = load_fit_subjects_by_seed(manifest, args.seeds)
    records, patient_index = load_cache(args.cache_path)
    seizure_map = patient_seizure_map(records, subjects)
    subset = build_subset_manifest(seizure_map, fold_maps, args.seeds, args.repeats)
    subset_path = paths["manifests"] / "seizure_subset_manifest.csv"
    if subset_path.is_file():
        existing = pd.read_csv(subset_path)
        if stable_hash(existing.to_dict("records")) != stable_hash(subset.to_dict("records")):
            raise RuntimeError("Existing seizure subset manifest differs from deterministic protocol; use a new output root")
    else:
        subset.to_csv(subset_path, index=False)
    rows: list[pd.DataFrame] = []
    repeats = [0] if mode == "all" else list(range(args.repeats))
    for seed in args.seeds:
        fold_map = fold_maps[int(seed)]
        prq_threshold_path = discover_threshold_file(
            _threshold_root(args.prq_root, seed, "PRQ-Net"),
            seed,
            model="PRQ-Net",
        )
        bcr_threshold_path = discover_threshold_file(
            _threshold_root(args.bcr_root, seed, "BCR-Net"),
            seed,
            model="BCR-Net",
        )
        prq_thresholds = read_thresholds(prq_threshold_path, model="PRQ-Net")
        bcr_thresholds = read_thresholds(bcr_threshold_path, model="BCR-Net")
        # CDEL threshold is only read from its frozen reference root; it is never selected here.
        cdel_threshold_path = discover_threshold_file(
            Path(args.reference_root) / f"seed_{seed}" / "cdel",
            seed,
            model="CDEL",
        )
        cdel_thresholds = read_thresholds(cdel_threshold_path, model="CDEL")
        for fold, test_subjects in sorted(fold_map.items()):
            prq_path = discover_checkpoint(args.prq_root, seed, fold, "PRQ-Net")
            bcr_path = discover_checkpoint(args.bcr_root, seed, fold, "BCR-Net")
            fit_subjects = fit_maps[int(seed)][int(fold)]
            prq_normalizer = resolve_model_normalizer(
                prq_path,
                model_kind="PRQ-Net",
                run_records=records,
                fit_subjects=fit_subjects,
            )
            bcr_normalizer = resolve_model_normalizer(
                bcr_path,
                model_kind="BCR-Net",
                run_records=records,
                fit_subjects=fit_subjects,
            )
            prq = load_model(
                prq_path,
                device=args.device,
                model_kind="PRQ-Net",
                normalizer=prq_normalizer,
            )
            bcr = load_model(
                bcr_path,
                device=args.device,
                model_kind="BCR-Net",
                normalizer=bcr_normalizer,
            )
            for repeat_id in repeats:
                completion = _completion_path(paths["predictions"], seed, fold, mode, repeat_id)
                output = completion.parent / "channel_predictions.csv"
                signature = stable_hash({
                    "inference_version": CROSS_SEIZURE_INFERENCE_VERSION,
                    "seed": seed,
                    "fold": fold,
                    "mode": mode,
                    "repeat": repeat_id,
                    "prq": str(prq_path),
                    "prq_sha256": sha256_file(prq_path),
                    "bcr": str(bcr_path),
                    "bcr_sha256": sha256_file(bcr_path),
                    "subset": subset_path.read_text(encoding="utf-8"),
                })
                if args.resume and completion.is_file() and output.is_file():
                    prior = json.loads(completion.read_text(encoding="utf-8"))
                    if prior.get("signature") != signature:
                        raise RuntimeError(f"Refusing to reuse mismatched completed inference: {completion}")
                    rows.append(pd.read_csv(output))
                    continue
                fold_rows = []
                for subject in sorted(test_subjects):
                    selection = subset[(subset.training_seed == seed) & (subset.outer_fold == fold) & (subset.subject_id == subject) & (subset.seizure_mode == mode) & (subset.repeat_id == repeat_id)]
                    if len(selection) != 1:
                        raise RuntimeError(f"Missing deterministic subset row for {seed}/{fold}/{subject}/{mode}/{repeat_id}")
                    item = selection.iloc[0]
                    selected = json.loads(item.selected_seizure_ids)
                    if not selected:
                        # One-seizure always has a row. Two-seizure unavailable patients are retained in manifests,
                        # but intentionally excluded only from the two-seizure available analysis.
                        continue
                    from .core import infer_patient
                    prq_result, _ = infer_patient(prq, selected_samples(records, subject, selected), patient_index, device=args.device, amp=args.amp)
                    bcr_result, _ = infer_patient(bcr, selected_samples(records, subject, selected), patient_index, device=args.device, amp=args.amp)
                    fold_rows.append(make_prediction_rows(
                        prq_result, bcr_result, seed=seed, fold=fold, mode=mode, repeat_id=repeat_id,
                        selected_ids=selected, total=int(item.total_available_seizures),
                        thresholds={"prq": prq_thresholds[fold], "bcr": bcr_thresholds[fold], "cdel": cdel_thresholds[fold]},
                    ))
                if not fold_rows:
                    raise RuntimeError(f"No eligible patients for seed={seed}, fold={fold}, mode={mode}, repeat={repeat_id}")
                merged = pd.concat(fold_rows, ignore_index=True)
                validate_prediction_frame(merged)
                output.parent.mkdir(parents=True, exist_ok=True)
                merged.to_csv(output, index=False)
                write_json(completion, {"status": "complete", "signature": signature, "threshold_source": FORMAL_THRESHOLD_SOURCE, "n_patients": int(merged.subject_id.nunique())})
                rows.append(merged)
            del prq, bcr
            gc.collect()
            try:
                import torch
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                pass
    result = pd.concat(rows, ignore_index=True)
    index = paths["predictions"] / f"cross_seizure_channel_predictions_{mode}.csv"
    result.to_csv(index, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("prq_root", "bcr_root", "cache_path", "protocol_root", "reference_root", "output_root"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seizure-mode", choices=("one", "two", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args, mode=args.seizure_mode)
    print({"status": "passed", "rows": len(result), "mode": args.seizure_mode})


if __name__ == "__main__":
    main()
