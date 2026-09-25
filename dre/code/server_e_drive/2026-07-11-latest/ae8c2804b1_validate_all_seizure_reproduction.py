"""Hard gate comparing all-seizure re-inference against frozen formal OOF files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .core import (
    MODELS,
    compare_all_seizure,
    discover_threshold_file,
    prepare_output_root,
    read_thresholds,
    write_json,
)


def _branch_root(args: argparse.Namespace, seed: int, model: str) -> Path:
    if model == "PRQ-Net":
        return Path(args.prq_root) / f"seed_{seed}" / "p2_q10"
    if model == "BCR-Net":
        return Path(args.bcr_root) / f"seed_{seed}" / "bcr_boundary_coverage"
    return Path(args.reference_root) / f"seed_{seed}" / "cdel"


def _load_reference(args: argparse.Namespace, seed: int, model: str) -> tuple[pd.DataFrame, list[Path]]:
    branch = _branch_root(args, seed, model)
    if model == "PRQ-Net":
        aggregate = branch / "p23_channel_predictions.csv"
        fold_pattern = "fold_{fold}/test_channel_predictions_neuroez_v2_fold_{fold}.csv"
    elif model == "BCR-Net":
        aggregate = branch / "oof_channel_ledger.csv"
        fold_pattern = "test_channel_predictions_neuroez_v2_fold_{fold}.csv"
    else:
        aggregate = branch / "ledgers" / "oof_channel_predictions.csv"
        fold_pattern = ""
    if aggregate.is_file():
        paths = [aggregate]
    elif fold_pattern:
        paths = [
            branch / fold_pattern.format(fold=fold)
            for fold in range(1, 6)
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Formal {model} reference ledgers are incomplete: {missing}"
            )
    else:
        raise FileNotFoundError(f"Formal {model} reference ledger is missing: {aggregate}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    reference = _standardize_reference(frame, seed=seed, model=model)
    threshold_path = discover_threshold_file(branch, seed, model=model)
    thresholds = read_thresholds(threshold_path, model=model)
    reference["threshold"] = reference["outer_fold"].astype(int).map(thresholds)
    if reference["threshold"].isna().any():
        raise ValueError(f"Missing frozen threshold for {model}, seed={seed}")
    return reference, paths


def _standardize_reference(frame: pd.DataFrame, *, seed: int, model: str) -> pd.DataFrame:
    """Normalize only column aliases; never derive a new threshold or prediction."""
    aliases = {
        "outer_fold": ("outer_fold", "fold_idx", "fold"),
        "label_nez": ("label_nez", "true_nez", "nez_label"),
        "center": ("center",),
    }
    out = frame.copy()
    out["training_seed"] = int(seed)
    for target, options in aliases.items():
        source = next((name for name in options if name in out.columns), None)
        if source is None:
            raise ValueError(f"Formal reference lacks required {target} field for {model}")
        out[target] = out[source]
    if model == "PRQ-Net":
        score = next((name for name in ("prq_score_nez", "score_nez", "direct_score_nez") if name in out.columns), None)
    elif model == "BCR-Net":
        score = next((name for name in ("bcr_score_nez", "score_nez", "score_nez_probability") if name in out.columns), None)
    else:
        score = next((name for name in ("cdel_score_nez", "fused_score_nez", "score_nez") if name in out.columns), None)
    if score is None:
        raise ValueError(f"Formal reference lacks NEZ probability for {model}")
    out["score_nez"] = pd.to_numeric(out[score], errors="raise")
    return out


def validate(args: argparse.Namespace) -> dict:
    paths = prepare_output_root(args.output_root)
    current = pd.read_csv(paths["predictions"] / "cross_seizure_channel_predictions_all.csv")
    rows = []
    for model in MODELS:
        for seed in sorted(current.training_seed.unique()):
            reference, reference_paths = _load_reference(args, int(seed), model)
            for fold in sorted(current.outer_fold.unique()):
                new = current[current.training_seed.eq(seed) & current.outer_fold.eq(fold)].copy()
                ref_fold = reference[reference.outer_fold.eq(fold)].copy()
                comparison = compare_all_seizure(new, ref_fold, model=model)
                comparison["training_seed"] = int(seed)
                comparison["outer_fold"] = int(fold)
                comparison["reference_path"] = ";".join(map(str, reference_paths))
                rows.append(comparison)
    frame = pd.DataFrame(rows)
    frame.to_csv(paths["audit"] / "all_seizure_reproduction_by_seed_fold.csv", index=False)
    summary = {"status": "PASS" if frame.status.eq("PASS").all() else "FAIL", "n_checks": len(frame), "checks": frame.to_dict("records")}
    write_json(paths["audit"] / "all_seizure_reproduction_summary.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("All-seizure reproduction gate failed; one/two seizure inference is forbidden")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("prq_root", "bcr_root", "reference_root", "output_root"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    args = parser.parse_args()
    print(validate(args))


if __name__ == "__main__":
    main()
