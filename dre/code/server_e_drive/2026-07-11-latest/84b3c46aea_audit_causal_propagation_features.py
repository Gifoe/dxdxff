from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroez_c.causal_propagation_features import CAUSAL_FEATURE_NAMES
from neuroez_c.cane_path_cohort import build_sensitivity80_cohort
from neuroez_c.dual_view_data import _channel_names, _sample
from neuroez_c.protocol import STEP4B_STATIC_TOP20_FEATURES
from neuroez_c.raw_brainbert_data import normalize_channel_name
from data_factory import build_outer_splits


def _step4b_channel_frame(feature_cache_path: str | Path) -> tuple[pd.DataFrame, dict, list[dict]]:
    with Path(feature_cache_path).open("rb") as handle:
        payload = pickle.load(handle)
    names = list(payload.get("window_feature_names") or payload.get("feature_names") or [])
    missing = sorted(set(STEP4B_STATIC_TOP20_FEATURES) - set(names))
    if missing:
        raise ValueError(f"Feature cache is missing Step4B columns: {missing}")
    indices = [names.index(name) for name in STEP4B_STATIC_TOP20_FEATURES]
    values: dict[tuple[str, str], list[np.ndarray]] = {}
    for record in payload.get("run_records", []):
        subject = str(record.get("subject_id"))
        sample = _sample(record)
        features = np.asarray(sample.get("window_features"), dtype=np.float64)
        if features.ndim != 3 or features.shape[-1] != len(names):
            continue
        for channel_idx, channel in enumerate(_channel_names(record)):
            block = features[:, channel_idx, indices]
            finite_rows = np.isfinite(block).all(axis=1)
            if finite_rows.any():
                values.setdefault((subject, normalize_channel_name(channel)), []).append(np.median(block[finite_rows], axis=0))
    rows = []
    for (subject, channel), blocks in values.items():
        row = {"subject_id": subject, "channel_name": channel}
        row.update(dict(zip(STEP4B_STATIC_TOP20_FEATURES, np.median(np.stack(blocks), axis=0))))
        rows.append(row)
    patient_index = payload.get("patient_index", {})
    return pd.DataFrame(rows), patient_index, build_outer_splits(patient_index, n_splits=5, random_seed=42)


def audit_causal_features(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    feature_cache_path: str | Path | None = None,
    exclusion_manifest: str | Path | None = None,
) -> dict[str, object]:
    frame = pd.read_parquet(cache_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    valid_frame = frame[frame["cp_feature_valid"].astype(bool)].copy()
    patient = valid_frame.assign(center=valid_frame.subject_id.str.split(":", n=1).str[0].str.lower()).groupby(["subject_id", "center"], as_index=False)[list(CAUSAL_FEATURE_NAMES)].mean()
    x = patient[list(CAUSAL_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = patient["center"].to_numpy()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prediction = cross_val_predict(LogisticRegression(max_iter=2000), x, y, cv=splitter)
    center_audit = {
        "center_prediction_accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "labels": sorted(np.unique(y).tolist()),
        "confusion_matrix": confusion_matrix(y, prediction, labels=sorted(np.unique(y))).tolist(),
        "model_selection_use": False,
    }
    pd.DataFrame([center_audit]).drop(columns=["confusion_matrix", "labels"]).to_csv(out / "causal_feature_center_predictability.csv", index=False)
    redundancy_rows = []
    if feature_cache_path is not None and exclusion_manifest is not None:
        step4b, patient_index, outer_splits = _step4b_channel_frame(feature_cache_path)
        _, filtered_splits, _ = build_sensitivity80_cohort(patient_index, outer_splits, exclusion_manifest)
        merged = valid_frame.merge(step4b, on=["subject_id", "channel_name"], how="inner", validate="one_to_one")
        for split in filtered_splits:
            train = merged[merged.subject_id.isin(split["train_subjects"])]
            for causal_name in CAUSAL_FEATURE_NAMES:
                for step4b_name in STEP4B_STATIC_TOP20_FEATURES:
                    pair = train[[causal_name, step4b_name]].dropna()
                    rho = pair[causal_name].corr(pair[step4b_name], method="spearman") if len(pair) >= 3 else np.nan
                    redundancy_rows.append({
                        "outer_fold": split["fold_idx"], "n_outer_train_channel_rows": len(pair),
                        "causal_feature": causal_name, "step4b_feature": step4b_name,
                        "spearman_rho": rho, "abs_rho_ge_0_90": bool(np.isfinite(rho) and abs(rho) >= 0.90),
                    })
    redundancy = pd.DataFrame(redundancy_rows, columns=[
        "outer_fold", "n_outer_train_channel_rows", "causal_feature", "step4b_feature",
        "spearman_rho", "abs_rho_ge_0_90",
    ])
    redundancy.to_csv(out / "causal_feature_redundancy_by_fold.csv", index=False)
    quality = frame.groupby(frame.subject_id.str.split(":", n=1).str[0].str.lower()).agg(
        valid_channels=("cp_feature_valid", "sum"),
        mean_valid_seizure_count=("cp_valid_seizure_count", "mean"),
        mean_valid_window_fraction=("cp_valid_window_fraction", "mean"),
        mean_var_stability_spearman=("cp_mean_var_stability", "mean"),
    ).reset_index(names="center")
    pair_path = out / "causal_cache_cross_seizure_stability.csv"
    if pair_path.is_file():
        pairs = pd.read_csv(pair_path)
        stability = pairs.groupby(["center", "causal_feature"], as_index=False).agg(
            n_seizure_pairs=("spearman_rho", "size"),
            n_finite_seizure_pairs=("spearman_rho", "count"),
            mean_pairwise_seizure_rank_spearman=("spearman_rho", "mean"),
            median_pairwise_seizure_rank_spearman=("spearman_rho", "median"),
            mean_common_channels=("n_common_channels", "mean"),
        )
    else:
        stability = pd.DataFrame([
            {"center": center, "causal_feature": feature, "n_seizure_pairs": 0,
             "n_finite_seizure_pairs": 0, "mean_pairwise_seizure_rank_spearman": np.nan,
             "median_pairwise_seizure_rank_spearman": np.nan, "mean_common_channels": np.nan}
            for center in sorted(patient.center.unique()) for feature in CAUSAL_FEATURE_NAMES[:5]
        ])
    sixth = pd.DataFrame([
        {"center": center, "causal_feature": CAUSAL_FEATURE_NAMES[5], "n_seizure_pairs": 0,
         "n_finite_seizure_pairs": 0, "mean_pairwise_seizure_rank_spearman": np.nan,
         "median_pairwise_seizure_rank_spearman": np.nan, "mean_common_channels": np.nan,
         "stability_note": "feature_is_itself_a_cross_seizure_consistency_fraction"}
        for center in sorted(patient.center.unique())
    ])
    stability = pd.concat([stability, sixth], ignore_index=True).merge(quality, on="center", how="left")
    stability.to_csv(out / "causal_feature_stability.csv", index=False)
    center_audit["n_redundancy_rows"] = int(len(redundancy))
    center_audit["n_abs_rho_ge_0_90"] = int(redundancy.get("abs_rho_ge_0_90", pd.Series(dtype=bool)).sum())
    center_audit["redundancy_used_for_model_selection"] = False
    (out / "causal_feature_audit.json").write_text(json.dumps(center_audit, indent=2, sort_keys=True), encoding="utf-8")
    return center_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit domain predictability and stability of causal-propagation proxies.")
    parser.add_argument("--causal-cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-window-cache-path")
    parser.add_argument("--exclude-subjects-file")
    args = parser.parse_args()
    print(audit_causal_features(
        args.causal_cache_path, args.output_dir,
        feature_cache_path=args.feature_window_cache_path,
        exclusion_manifest=args.exclude_subjects_file,
    ))


if __name__ == "__main__":
    main()
