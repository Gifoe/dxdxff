"""Build label-blind absolute and invariant-score patient context, fit-only PCA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SCORE_NAMES = ["q0", "log1p_C", "evidence_mean", "evidence_std", "evidence_min", "evidence_max",
               "near_operating_0p25z", "near_operating_0p5z", "score_z_q05", "score_z_q10",
               "score_z_q25", "score_z_q50", "score_z_q75", "score_z_q90", "score_z_q95",
               "score_z_skew", "upper_tail_gap", "lower_tail_gap"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def raw_descriptor(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float)
    values[~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        low, high = np.nanquantile(values, [0.25, 0.75], axis=0)
    return np.r_[median, high - low]


def score_descriptor(row: dict) -> np.ndarray:
    a = np.asarray(row["a"], float)
    e = np.asarray(row["evidence"], float)
    mean, std = float(a.mean()), max(float(a.std()), 1e-8)
    z = (a - mean) / std
    t_nez = float(row["formal_threshold_nez"])
    operating_z = (math.log((1 - t_nez) / t_nez) - mean) / std
    q05, q10, q25, q50, q75, q90, q95 = np.quantile(z, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    output = np.array([row["q0"], math.log1p(len(a)), e.mean(), e.std(), e.min(), e.max(),
                       np.mean(np.abs(z - operating_z) <= 0.25),
                       np.mean(np.abs(z - operating_z) <= 0.5),
                       q05, q10, q25, q50, q75, q90, q95, np.mean(z ** 3), q95 - q75, q25 - q05], dtype=float)
    if len(output) != len(SCORE_NAMES) or not np.isfinite(output).all():
        raise RuntimeError("Invalid label-blind score descriptor")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "feature-table", "feature-manifest", "target-dir", "protocol-lock", "output-dir"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if sha(args.feature_table) != lock["feature_table_sha256"] or sha(args.feature_manifest) != lock["feature_manifest_sha256"]:
        raise RuntimeError("Historical raw feature source hash mismatch")
    target_audit = json.loads((args.target_dir / "K_ORACLE_SUMMARY.json").read_text(encoding="utf-8"))
    if target_audit["status"] != "PASS" or target_audit["new_lock_sha256"] != sha(args.protocol_lock):
        raise RuntimeError("Optimal-K target provenance failed")
    table = pd.read_pickle(args.feature_table)
    sys.path.insert(0, str(args.source_root.resolve()))
    from task1_baselines.patient_controls import feature_columns
    columns = feature_columns(table)
    if columns != json.loads(args.feature_manifest.read_text(encoding="utf-8"))["feature_names"] or len(columns) != 88:
        raise RuntimeError("Feature axes/order differ from historical B0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_audits = []
    for fold in range(1, 6):
        source = args.target_dir / f"fold{fold}_PATIENTS_PRIVATE.pkl"
        expected_hash = target_audit["folds"][fold - 1]["private_patient_table_sha256"]
        if sha(source) != expected_hash:
            raise RuntimeError(f"Fold {fold} patient target archive changed")
        with source.open("rb") as stream:
            rows = pickle.load(stream)
        fit_ids = {r["patient_id"] for r in rows if r["role"] == "fit"}
        val_ids = {r["patient_id"] for r in rows if r["role"] == "validation"}
        if fit_ids & val_ids or len(fit_ids) not in (50, 51, 52) or len(val_ids) != 13:
            raise RuntimeError("Fit/validation patient partition changed")
        sub = table[table.subject_id.astype(str).isin(fit_ids | val_ids)]
        grouped = {str(pid): group for pid, group in sub.groupby(sub.subject_id.astype(str), sort=False)}
        if set(grouped) != fit_ids | val_ids:
            raise RuntimeError("Raw feature coverage incomplete")
        raw = []
        for row in rows:
            patient_table = grouped[row["patient_id"]]
            if (len(patient_table) != len(row["a"])
                    or set(patient_table.channel_name.astype(str)) != set(row["channels"])):
                raise RuntimeError("Patient raw channel keys do not align with frozen B0 scores")
            raw.append(raw_descriptor(patient_table, columns))
            row["z_score"] = score_descriptor(row)
        x = np.stack(raw)
        fit_mask = np.asarray([r["role"] == "fit" for r in rows])
        empty_fit_columns = np.isnan(x[fit_mask]).all(axis=0)
        imp = imputer()
        filled_fit = imp.fit_transform(x[fit_mask])
        filled_all = imp.transform(x)
        if filled_fit.shape[1] != 176:
            # On older scikit-learn, preserve all-empty columns with zero fill.
            if filled_fit.shape[1] + int(empty_fit_columns.sum()) != 176:
                raise RuntimeError("Raw descriptor imputer unexpectedly changed dimension")
            x[:, empty_fit_columns] = 0
            imp = imputer()
            filled_fit = imp.fit_transform(x[fit_mask])
            filled_all = imp.transform(x)
        scaler = StandardScaler().fit(filled_fit)
        standardized_fit = scaler.transform(filled_fit)
        pca = PCA(n_components=8, svd_solver="full").fit(standardized_fit)
        absolute = pca.transform(scaler.transform(filled_all))
        if absolute.shape != (len(rows), 8) or not np.isfinite(absolute).all():
            raise RuntimeError("Invalid 8D absolute context")
        for j, row in enumerate(rows):
            row["z_abs"] = absolute[j]
        pipeline_file = args.output_dir / f"fold{fold}_CONTEXT_TRANSFORM_PRIVATE.joblib"
        joblib.dump({"imputer": imp, "scaler": scaler, "pca": pca,
                     "empty_fit_columns": empty_fit_columns, "feature_columns": columns}, pipeline_file)
        out = args.output_dir / f"fold{fold}_CONTEXT_PRIVATE.pkl"
        with out.open("wb") as stream:
            pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
        fold_audits.append({"fold": fold, "fit_patients": len(fit_ids), "validation_patients": len(val_ids),
                            "raw_axes": 88, "raw_summary_dimensions": 176, "score_dimensions": len(SCORE_NAMES),
                            "absolute_pca_dimensions": 8, "all_missing_raw_summary_axes_in_fit": int(empty_fit_columns.sum()),
                            "fit_missing_raw_summary_cells": int(np.isnan(x[fit_mask]).sum()),
                            "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
                            "pca_total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
                            "private_context_sha256": sha(out),
                            "private_transform_sha256": sha(pipeline_file)})
        print(f"PATIENT_CONTEXT_FOLD_PASS fold={fold} pca_variance={fold_audits[-1]['pca_total_explained_variance_ratio']:.4f}", flush=True)
    audit = {"status": "PASS", "protocol_lock_sha256": sha(args.protocol_lock),
             "pca_training_scope": "outer-fit only, fit-only imputer and StandardScaler",
             "raw_feature_axes_aligned_across_models": True,
             "score_descriptor_names": SCORE_NAMES, "folds": fold_audits}
    (args.output_dir / "PATIENT_CONTEXT_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("PATIENT_CONTEXT_PASS", flush=True)


if __name__ == "__main__":
    main()
