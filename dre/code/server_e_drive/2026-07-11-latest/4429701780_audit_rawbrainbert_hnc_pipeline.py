from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def _check(result: dict[str, Any], name: str, ok: bool, detail: Any = None, *, critical: bool = True) -> None:
    entry = {"name": name, "ok": bool(ok), "critical": bool(critical), "detail": detail}
    result["checks"].append(entry)
    if critical and not ok:
        result["failed_checks"].append(entry)


def _embedding_exists(base_dir: Path, fold_idx: int) -> bool:
    base = base_dir / "embeddings" / f"rawbrainbert_patient_channel_embeddings_fold_{fold_idx}"
    return base.with_suffix(".parquet").exists() or base.with_suffix(".csv").exists()


def audit_pipeline(args: Any) -> dict[str, Any]:
    base_dir = Path(args.base_dir)
    result: dict[str, Any] = {
        "ok": False,
        "base_dir": str(base_dir),
        "checks": [],
        "failed_checks": [],
    }
    base_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = Path(args.feature_cache_path)
    raw_cache = Path(args.raw_cache_path)
    v3_output = Path(args.v3_output_dir)
    ledger_path = base_dir / "v3_oof_channel_ledger.csv"
    hnc_dir = base_dir / "hnc"
    hnc_audit_path = hnc_dir / "rawbrainbert_hnc_audit.json"
    hnc_audit = _read_json(hnc_audit_path)

    _check(result, "feature_cache_exists", feature_cache.exists(), str(feature_cache))
    _check(result, "raw_cache_exists", raw_cache.exists(), str(raw_cache))
    missing_v3 = [str(v3_output / f"test_channel_predictions_neuroez_v2_fold_{idx}.csv") for idx in range(1, 6) if not (v3_output / f"test_channel_predictions_neuroez_v2_fold_{idx}.csv").exists()]
    _check(result, "v3_fold_prediction_csvs_exist", not missing_v3, missing_v3)
    _check(result, "v3_oof_channel_ledger_exists", ledger_path.exists(), str(ledger_path))
    missing_encoders = [str(base_dir / "raw_brainbert" / f"raw_brainbert_encoder_fold_{idx}.pt") for idx in range(1, 6) if not (base_dir / "raw_brainbert" / f"raw_brainbert_encoder_fold_{idx}.pt").exists()]
    _check(result, "raw_brainbert_encoders_exist", not missing_encoders, missing_encoders)
    missing_embeddings = [idx for idx in range(1, 6) if not _embedding_exists(base_dir, idx)]
    _check(result, "rawbrainbert_embedding_files_exist", not missing_embeddings, missing_embeddings)
    corrected_paths = sorted(hnc_dir.glob("corrected_oof_rawbrainbert_hnc_*.csv"))
    _check(result, "corrected_ledgers_exist", bool(corrected_paths), [str(path) for path in corrected_paths])
    _check(result, "rawbrainbert_hnc_audit_exists", hnc_audit_path.exists(), str(hnc_audit_path))

    ledger_rows = None
    if ledger_path.exists():
        try:
            ledger_rows = len(pd.read_csv(ledger_path))
        except Exception as exc:
            _check(result, "v3_ledger_readable", False, f"{type(exc).__name__}: {exc}")
    if ledger_rows is not None and corrected_paths:
        row_mismatches = {}
        noncandidate_failures = {}
        for path in corrected_paths:
            frame = pd.read_csv(path)
            if len(frame) != ledger_rows:
                row_mismatches[path.name] = {"corrected_rows": int(len(frame)), "ledger_rows": int(ledger_rows)}
            required_cols = {"is_candidate", "final_logit", "patient_zscore_logit"}
            if required_cols.issubset(frame.columns):
                noncandidate = frame[~frame["is_candidate"].astype(bool)]
                if not noncandidate.empty and not np.allclose(
                    noncandidate["final_logit"].to_numpy(dtype=float),
                    noncandidate["patient_zscore_logit"].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=0.0,
                ):
                    noncandidate_failures[path.name] = int(len(noncandidate))
            else:
                noncandidate_failures[path.name] = f"missing columns {sorted(required_cols - set(frame.columns))}"
        _check(result, "corrected_row_count_equals_v3_ledger", not row_mismatches, row_mismatches)
        _check(result, "noncandidate_final_logit_equals_v3", not noncandidate_failures, noncandidate_failures)
    else:
        _check(result, "corrected_row_count_equals_v3_ledger", False, "ledger or corrected ledgers unavailable")
        _check(result, "noncandidate_final_logit_equals_v3", False, "ledger or corrected ledgers unavailable")

    leakage_failures: dict[str, Any] = {}
    for fold_idx in range(1, 6):
        audit_path = base_dir / "raw_brainbert" / f"raw_brainbert_pretrain_audit_fold_{fold_idx}.json"
        audit = _read_json(audit_path)
        leakage = audit.get("leakage_success_test_subjects_in_ssl", [])
        if leakage:
            leakage_failures[str(fold_idx)] = leakage
    _check(result, "success_test_subjects_not_in_ssl_train", not leakage_failures, leakage_failures)

    hnc_leakage = hnc_audit.get("leakage_checks", {})
    failure_train = {
        str(fold): detail.get("failure_subjects_in_hnc_train", [])
        for fold, detail in hnc_leakage.items()
        if detail.get("failure_subjects_in_hnc_train")
    }
    _check(result, "failure_subjects_not_in_hnc_supervised_train", not failure_train, failure_train)
    pca_audit = hnc_audit.get("pca_scaler_audit", {})
    pca_scope_ok = bool(pca_audit) and all(
        fold_audit.get("pca_fit_scope") == "per_fold_train_candidate_rows_only"
        for fold_audit in pca_audit.values()
        if isinstance(fold_audit, dict)
    )
    _check(result, "pca_scaler_fold_only_fit_audit_present", pca_scope_ok, pca_audit)
    embedding_column_counts = hnc_audit.get("embedding_columns_count_by_fold", {})
    embedding_columns_ok = bool(embedding_column_counts) and all(int(value) > 0 for value in embedding_column_counts.values())
    _check(result, "embedding_columns_count_by_fold_positive", embedding_columns_ok, embedding_column_counts)
    missing_embedding_fractions = hnc_audit.get("missing_embedding_fraction_by_fold", {})
    missing_fraction_ok = bool(missing_embedding_fractions) and all(
        float(value) < 0.2 for value in missing_embedding_fractions.values()
    )
    _check(result, "missing_embedding_fraction_by_fold_below_0_2", missing_fraction_ok, missing_embedding_fractions)
    result["ok"] = len(result["failed_checks"]) == 0
    with (base_dir / "pipeline_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False, sort_keys=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit V3-RawBrainBERT-HNC pipeline artifacts.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--feature-cache-path", required=True)
    parser.add_argument("--raw-cache-path", required=True)
    parser.add_argument("--v3-output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = audit_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
