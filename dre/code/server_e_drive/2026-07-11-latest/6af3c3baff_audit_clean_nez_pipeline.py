from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import LABEL_ENCODING_COLUMNS, json_safe


FORBIDDEN_FEATURE_NAMES = {
    "center",
    "center_id",
    "subject_id",
    "patient_id",
    "fold_idx",
    "outcome_group",
    "surgery_success",
    "true_ez",
    "true_nez",
    "true_ez_count",
    "predicted_by_oracle_k",
    "predicted_by_kcal",
}


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
    if not critical and not ok:
        result["warnings"].append(entry)


def _ledger_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(len(pd.read_csv(path)))


def _subjects(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["subject_id"])
    return set(frame["subject_id"].astype(str).unique())


def _patient_count(path: Path) -> int | None:
    subjects = _subjects(path)
    return None if subjects is None else len(subjects)


def _label_encoding_checks(path: Path, expected_mode: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path)
    missing = [col for col in LABEL_ENCODING_COLUMNS if col not in df.columns]
    detail: dict[str, Any] = {"exists": True, "missing_columns": missing}
    if missing:
        return detail
    modes = sorted(set(df["label_encoding_mode"].astype(str)))
    detail["modes"] = modes
    mode = modes[0] if len(modes) == 1 else None
    detail["mode"] = mode
    detail["expected_mode"] = expected_mode
    true_ez = pd.to_numeric(df["true_ez"], errors="coerce").fillna(-1).astype(int)
    true_nez = pd.to_numeric(df["true_nez"], errors="coerce").fillna(-1).astype(int)
    clinical_ez = pd.to_numeric(df["clinical_true_ez"], errors="coerce").fillna(-2).astype(int)
    clinical_nez = pd.to_numeric(df["clinical_true_nez"], errors="coerce").fillna(-2).astype(int)
    raw = pd.to_numeric(df["raw_binary_label"], errors="coerce").fillna(-3).astype(int)
    ez_value = pd.to_numeric(df["ez_label_value"], errors="coerce").fillna(-1).astype(int)
    nez_value = pd.to_numeric(df["nez_label_value"], errors="coerce").fillna(-1).astype(int)
    detail["true_ez_equals_clinical_true_ez"] = bool((true_ez == clinical_ez).all())
    detail["true_nez_equals_clinical_true_nez"] = bool((true_nez == clinical_nez).all())
    if mode == "ez1":
        detail["raw_binary_matches_mode"] = bool((raw == clinical_ez).all() and ez_value.eq(1).all() and nez_value.eq(0).all())
    elif mode == "ez0":
        detail["raw_binary_matches_mode"] = bool((raw == clinical_nez).all() and ez_value.eq(0).all() and nez_value.eq(1).all())
    else:
        detail["raw_binary_matches_mode"] = False
    detail["mode_matches_expected"] = expected_mode is None or mode == expected_mode
    return detail


def audit_pipeline(args: Any) -> dict[str, Any]:
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"ok": False, "base_dir": str(base_dir), "checks": [], "failed_checks": [], "warnings": []}
    require_n = getattr(args, "require_n_patients", None)
    allow_ridge = bool(getattr(args, "allow_ridge_baseline", False))
    requested_candidate_rule = getattr(args, "candidate_rule", None)
    expected_label_mode = getattr(args, "label_encoding_mode", None)

    v3_ledger = base_dir / "v3_clean_nez_ledger.csv"
    distance_ledger = base_dir / "clean_nez_distance" / "clean_nez_distance_ledger.csv"
    settopo_ledger = base_dir / "settopo" / "corrected_suspicious_ledger_alpha0.10.csv"
    kcal_ledger = base_dir / "kcal" / "settopo_ledger_with_kcal.csv"
    eval_summary = base_dir / "eval" / "metrics_summary.csv"
    eval_patient = base_dir / "eval" / "patient_level_metrics.csv"
    v3_audit = _read_json(base_dir / "v3_clean_nez_ledger_audit.json")
    v3_allowed_audit = _read_json(base_dir / "v3_allowed_subject_filter_audit.json")
    anchor_audit = _read_json(base_dir / "clean_nez_distance" / "clean_nez_anchor_audit.json")
    settopo_audit = _read_json(base_dir / "settopo" / "settopo_training_audit.json")
    kcal_audit = _read_json(base_dir / "kcal" / "kcal_training_audit.json")
    eval_audit = _read_json(base_dir / "eval" / "evaluation_audit.json")

    _check(result, "v3_clean_nez_ledger_exists", v3_ledger.exists(), str(v3_ledger))
    _check(result, "clean_nez_distance_ledger_exists", distance_ledger.exists(), str(distance_ledger))
    _check(result, "settopo_ledger_exists", settopo_ledger.exists(), str(settopo_ledger))
    _check(result, "kcal_ledger_exists", kcal_ledger.exists(), str(kcal_ledger))
    _check(result, "eval_metrics_summary_exists", eval_summary.exists(), str(eval_summary))
    _check(result, "eval_patient_level_metrics_exists", eval_patient.exists(), str(eval_patient))
    rawbb_required_missing: list[str] = []
    rawbb_pretrain_audits: dict[int, dict[str, Any]] = {}
    rawbb_embedding_audits: dict[int, dict[str, Any]] = {}
    for fold_idx in range(1, 6):
        pretrain_audit_path = base_dir / "rawbrainbert" / f"rawbrainbert_pretrain_audit_fold_{fold_idx}.json"
        encoder_path = base_dir / "rawbrainbert" / f"rawbrainbert_encoder_fold_{fold_idx}.pt"
        preproc_path = base_dir / "rawbrainbert" / f"rawbrainbert_preproc_fold_{fold_idx}.pkl"
        embedding_audit_path = base_dir / "embeddings" / f"rawbrainbert_embedding_audit_fold_{fold_idx}.json"
        for path in (pretrain_audit_path, encoder_path, preproc_path, embedding_audit_path):
            if not path.exists():
                rawbb_required_missing.append(str(path))
        rawbb_pretrain_audits[fold_idx] = _read_json(pretrain_audit_path)
        rawbb_embedding_audits[fold_idx] = _read_json(embedding_audit_path)
    _check(
        result,
        "rawbrainbert_required_artifacts_fold_1_to_5_exist",
        not rawbb_required_missing,
        rawbb_required_missing,
    )
    # ---- V3 pre-split and LZU filter checks ----
    drop_ez_enabled = v3_allowed_audit.get("drop_high_ez_fraction_lzu_enabled", True)
    n_dropped_ez = int(v3_allowed_audit.get("n_patients_dropped_by_high_ez_fraction", 0))
    _check(result, "drop_high_ez_fraction_lzu_disabled", not drop_ez_enabled,
           {"enabled": drop_ez_enabled})
    _check(result, "zero_patients_dropped_by_high_ez_fraction", n_dropped_ez == 0,
           {"n_dropped": n_dropped_ez})
    _check(result, "v3_split_built_after_allowed_filter",
           v3_allowed_audit.get("v3_split_built_after_allowed_filter", False) is True)

    n_v3_after = int(v3_allowed_audit.get("v3_n_patients_after_allowed_filter", 0))
    n_v3_before = int(v3_allowed_audit.get("v3_n_patients_before_allowed_filter", 0))
    _check(result, "v3_strict_90_patients_at_split_time", n_v3_after == 90,
           {"n_patients_after_filter": n_v3_after, "n_patients_before_filter": n_v3_before})

    if n_v3_after == 82:
        _check(result, "cohort_is_82_not_90_drop_high_ez_suspect",
               False, "Detected 82 patients. This usually means drop_high_ez_fraction_lzu was enabled.",
               critical=True)
    # 96→90 is OK when split was built after allowed filter
    # Fail only if split_built_after_allowed_filter is not true OR count is not 90
    split_ok = (
        v3_allowed_audit.get("v3_split_built_after_allowed_filter", False) is True
        and n_v3_after == 90
    )
    _check(result, "v3_strict_90_with_split_after_filter", split_ok,
           {"split_after_filter": v3_allowed_audit.get("v3_split_built_after_allowed_filter"),
            "n_after": n_v3_after, "n_before": n_v3_before})

    # ---- SSL fold alignment checks ----
    for fold_idx in range(1, 6):
        pa = rawbb_pretrain_audits.get(fold_idx, {})
        ssl_source = pa.get("ssl_split_source", "unknown")
        _check(result, f"ssl_fold{fold_idx}_split_source_v3_ledger",
               ssl_source == "v3_ledger", ssl_source)
        _check(result, f"ssl_fold{fold_idx}_strict_alignment",
               pa.get("strict_v3_fold_alignment", False) is True)
        _check(result, f"ssl_fold{fold_idx}_n_ledger_subjects_90",
               int(pa.get("n_all_ledger_subjects", 0)) == 90,
               int(pa.get("n_all_ledger_subjects", 0)))
        leakage = pa.get("leakage_v3_test_subjects_in_ssl", [])
        _check(result, f"ssl_fold{fold_idx}_no_test_leakage",
               not leakage, leakage[:10] if leakage else [])
        _check(result, f"ssl_fold{fold_idx}_no_failure_in_ssl",
               not pa.get("failure_used_in_ssl", True))
        _check(result, f"ssl_fold{fold_idx}_item_subjects_in_ssl",
               pa.get("rawbrainbert_item_subjects_subset_of_ssl_subjects", False) is True)

    if require_n:
        counts = {
            "v3": _patient_count(v3_ledger),
            "distance": _patient_count(distance_ledger),
            "settopo": _patient_count(settopo_ledger),
            "kcal": _patient_count(kcal_ledger),
        }
        _check(result, "required_patient_count_matches_all_ledgers", all(v == int(require_n) for v in counts.values()), {"counts": counts, "required": int(require_n)})

    label_details = {
        "v3": _label_encoding_checks(v3_ledger, expected_label_mode),
        "distance": _label_encoding_checks(distance_ledger, expected_label_mode),
        "settopo": _label_encoding_checks(settopo_ledger, expected_label_mode),
        "kcal": _label_encoding_checks(kcal_ledger, expected_label_mode),
    }
    _check(
        result,
        "ledgers_have_required_label_encoding_columns",
        all(not detail.get("missing_columns") for detail in label_details.values() if detail.get("exists")),
        label_details,
    )
    _check(
        result,
        "true_labels_equal_clinical_labels",
        all(
            detail.get("true_ez_equals_clinical_true_ez") is True and detail.get("true_nez_equals_clinical_true_nez") is True
            for detail in label_details.values()
            if detail.get("exists") and not detail.get("missing_columns")
        ),
        label_details,
    )
    _check(
        result,
        "raw_binary_label_matches_label_encoding_mode",
        all(
            detail.get("raw_binary_matches_mode") is True and detail.get("mode_matches_expected") is True
            for detail in label_details.values()
            if detail.get("exists") and not detail.get("missing_columns")
        ),
        label_details,
    )

    v3_rows = _ledger_rows(v3_ledger)
    distance_rows = _ledger_rows(distance_ledger)
    settopo_rows = _ledger_rows(settopo_ledger)
    kcal_rows = _ledger_rows(kcal_ledger)
    subject_sets = {
        "v3": _subjects(v3_ledger),
        "distance": _subjects(distance_ledger),
        "settopo": _subjects(settopo_ledger),
        "kcal": _subjects(kcal_ledger),
    }
    if all(value is not None for value in subject_sets.values()):
        reference = subject_sets["v3"] or set()
        mismatches = {
            key: {
                "missing_vs_v3": sorted(reference - (value or set()))[:20],
                "extra_vs_v3": sorted((value or set()) - reference)[:20],
            }
            for key, value in subject_sets.items()
            if value != reference
        }
        _check(result, "subject_sets_equal_across_ledgers", not mismatches, mismatches)
    else:
        _check(result, "subject_sets_equal_across_ledgers", False, "one or more ledgers missing")
    _check(
        result,
        "final_ledger_row_count_equals_v3",
        v3_rows is not None and kcal_rows is not None and v3_rows == kcal_rows,
        {"v3_rows": v3_rows, "kcal_rows": kcal_rows},
    )
    _check(
        result,
        "distance_and_settopo_rows_preserved",
        v3_rows is not None and distance_rows == v3_rows and settopo_rows == v3_rows,
        {"v3_rows": v3_rows, "distance_rows": distance_rows, "settopo_rows": settopo_rows},
    )
    _check(
        result,
        "pseudo_anchor_selection_did_not_use_true_labels",
        anchor_audit.get("whether_true_label_used_for_anchor_selection") is False,
        anchor_audit.get("whether_true_label_used_for_anchor_selection"),
    )
    leakage_failures = {}
    for fold_idx, audit in rawbb_pretrain_audits.items():
        if audit.get("failure_used_in_ssl") or audit.get("leakage_success_test_subjects_in_ssl"):
            leakage_failures[f"fold_{fold_idx}"] = {
                "failure_used_in_ssl": audit.get("failure_used_in_ssl"),
                "leakage_success_test_subjects_in_ssl": audit.get("leakage_success_test_subjects_in_ssl"),
            }
    _check(result, "rawbrainbert_ssl_no_failure_or_test_leakage", not leakage_failures, leakage_failures)
    onset_unverified = {
        f"pretrain_fold_{fold_idx}": audit.get("rawbb_temporal_pooling_basis", "unknown")
        for fold_idx, audit in rawbb_pretrain_audits.items()
        if audit and audit.get("onset_timing_verified") is not True
    }
    onset_unverified.update(
        {
            f"embedding_fold_{fold_idx}": audit.get("rawbb_temporal_pooling_basis", "unknown")
            for fold_idx, audit in rawbb_embedding_audits.items()
            if audit and audit.get("onset_timing_verified") is not True
        }
    )
    _check(
        result,
        "rawbrainbert_onset_timing_verified_or_marked_unverified",
        not any(audit.get("onset_timing_verified") is True for audit in rawbb_pretrain_audits.values() if audit)
        or not onset_unverified,
        onset_unverified,
        critical=False,
    )
    _check(result, "settopo_model_type_real_or_allowed", settopo_audit.get("model_type") == "real_settopo" or allow_ridge, settopo_audit.get("model_type"))
    _check(result, "settopo_real_settopo_used", bool(settopo_audit.get("real_settopo_used")) or allow_ridge, settopo_audit.get("real_settopo_used"))
    forbidden_features = sorted(FORBIDDEN_FEATURE_NAMES.intersection(set(settopo_audit.get("feature_columns", []))))
    _check(result, "settopo_forbidden_features_absent", not forbidden_features, forbidden_features)
    _check(result, "settopo_no_center_features", bool(settopo_audit.get("no_center_features")), settopo_audit.get("feature_columns"))
    _check(result, "settopo_no_outcome_features", bool(settopo_audit.get("no_outcome_features")), settopo_audit.get("feature_columns"))
    _check(result, "settopo_true_ez_count_not_used_as_feature", bool(settopo_audit.get("true_ez_count_not_used_as_feature")), settopo_audit.get("feature_columns"))
    _check(result, "settopo_uses_clinical_true_ez_for_ranking", settopo_audit.get("settopo_uses_clinical_true_ez_for_ranking") is True, settopo_audit)
    _check(result, "settopo_uses_clinical_true_nez_for_clean_loss", settopo_audit.get("settopo_uses_clinical_true_nez_for_clean_loss") is True, settopo_audit)
    _check(result, "settopo_raw_binary_label_not_used_as_clinical_target", settopo_audit.get("raw_binary_label_not_used_as_clinical_target") is True, settopo_audit)
    _check(result, "alpha_protocol_no_test_selection", settopo_audit.get("alpha_protocol_no_test_selection") is True, settopo_audit)
    candidate_audit = settopo_audit.get("candidate_audit", {})
    if requested_candidate_rule:
        _check(result, "candidate_rule_matches_requested", candidate_audit.get("candidate_rule") == requested_candidate_rule, {"requested": requested_candidate_rule, "actual": candidate_audit.get("candidate_rule")})
    _check(result, "candidate_count_positive", int(candidate_audit.get("candidate_count", 0)) > 0, candidate_audit)
    cand_frac = candidate_audit.get("candidate_fraction")
    if cand_frac is not None:
        _check(result, "candidate_fraction_reasonable", 0.05 <= float(cand_frac) <= 0.90, cand_frac, critical=False)
    _check(
        result,
        "kcal_true_ez_count_not_in_inference_features",
        "true_ez_count" not in set(kcal_audit.get("inference_feature_columns", [])),
        kcal_audit.get("inference_feature_columns"),
    )
    kcal_forbidden = sorted(FORBIDDEN_FEATURE_NAMES.intersection(set(kcal_audit.get("inference_feature_columns", []))))
    _check(result, "kcal_forbidden_inference_features_absent", not kcal_forbidden and not kcal_audit.get("forbidden_inference_feature_intersection"), kcal_audit.get("inference_feature_columns"))
    _check(result, "kcal_actual_model_matches_requested", kcal_audit.get("actual_model_type") == kcal_audit.get("requested_model"), {"actual": kcal_audit.get("actual_model_type"), "requested": kcal_audit.get("requested_model")})
    _check(result, "kcal_k_true_source_is_clinical_true_ez", kcal_audit.get("k_true_source") == "clinical_true_ez", kcal_audit)
    _check(result, "kcal_raw_binary_label_not_used_for_k_true", kcal_audit.get("raw_binary_label_not_used_for_k_true") is True, kcal_audit)
    main_alpha = settopo_audit.get("main_alpha")
    train_alpha = settopo_audit.get("train_alpha")
    kcal_input_alpha = kcal_audit.get("input_alpha")
    kcal_alpha_values = []
    if kcal_ledger.exists() and "alpha" in pd.read_csv(kcal_ledger, nrows=1).columns:
        kcal_alpha_values = sorted(float(value) for value in pd.to_numeric(pd.read_csv(kcal_ledger, usecols=["alpha"])["alpha"], errors="coerce").dropna().unique())
    alpha_ok = (
        main_alpha is not None
        and train_alpha is not None
        and abs(float(main_alpha) - float(train_alpha)) <= 1e-12
        and kcal_input_alpha is not None
        and abs(float(kcal_input_alpha) - float(main_alpha)) <= 1e-12
        and (not kcal_alpha_values or (len(kcal_alpha_values) == 1 and abs(float(kcal_alpha_values[0]) - float(main_alpha)) <= 1e-12))
    )
    _check(
        result,
        "kcal_input_alpha_matches_settopo_main_alpha",
        alpha_ok,
        {"kcal_input_alpha": kcal_input_alpha, "kcal_ledger_alpha_values": kcal_alpha_values, "main_alpha": main_alpha, "train_alpha": train_alpha},
    )
    fold_bad = {
        key: detail
        for key, detail in (kcal_audit.get("fold_audits") or {}).items()
        if int(detail.get("train_patients", 0)) <= 0 or int(detail.get("test_patients", 0)) <= 0
    }
    _check(result, "kcal_each_fold_has_train_and_test_patients", not fold_bad, fold_bad)
    _check(result, "evaluation_macro_formula_recorded", eval_audit.get("macro_f1_formula") == "0.5*(ez_f1+nez_f1)", eval_audit)
    _check(result, "evaluation_v3_subject_set_equals_main", eval_audit.get("v3_subject_set_equals_main") is True, eval_audit)
    _check(result, "evaluation_target_is_clinical_true_ez", eval_audit.get("evaluation_target") == "clinical_true_ez", eval_audit)
    _check(result, "evaluation_raw_binary_label_not_used_as_target", eval_audit.get("raw_binary_label_used_as_metric_target") is False, eval_audit)
    if eval_patient.exists():
        patient_df = pd.read_csv(eval_patient)
        if not patient_df.empty and {"patient_macro_f1", "patient_ez_f1", "patient_nez_f1"}.issubset(patient_df.columns):
            formula_diff = (
                patient_df["patient_macro_f1"].astype(float)
                - 0.5 * (patient_df["patient_ez_f1"].astype(float) + patient_df["patient_nez_f1"].astype(float))
            ).abs()
            _check(result, "patient_macro_f1_formula_matches", bool((formula_diff <= 1e-8).all()), float(formula_diff.max()))
            all_equal = bool(np.allclose(patient_df["patient_ez_f1"].astype(float), patient_df["patient_nez_f1"].astype(float)))
            _check(result, "patient_nez_f1_not_globally_identical_to_ez_f1", not all_equal, "all patient EZ-F1 and NEZ-F1 are identical", critical=False)
    result["ok"] = len(result["failed_checks"]) == 0
    with (base_dir / "clean_nez_pipeline_audit.json").open("w", encoding="utf-8") as fout:
        json.dump(json_safe(result), fout, indent=2, ensure_ascii=False, sort_keys=True)
    errors_path = base_dir / "clean_nez_pipeline_audit_errors.txt"
    errors = [f"{item['name']}: {item['detail']}" for item in result["failed_checks"]]
    errors_path.write_text("\n".join(errors), encoding="utf-8")
    warnings_path = base_dir / "clean_nez_pipeline_audit_warnings.txt"
    warnings = [f"{item['name']}: {item['detail']}" for item in result["warnings"]]
    warnings_path.write_text("\n".join(warnings), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CleanNEZ RawBB SetTopo KCal pipeline artifacts.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    parser.add_argument("--candidate-rule", default=None)
    parser.add_argument("--allow-ridge-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = audit_pipeline(args)
    print(json.dumps(json_safe(result), ensure_ascii=False, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
