from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from neuroez_c.cane_path_cohort import (
    EXPECTED_CENTER_COUNTS,
    build_sensitivity80_cohort,
    read_exclusion_manifest,
)
from neuroez_c.protocol import (
    SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL,
    STEP4B_STATIC_TOP20_FEATURES,
    assert_fixed_all90_protocol,
)
from run_neuroez_c import validate_cane_path_cp_args
from tests.cane_path_test_helpers import cane_args


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "task1_sensitivity80_exclude_suspected_10.csv"


def _all90():
    excluded = read_exclusion_manifest(MANIFEST)
    ids = [row["subject_id"] for row in excluded]
    needed = {"hup": 36, "lzu": 28, "multicenter": 15, "pediatric": 11}
    counts = Counter(value.split(":", 1)[0].lower() for value in ids)
    for center, total in needed.items():
        ids.extend(f"{center}:generated_{idx:03d}" for idx in range(total - counts[center]))
    return {subject: {"center": subject.split(":", 1)[0]} for subject in ids}


def _folds(subjects):
    values = sorted(subjects)
    tests = [values[index::5] for index in range(5)]
    return [{"fold_idx": index + 1, "train_subjects": sorted(set(values) - set(test)), "test_subjects": test} for index, test in enumerate(tests)]


def _formal_args(*extra):
    args = cane_args(
        "--fixed-all90-protocol-name", SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL,
        "--enforce_fixed_all90_protocol", *extra,
    )
    # Config-only protocol checks intentionally do not claim that a causal
    # cache audit exists. Formal non-dry runs fail closed on the real path.
    args.causal_cache_audit_path = ""
    return args


def test_manifest_contains_exactly_ten_unique_subjects() -> None:
    rows = read_exclusion_manifest(MANIFEST)
    assert len(rows) == 10 and len({row["subject_id"].casefold() for row in rows}) == 10


def test_manifest_status_is_explicitly_unconfirmed_posthoc() -> None:
    rows = read_exclusion_manifest(MANIFEST)
    assert {row["status"] for row in rows} == {"unconfirmed_posthoc_sensitivity_exclusion"}
    assert {row["source"] for row in rows} == {"posthoc_oracle_review"}


def test_sensitivity_filter_produces_exactly_80() -> None:
    patients = _all90()
    filtered, _, audit = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    assert len(filtered) == 80 and audit["n_excluded_patients"] == 10


def test_sensitivity_filter_center_counts() -> None:
    patients = _all90()
    _, _, audit = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    assert audit["center_counts"] == EXPECTED_CENTER_COUNTS


def test_sensitivity_filter_preserves_original_manifest_mapping() -> None:
    patients = _all90(); original = set(patients)
    build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    assert set(patients) == original and len(patients) == 90


def test_sensitivity_filter_preserves_outer_fold_membership() -> None:
    patients = _all90(); folds = _folds(patients)
    _, filtered, _ = build_sensitivity80_cohort(patients, folds, MANIFEST)
    for before, after in zip(folds, filtered):
        assert set(after["test_subjects"]).issubset(before["test_subjects"])


def test_sensitivity_filter_case_insensitive_exact_match(tmp_path: Path) -> None:
    patients = _all90(); target = "PEDIATRIC:d008"
    patients[target] = patients.pop("pediatric:D008")
    filtered, _, _ = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    assert target not in filtered


def test_sensitivity_filter_fails_when_one_id_missing() -> None:
    patients = _all90(); patients.pop("pediatric:D008"); patients["pediatric:replacement"] = {}
    with pytest.raises(ValueError, match="not found"):
        build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)


def test_sensitivity_filter_fails_on_fold_leakage() -> None:
    patients = _all90(); folds = _folds(patients)
    retained = next(value for value in folds[0]["test_subjects"] if value not in {row["subject_id"] for row in read_exclusion_manifest(MANIFEST)})
    folds[0]["train_subjects"].append(retained)
    with pytest.raises(ValueError, match="leakage"):
        build_sensitivity80_cohort(patients, folds, MANIFEST)


def test_protocol_accepts_valid_sensitivity80_configuration() -> None:
    patients = _all90(); filtered, folds, _ = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    audit = assert_fixed_all90_protocol(_formal_args(), filtered, folds, STEP4B_STATIC_TOP20_FEATURES)
    assert audit["n_patients"] == 80 and audit["protocol_name"] == SENSITIVITY80_CANE_PATH_CP_NEZ_PROTOCOL


@pytest.mark.parametrize("flag", ["--use_two_expert_router", "--use_hard_topk_loss", "--use_a9v8_lcbo"])
def test_validator_rejects_forbidden_historical_modules(flag: str) -> None:
    args = cane_args()
    setattr(args, flag.removeprefix("--"), True)
    with pytest.raises(ValueError, match="conflicts"):
        validate_cane_path_cp_args(args)


def test_validator_rejects_wrong_positive_label() -> None:
    args = cane_args(); args.positive_label = "ez"
    with pytest.raises(ValueError, match="NEZ=1"):
        validate_cane_path_cp_args(args)


def test_validator_rejects_wrong_inner_fold_count() -> None:
    args = cane_args(); args.inner_splits = 3
    with pytest.raises(ValueError, match="four inner"):
        validate_cane_path_cp_args(args)


def test_validator_outer_only_does_not_require_inner_crossfit() -> None:
    args = cane_args("--cane-direct-outer-only")
    args.inner_splits = 1
    args.inner_split_seed = 999
    validate_cane_path_cp_args(args)


def test_protocol_outer_only_records_fixed_threshold_without_path_head() -> None:
    patients = _all90()
    filtered, folds, _ = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    args = _formal_args("--cane-direct-outer-only")
    audit = assert_fixed_all90_protocol(args, filtered, folds, STEP4B_STATIC_TOP20_FEATURES)
    assert audit["training_mode"] == "direct_outer_only"
    assert audit["inner_crossfit_used"] is False and audit["inner_splits"] == 0
    assert audit["patient_adaptive_threshold_used"] is False
    assert audit["decision_rule"] == "fixed_nez_probability_threshold"


def test_protocol_outer_only_f1_selection_is_recorded() -> None:
    patients = _all90()
    filtered, folds, _ = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    args = _formal_args(
        "--cane-direct-outer-only", "--cane-selection-objective", "f1",
        "--early_stop_metric", "patient_macro_f1",
    )
    audit = assert_fixed_all90_protocol(args, filtered, folds, STEP4B_STATIC_TOP20_FEATURES)
    assert audit["selection_objective"] == "f1"
    assert audit["threshold_source"] == "fold_validation_macro_f1"


def test_protocol_rejects_center_as_model_feature() -> None:
    patients = _all90(); filtered, folds, _ = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    args = _formal_args(); args.model_input_features = "power,center_id"
    with pytest.raises(ValueError, match="center_id"):
        assert_fixed_all90_protocol(args, filtered, folds, STEP4B_STATIC_TOP20_FEATURES)


def test_protocol_rejects_missing_step4b_feature() -> None:
    patients = _all90(); filtered, folds, _ = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    args = _formal_args(); args.dry_run_config_only = False; args.causal_cache_audit_path = ""
    with pytest.raises(ValueError, match="missing Step4B"):
        assert_fixed_all90_protocol(args, filtered, folds, STEP4B_STATIC_TOP20_FEATURES[:-1])


def test_formal_p2_requires_and_accepts_passed_causal_audit(tmp_path: Path) -> None:
    causal_audit = tmp_path / "causal.json"
    causal_audit.write_text(json.dumps({
        "status": "passed", "n_patients": 80, "patient_match_rate": 1.0,
        "channel_match_rate": 0.96, "window_match_rate": 0.91,
        "feature_valid_rate": 0.96, "valid_rate_by_center": {
            "hup": 0.96, "lzu": 0.96, "multicenter": 0.96, "pediatric": 0.96,
        }, "duplicate_key_count": 0, "nonfinite_count": 0,
    }), encoding="utf-8")
    patients = _all90(); filtered, folds, _ = build_sensitivity80_cohort(patients, _folds(patients), MANIFEST)
    args = _formal_args(); args.dry_run_config_only = False; args.causal_cache_audit_path = str(causal_audit)
    audit = assert_fixed_all90_protocol(args, filtered, folds, STEP4B_STATIC_TOP20_FEATURES)
    assert audit["causal_cache_status"] == "passed"
