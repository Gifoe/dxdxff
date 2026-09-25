from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data_factory import build_outer_splits
from ez_features import WINDOW_NODE_FEATURE_NAMES
from exp_ez_hybrid import Exp_EZHybridLocalization
from neuroez_c.dataset import build_patient_examples
from neuroez_c.protocol import (
    STEP4B_STATIC_TOP20_FEATURES,
    assert_fixed_all90_protocol,
)
from scripts.prepare_step4b_static_top20_all90_cache import prepare_step4b_cache


ROOT = Path(__file__).resolve().parents[1]


def _args(**updates):
    values = {
        "fixed_all90_protocol_name": "fixed_all90_step4b_static_top20_nez",
        "fixed_all90_cache_audit_path": "",
        "positive_label": "nez",
        "score_semantics": "nez_probability",
        "split_strategy": "5fold",
        "n_splits": 5,
        "random_seed": 42,
        "drop_high_ez_fraction_lzu": False,
        "require_n_patients": 90,
        "physics_state_features": ",".join(STEP4B_STATIC_TOP20_FEATURES),
        "physics_feature_parts": "abs",
        "use_physics_dynamics": True,
        "use_channel_attention": True,
        "use_patient_relative_z": True,
        "group_robust_mode": "none",
        "use_diffusion_residual": False,
        "use_ez_ranking_loss": False,
        "use_hard_topk_loss": False,
        "use_negative_anchor_head": False,
        "use_two_expert_router": False,
        "use_feature_separated_two_expert": False,
        "use_broad_ez_mil_loss": False,
        "use_a9v8_lcbo": False,
        "use_teacher_anchor_eval": False,
        "teacher_anchor_apply_to_train_loss": False,
        "train_subject_dropout_file": "",
        "train_subject_dropout_count": 0,
        "train_subject_dropout_seed": -1,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _patients(count: int = 90) -> dict[str, dict]:
    return {f"p{idx:03d}": {} for idx in range(count)}


def _valid_protocol(count: int = 90):
    patient_index = _patients(count)
    splits = build_outer_splits(patient_index, split_strategy="5fold", n_splits=5, random_seed=42)
    return patient_index, splits


def test_step4b_protocol_accepts_fixed_all90_nez_config() -> None:
    patient_index, splits = _valid_protocol()
    audit = assert_fixed_all90_protocol(
        _args(), patient_index, splits,
        cache_feature_names=list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES),
    )
    assert audit["protocol_name"] == "fixed_all90_step4b_static_top20_nez"
    assert audit["positive_label"] == "nez"
    assert audit["score_semantics"] == "nez_probability"
    assert audit["n_patients"] == 90
    assert audit["n_outer_splits"] == 5
    assert audit["complete_oof_coverage"] is True


@pytest.mark.parametrize("count", [89, 91])
def test_step4b_protocol_rejects_non_all90_counts(count: int) -> None:
    patient_index, splits = _valid_protocol(count)
    with pytest.raises(ValueError, match="90 patients"):
        assert_fixed_all90_protocol(
            _args(), patient_index, splits,
            cache_feature_names=list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"drop_high_ez_fraction_lzu": True}, "drop_high_ez_fraction_lzu"),
        ({"positive_label": "ez", "score_semantics": "ez_probability"}, "positive_label"),
        ({"score_semantics": "ez_probability"}, "score_semantics"),
        ({"fixed_all90_protocol_name": "fixed_all90_step4b_static_top20_ez"}, "unsupported"),
        ({"train_subject_dropout_count": 1}, "train_subject_dropout_file"),
        ({"random_seed": 7}, "random_seed"),
        ({"n_splits": 4}, "n_splits"),
        ({"model_input_features": "center_id"}, "center_id"),
    ],
)
def test_step4b_protocol_rejects_invalid_config(updates: dict, message: str) -> None:
    patient_index, splits = _valid_protocol()
    with pytest.raises(ValueError, match=message):
        assert_fixed_all90_protocol(
            _args(**updates), patient_index, splits,
            cache_feature_names=list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES),
        )


def test_step4b_protocol_rejects_missing_feature() -> None:
    patient_index, splits = _valid_protocol()
    names = list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES[:-1])
    with pytest.raises(ValueError, match="missing Step4B features"):
        assert_fixed_all90_protocol(_args(), patient_index, splits, cache_feature_names=names)


def test_step4b_protocol_rejects_fold_leakage() -> None:
    patient_index, splits = _valid_protocol()
    leaked = [dict(split) for split in splits]
    leaked[0] = dict(leaked[0])
    leaked[0]["train_subjects"] = list(leaked[0]["train_subjects"]) + [leaked[0]["test_subjects"][0]]
    with pytest.raises(ValueError, match="leakage"):
        assert_fixed_all90_protocol(
            _args(), patient_index, leaked,
            cache_feature_names=list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES),
        )


def test_step4b_protocol_rejects_incomplete_test_union() -> None:
    patient_index, splits = _valid_protocol()
    incomplete = [dict(split) for split in splits]
    incomplete[0] = dict(incomplete[0])
    removed = incomplete[0]["test_subjects"][0]
    incomplete[0]["test_subjects"] = incomplete[0]["test_subjects"][1:]
    incomplete[0]["train_subjects"] = list(incomplete[0]["train_subjects"]) + [removed]
    with pytest.raises(ValueError, match="held-out test union"):
        assert_fixed_all90_protocol(
            _args(), patient_index, incomplete,
            cache_feature_names=list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES),
        )


def _write_synthetic_source(path: Path, subjects: list[str]) -> None:
    rng = np.random.default_rng(42)
    patient_index = {}
    records = []
    centers = np.asarray([-5.0, -2.0, 0.5, 2.0, 5.0, 8.0, 10.0], dtype=np.float32)
    for subject_idx, subject in enumerate(subjects):
        patient_index[subject] = {
            "canonical_channels": ["a", "b", "c"],
            "labels": np.asarray([1.0, 0.0, float(subject_idx % 2)], dtype=np.float32),
            "source_center": "lzu" if subject_idx == 0 else "hup",
        }
        features = rng.normal(size=(len(centers), 3, len(WINDOW_NODE_FEATURE_NAMES))).astype(np.float32)
        high_gamma = list(WINDOW_NODE_FEATURE_NAMES).index("log_bp_high_gamma")
        line_length = list(WINDOW_NODE_FEATURE_NAMES).index("line_length_per_sec")
        features[:, :, high_gamma] += np.arange(len(centers), dtype=np.float32)[:, None] * np.asarray([0.2, 0.4, 0.7])
        features[:, :, line_length] += np.arange(len(centers), dtype=np.float32)[:, None] * np.asarray([0.5, 0.1, 0.8])
        records.append({
            "subject_id": subject,
            "run_id": f"run-{subject_idx}",
            "sample": {
                "window_features": features,
                "window_relative_centers_sec": centers,
                "window_feature_names": list(WINDOW_NODE_FEATURE_NAMES),
            },
        })
    payload = {
        "cache_version": "synthetic-base20",
        "label_semantics": "cache labels are EZ-positive; model training converts to NEZ-positive",
        "window_feature_names": list(WINDOW_NODE_FEATURE_NAMES),
        "patient_index": patient_index,
        "run_records": records,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_feature_only_builder_creates_exact_20_plus_8_cache(tmp_path: Path) -> None:
    subjects = ["lzu:p1", "hup:p2"]
    source = tmp_path / "feature.pkl"
    target = tmp_path / "derived.pkl"
    ledger = tmp_path / "subjects.csv"
    audit = tmp_path / "audit.json"
    pd.DataFrame({"subject_id": subjects}).to_csv(ledger, index=False)
    _write_synthetic_source(source, subjects)
    report = prepare_step4b_cache(
        source, target, ledger, audit,
        expected_patients=2, expected_runs=2,
    )
    assert target.exists()
    assert report["raw_cache_used"] is False
    assert report["n_patients"] == 2
    assert report["n_runs"] == 2
    assert report["window_feature_count"] == 28
    assert report["required_feature_count"] == 8
    with target.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["window_feature_names"] == list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES)


def test_step4b_nez_training_label_is_inverted_exactly_once() -> None:
    feature_names = list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES)
    sample = {
        "subject_id": "hup:p1",
        "run_id": "run-1",
        "sample_id": "sample-1",
        "channel_names_norm": ["a", "b", "c"],
        "labels": np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
        "window_features": np.ones((2, 3, len(feature_names)), dtype=np.float32),
        "window_feature_names": feature_names,
        "window_adjacency": np.zeros((2, 3, 3), dtype=np.float32),
        "window_relative_centers_sec": np.asarray([-1.0, 1.0], dtype=np.float32),
    }
    patient_index = {
        "hup:p1": {
            "canonical_channels": ["a", "b", "c"],
            "labels": np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
            "label_mask": np.asarray([True, True, True]),
            "source_center": "hup",
        }
    }

    example = build_patient_examples([sample], patient_index, args=_args())[0]

    np.testing.assert_array_equal(example["labels_ez"], np.asarray([1.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(example["labels_nez"], np.asarray([0.0, 1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(example["labels"], example["labels_nez"])
    assert example["label_semantics"] == "1=NEZ,0=EZ"


def test_train_subject_dropout_is_deterministic_and_fit_only() -> None:
    experiment = object.__new__(Exp_EZHybridLocalization)
    experiment.train_subject_dropout_count = 2
    experiment.train_subject_dropout_seed = 700
    experiment.train_subject_dropout_candidates = ("p1", "p2", "p3", "p9")
    experiment.args = SimpleNamespace(random_seed=42)

    fit_before = ["p0", "p1", "p2", "p3", "p4"]
    retained_a, dropped_a = experiment._sample_train_subject_dropout(fit_before, fold_idx=2)
    retained_b, dropped_b = experiment._sample_train_subject_dropout(fit_before, fold_idx=2)

    assert dropped_a == dropped_b
    assert len(dropped_a) == 2
    assert set(dropped_a).issubset({"p1", "p2", "p3"})
    assert set(retained_a) | set(dropped_a) == set(fit_before)
    assert not set(retained_a) & set(dropped_a)
    assert retained_a == retained_b


def test_feature_only_dry_run_does_not_write_target_cache(tmp_path: Path) -> None:
    subjects = ["lzu:p1", "hup:p2"]
    source = tmp_path / "feature.pkl"
    target = tmp_path / "derived.pkl"
    ledger = tmp_path / "subjects.csv"
    audit = tmp_path / "audit.json"
    pd.DataFrame({"subject_id": subjects}).to_csv(ledger, index=False)
    _write_synthetic_source(source, subjects)
    report = prepare_step4b_cache(
        source, target, ledger, audit,
        dry_run=True, expected_patients=2, expected_runs=2,
    )
    assert not target.exists()
    assert audit.exists()
    assert report["target_cache_action"] == "dry_run_simulated"


def test_skip_existing_requires_all_complete_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    run_dir = output_root / "step4b_static_top20_all90_posNEZ_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "heldout_summary_neuroez_v3.csv").write_text("patient_macro_f1\n0.5\n", encoding="utf-8")
    command = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(ROOT / "scripts" / "run_step4b_static_top20_all90_seed42.ps1"),
        "-RepoRoot", str(ROOT),
        "-FeatureCachePath", str(tmp_path / "missing-feature.pkl"),
        "-CachePath", str(tmp_path / "missing-target.pkl"),
        "-OutputRoot", str(output_root),
        "-SkipExisting", "-DryRun",
    ]
    partial = subprocess.run(command, text=True, capture_output=True, check=False)
    assert partial.returncode != 0
    assert "Complete output exists; skipping" not in partial.stdout

    (run_dir / "heldout_summary_neuroez_v3.json").write_text(json.dumps({"patient_macro_f1": 0.5, "classification_threshold": 0.5}), encoding="utf-8")
    (run_dir / "run_args_b0_pruned.json").write_text(json.dumps({"classification_threshold": 0.5}), encoding="utf-8")
    (run_dir / "fixed_all90_protocol_audit.json").write_text(json.dumps({
        "protocol_name": "fixed_all90_step4b_static_top20_nez",
        "n_patients": 90,
        "n_outer_splits": 5,
        "positive_label": "nez",
        "score_semantics": "nez_probability",
    }), encoding="utf-8")
    complete = subprocess.run(command, text=True, capture_output=True, check=False)
    assert complete.returncode == 0
    assert "Complete output exists; skipping" in complete.stdout
