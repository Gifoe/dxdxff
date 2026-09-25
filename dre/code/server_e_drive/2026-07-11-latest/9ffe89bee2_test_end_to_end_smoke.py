from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run_outcome_hifos import main


def _synthetic_cache(path: Path, patients: int = 8) -> Path:
    records = []
    patient_index = {}
    rng = np.random.default_rng(12)
    for patient_index_value in range(patients):
        subject = f"p{patient_index_value:02d}"
        outcome = "success" if patient_index_value % 2 else "failure"
        center = f"c{patient_index_value % 2}"
        channels = [f"A{index + 1}" for index in range(4 + patient_index_value % 2)]
        patient_index[subject] = {
            "canonical_channels": channels,
            "outcome_group": outcome,
            "source_center": center,
        }
        for seizure in range(1 + patient_index_value % 2):
            windows = 3 + seizure
            signal = rng.normal(loc=float(patient_index_value % 2) * 0.3, scale=1.0, size=(windows, len(channels), 5)).astype(np.float32)
            records.append(
                {
                    "subject_id": subject,
                    "run_id": f"r{seizure}",
                    "channel_names_norm": channels,
                    "source_center": center,
                    "sample": {
                        "sample_id": f"{subject}-s{seizure}",
                        "window_features": signal,
                        "window_relative_centers_sec": np.arange(windows, dtype=np.float32),
                        "window_feature_names": [f"f{index}" for index in range(5)],
                    },
                }
            )
    with path.open("wb") as handle:
        pickle.dump({"cache_version": "synthetic", "run_records": records, "patient_index": patient_index}, handle)
    return path


def _synthetic_fm_cache(feature_path: Path, output_path: Path) -> Path:
    with feature_path.open("rb") as handle:
        source = pickle.load(handle)
    embeddings = []
    manifest = []
    for record in source["run_records"]:
        sample = record["sample"]
        values = np.asarray(sample["window_features"], dtype=np.float32)
        centers = np.asarray(sample["window_relative_centers_sec"], dtype=np.float32)
        for window_idx in range(values.shape[0]):
            for channel_idx, channel_name in enumerate(record["channel_names_norm"]):
                manifest.append(
                    {
                        "subject_id": record["subject_id"],
                        "run_id": record["run_id"],
                        "sample_id": sample["sample_id"],
                        "window_idx": window_idx,
                        "window_center_sec": float(centers[window_idx]),
                        "channel_name": channel_name,
                        "embedding_index": len(embeddings),
                    }
                )
                embeddings.append(values[window_idx, channel_idx])
    with output_path.open("wb") as handle:
        pickle.dump(
            {
                "model_name": "synthetic-fm",
                "embeddings": np.stack(embeddings),
                "manifest": manifest,
                "patient_index": source["patient_index"],
            },
            handle,
        )
    return output_path


def test_synthetic_screening_writes_oof_and_report(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "feature.pkl")
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  feature_cache_path: '{cache.as_posix()}'",
                f"  output_dir: '{output.as_posix()}'",
                "model:",
                "  model_dim: 8",
                "  num_cores: 2",
                "  dropout: 0.0",
                "  sinkhorn_iterations: 8",
                "training:",
                "  epochs: 1",
                "  patience: 1",
                "  batch_size: 2",
                "  learning_rate: 0.001",
                "protocol:",
                "  outer_folds: 2",
                "  inner_folds: 2",
                "  random_seed: 42",
            ]
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "run",
            "--config",
            str(config),
            "--protocol",
            "screening",
            "--variants",
            "H2_HIER_POOL,H5_ANCHORED_CORE,H6_ANCHORED_UOT_DESC",
            "--max-outer-folds",
            "1",
            "--seeds",
            "42",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    for name in (
        "outcome_oof_predictions.csv",
        "outcome_metrics_summary.csv",
        "outcome_metrics_by_fold.csv",
        "outcome_metrics_by_center.csv",
        "outcome_training_failures.csv",
        "outcome_run_manifest.json",
        "outcome_final_report.md",
    ):
        assert (output / "reports" / name).exists(), name
    for name in ("resolved_config.yaml", "config_hash.txt", "fold_ledger_hash.txt", "data_manifest_hash.txt", "git_commit.txt", "environment.json"):
        assert (output / name).exists(), name


def test_synthetic_final_nested_selects_candidate_from_inner_oof(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "feature.pkl")
    output = tmp_path / "final-output"
    config = tmp_path / "final-config.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  feature_cache_path: '{cache.as_posix()}'",
                f"  output_dir: '{output.as_posix()}'",
                "model:",
                "  model_dim: 8",
                "  num_cores: 2",
                "  dropout: 0.0",
                "training:",
                "  epochs: 1",
                "  patience: 1",
                "  batch_size: 2",
                "  learning_rate: 0.001",
                "  early_stop_metric: auroc",
                "protocol:",
                "  outer_folds: 2",
                "  inner_folds: 2",
                "  random_seed: 42",
                "candidate_profiles:",
                "  compact:",
                "    model:",
                "      model_dim: 6",
                "      num_cores: 2",
                "  default:",
                "    model:",
                "      model_dim: 8",
                "      num_cores: 2",
            ]
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "run",
            "--config",
            str(config),
            "--protocol",
            "final_nested",
            "--variants",
            "H2_HIER_POOL",
            "--max-outer-folds",
            "1",
            "--seeds",
            "42",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert result == 0
    run_dirs = list((output / "final_nested" / "H2_HIER_POOL").glob("outer_fold_*/seed_42"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    selected = json.loads((run_dir / "selected_candidate.json").read_text(encoding="utf-8"))
    assert selected["selection_source"] == "inner_oof"
    assert selected["candidate_profile"] in {"compact", "default"}
    metrics = pd.read_csv(run_dir / "inner_candidate_metrics.csv")
    assert set(metrics["role"]) == {"inner_oof"}


def test_synthetic_h10_builds_shared_fusion_cohort_and_stacker(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "feature.pkl")
    fm_cache = _synthetic_fm_cache(cache, tmp_path / "fm.pkl")
    output = tmp_path / "fusion-output"
    config = tmp_path / "fusion-config.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  feature_cache_path: '{cache.as_posix()}'",
                f"  fm_embedding_cache_path: '{fm_cache.as_posix()}'",
                f"  output_dir: '{output.as_posix()}'",
                "model:",
                "  model_dim: 8",
                "  num_cores: 2",
                "  dropout: 0.0",
                "  sinkhorn_iterations: 5",
                "training:",
                "  epochs: 1",
                "  patience: 1",
                "  batch_size: 2",
                "  learning_rate: 0.001",
                "protocol:",
                "  outer_folds: 2",
                "  inner_folds: 2",
                "  random_seed: 42",
                "fusion:",
                "  feature_variant: H2_HIER_POOL",
            ]
        ),
        encoding="utf-8",
    )
    result = main(
        [
            "run", "--config", str(config), "--protocol", "screening", "--variants", "H10_LATE_FUSION",
            "--max-outer-folds", "1", "--seeds", "42", "--epochs", "1", "--device", "cpu",
        ]
    )
    assert result == 0
    assert (output / "manifests" / "outcome_fusion_patient_manifest.csv").exists()
    assert (output / "folds" / "outcome_fold_assignments_fusion.csv").exists()
    audits = list((output / "screening" / "H10_LATE_FUSION").glob("outer_fold_*/seed_42/late_fusion_protocol_audit.json"))
    assert len(audits) == 1
    predictions = pd.read_csv(output / "reports" / "outcome_oof_predictions.csv")
    assert set(predictions["variant"]) == {"H2_HIER_POOL_FEATURE_INTERSECTION", "H9_FM_INTERSECTION", "H10_LATE_FUSION"}
    assert set(predictions["threshold_source"]) == {"inner_oof"}


def test_synthetic_loco_uses_fixed_config_without_candidate_profiles(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path / "feature.pkl", patients=12)
    with cache.open("rb") as handle:
        payload = pickle.load(handle)
    for subject_index, subject in enumerate(sorted(payload["patient_index"])):
        center = f"c{(subject_index // 2) % 3}"
        payload["patient_index"][subject]["source_center"] = center
        for record in payload["run_records"]:
            if record["subject_id"] == subject:
                record["source_center"] = center
    with cache.open("wb") as handle:
        pickle.dump(payload, handle)
    output = tmp_path / "loco-output"
    config = tmp_path / "loco-config.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  feature_cache_path: '{cache.as_posix()}'",
                f"  output_dir: '{output.as_posix()}'",
                "model:",
                "  model_dim: 8",
                "  num_cores: 2",
                "  dropout: 0.0",
                "training:",
                "  epochs: 1",
                "  patience: 1",
                "  batch_size: 2",
                "  learning_rate: 0.001",
                "protocol:",
                "  inner_folds: 2",
                "  random_seed: 42",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["loco", "--config", str(config), "--variants", "H2_HIER_POOL", "--seeds", "42", "--device", "cpu"]) == 0
    predictions = pd.read_csv(output / "loco" / "reports" / "outcome_oof_predictions.csv")
    assert set(predictions["role"]) == {"loco_test"}
    assert predictions["held_out_center"].nunique() == 3


def test_synthetic_audit_cli_writes_schema_and_code_audit(tmp_path: Path) -> None:
    feature = _synthetic_cache(tmp_path / "feature.pkl")
    raw = _synthetic_cache(tmp_path / "raw.pkl")
    output = tmp_path / "audit"
    assert main(["audit", "--feature-cache-path", str(feature), "--raw-cache-path", str(raw), "--output-dir", str(output)]) == 0
    assert (output / "outcome_cache_schema_feature.json").exists()
    assert (output / "code_audit.md").exists()
    code_audit = (output / "code_audit.md").read_text(encoding="utf-8")
    assert str(feature) in code_audit
    assert str(raw) in code_audit


def test_summarize_cli_preserves_existing_run_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "reports"
    input_dir.mkdir()
    pd.DataFrame(
        [
            {"variant": "H2_HIER_POOL", "subject_id": "a", "center": "x", "outcome": 0, "seed": 42, "outer_fold_idx": 0, "role": "outer_test", "raw_logit": -1.0, "calibrated_probability": 0.2, "selected_threshold": 0.6, "predicted": 0, "calibration_method": "platt_inner_oof", "threshold_source": "inner_oof", "checkpoint_path": "checkpoint.pt", "fold_ledger_hash": "ledger", "config_hash": "config", "cohort_id": "feature_full", "subject_set_hash": "subjects"},
            {"variant": "H2_HIER_POOL", "subject_id": "b", "center": "y", "outcome": 1, "seed": 42, "outer_fold_idx": 0, "role": "outer_test", "raw_logit": 1.0, "calibrated_probability": 0.8, "selected_threshold": 0.6, "predicted": 1, "calibration_method": "platt_inner_oof", "threshold_source": "inner_oof", "checkpoint_path": "checkpoint.pt", "fold_ledger_hash": "ledger", "config_hash": "config", "cohort_id": "feature_full", "subject_set_hash": "subjects"},
        ]
    ).to_csv(input_dir / "outcome_oof_predictions.csv", index=False)
    manifest = {"protocol": "screening", "random_seed": 17, "git_commit": "abc123", "fold_ledger_hash": "ledger"}
    (input_dir / "outcome_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["summarize", "--input-dir", str(input_dir)]) == 0

    rewritten = json.loads((input_dir / "outcome_run_manifest.json").read_text(encoding="utf-8"))
    assert rewritten == manifest
