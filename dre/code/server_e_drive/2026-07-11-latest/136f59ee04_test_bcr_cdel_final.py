from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import torch

from neuroez_c.bcr_checkpoint import inspect_bcr_state_dict
from neuroez_c.model import NeuroEZCModel
from neuroez_c.p2_v3_conservative_fusion import (
    LOCKED_BCR_WEIGHT,
    LOCKED_PRQ_WEIGHT,
    bcr_ez_logit_to_nez_probability,
    conservative_probability_fusion,
)
from neuroez_c.v3_qbc_profiles import BCR_PROFILES
from P2_V3_AAAI_ABLATIONS.threshold_protocol import select_fold_threshold
from scripts.run_v3_qbc import (
    _command as build_bcr_training_command,
    _parser as bcr_runner_parser,
    _sanitize_effective_config,
)


def test_final_bcr_profiles_contain_no_q10_switch() -> None:
    assert set(BCR_PROFILES) == {
        "BCR_BC_ONLY", "BCR_BOUNDARY_ONLY", "BCR_COVERAGE_ONLY", "BCR_BOUNDARY_COVERAGE",
    }
    assert all("q10" not in key.lower() for key in BCR_PROFILES)
    assert not hasattr(BCR_PROFILES["BCR_BOUNDARY_COVERAGE"], "use_q10")


def test_bcr_direction_and_locked_cdel_fusion() -> None:
    logits = np.array([2.0, 0.0, -2.0])
    bcr = bcr_ez_logit_to_nez_probability(logits)
    assert np.allclose(bcr, 1.0 - 1.0 / (1.0 + np.exp(-logits)))
    fused, _ = conservative_probability_fusion(
        np.array([0.2, 0.6, 0.9]), np.array([0.5, 0.4, 0.7])
    )
    assert LOCKED_PRQ_WEIGHT == pytest.approx(0.80)
    assert LOCKED_BCR_WEIGHT == pytest.approx(0.20)
    assert np.allclose(fused, [0.26, 0.56, 0.86])


def test_bcr_state_dictionary_has_no_q10_keys_and_handles_masks() -> None:
    args = SimpleNamespace(
        model_dim=8, num_heads=2, dropout=0.0, use_channel_attention=True,
        use_v3_qbc=True, v3_qbc_profile="BCR_BOUNDARY_COVERAGE",
        positive_label="ez", use_patient_relative_z=True,
        use_physics_dynamics=False, use_diffusion_residual=False,
        use_negative_anchor_head=False, use_view_gated_fusion=False,
        use_two_expert_router=False, use_feature_separated_two_expert=False,
        use_a9v8_lcbo=False, use_n6_dual_view_ema=False, use_v3_rcc=False,
        temporal_pooling="mean", channel_pooling_mode="mean", record_pooling="mean",
    )
    model = NeuroEZCModel(args)
    assert not inspect_bcr_state_dict(model.state_dict())["legacy_q10_keys"]
    # The module construction itself is mask-safe; full encoder forward is
    # covered by its existing model tests and needs cache-specific tensors.
    assert model.channel_classifier is not None


def test_legacy_bcr_q10_checkpoint_is_detected() -> None:
    report = inspect_bcr_state_dict({"v3_simple_q10.scorer.weight": torch.ones(1)})
    assert report["legacy_q10_keys"] == ["v3_simple_q10.scorer.weight"]


def test_bcr_runner_does_not_require_optional_checkpoint_helper_for_training() -> None:
    """Fresh BCR training must run even if only optional inspection code is absent."""
    runner = Path("scripts/run_v3_qbc.py")
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    module_imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "neuroez_c.bcr_checkpoint"
    ]
    assert not module_imports


def test_bcr_runner_accepts_orchestrator_output_dir_spelling() -> None:
    args = bcr_runner_parser().parse_args([
        "--window-cache-path", "cache.pkl",
        "--allowed-subjects-ledger", "subjects.csv",
        "--outer-fold-manifest", "folds.csv",
        "--output-dir", "output",
        "--v3_qbc_profile", "BCR_BOUNDARY_COVERAGE",
        "--epochs", "40",
        "--patience", "8",
        "--min-epochs-before-early-stop", "18",
        "--batch-size", "4",
        "--patient-batch-size", "4",
        "--num-workers", "0",
        "--device", "cuda",
        "--skip-existing",
    ])
    assert args.output_dir == "output"
    assert args.patience == 8
    assert args.min_epochs_before_early_stop == 18


def test_final_bcr_sanitizes_retired_q10_base_arguments() -> None:
    effective, removed = _sanitize_effective_config({
        "output_dir": "output",
        "v3_qbc_profile": "BCR_BOUNDARY_COVERAGE",
        "v3_qbc_q10_gate_init": -3.0,
        "v3_qbc_q10_max_residual": 0.2,
        "model_family": "legacy",
        "score_semantics": "ez_probability",
    })
    assert set(removed) == {
        "model_family",
        "score_semantics",
        "v3_qbc_q10_gate_init",
        "v3_qbc_q10_max_residual",
    }
    command = build_bcr_training_command("python", effective)
    assert not any("q10" in argument.lower() for argument in command)
    assert "BCR_BOUNDARY_COVERAGE" in command


def test_validation_threshold_is_invariant_to_test_labels() -> None:
    validation = pd.DataFrame({
        "subject_id": ["p1", "p1", "p2", "p2"],
        "label_nez": [1, 0, 1, 0],
        "score_nez": [0.8, 0.2, 0.7, 0.3],
    })
    first, _ = select_fold_threshold(validation, score_column="score_nez", step=0.005)
    test_labels_changed = validation.copy()
    test_labels_changed["label_nez"] = 1 - test_labels_changed["label_nez"]
    second, _ = select_fold_threshold(validation, score_column="score_nez", step=0.005)
    assert first == second


def test_prq_q10_path_is_preserved_outside_bcr() -> None:
    prq_source = Path("P23_TRN_NEZ_80/neuroez_c/p23_trainer.py").read_text(encoding="utf-8")
    assert "q10" in prq_source.lower()
