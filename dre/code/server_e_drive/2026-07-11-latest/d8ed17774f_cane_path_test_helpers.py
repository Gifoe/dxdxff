from __future__ import annotations

import torch

from neuroez_c.config import apply_pruned_defaults
from neuroez_c.model import NeuroEZCModel
from neuroez_c.protocol import STEP4B_STATIC_TOP20_FEATURES
from run_neuroez_c import build_parser, validate_cane_path_cp_args


def cane_args(*extra: str, model_dim: int = 8):
    values = [
        "--use-cane-path-cp-nez", "--use-causal-propagation-residual", "--cohort-mode", "sensitivity80",
        "--exclude-subjects-file", "dummy.csv", "--causal-propagation-cache-path", "dummy.parquet",
        "--causal-cache-audit-path", "dummy.json", "--require-n-patients", "80", "--positive-label", "nez",
        "--loss-mode", "cane_path_cp_nez", "--outer-split-seed", "42", "--inner-split-seed", "42",
        "--inner-splits", "4", "--model-seed", "42", "--use_physics_dynamics", "--physics_feature_parts", "abs",
        "--physics_state_features", ",".join(STEP4B_STATIC_TOP20_FEATURES), "--class_weight_mode", "none",
        "--ez_negative_weight", "1", "--center-balanced-batches", "--early_stop_metric", "balanced_patient_auprc_hmean",
        "--min_epochs_before_early_stop", "18", "--model_dim", str(model_dim), "--num_heads", "2",
        "--dropout", "0", "--dry_run_config_only",
    ]
    args = build_parser().parse_args(values + list(extra))
    apply_pruned_defaults(args)
    args.random_seed = args.outer_split_seed
    validate_cane_path_cp_args(args)
    return args


def synthetic_batch(labels: bool = False):
    torch.manual_seed(7)
    batch = {
        "b0_features": torch.randn(4, 4, 5, 6, 36),
        "physics_features": torch.randn(4, 4, 5, 6, 8),
        "channel_mask": torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,0],[1,1,1,1,1,1],[1,1,1,0,0,0]], dtype=torch.bool),
        "seizure_mask": torch.tensor([[1,1,0,0],[1,1,1,0],[1,1,1,1],[1,0,0,0]], dtype=torch.bool),
        "seizure_channel_mask": torch.ones(4, 4, 6, dtype=torch.bool),
        "window_mask": torch.ones(4, 4, 5, dtype=torch.bool),
        "causal_propagation_features": torch.rand(4, 6, 6),
        "causal_propagation_valid": torch.ones(4, 6, dtype=torch.bool),
        "center_id": torch.arange(4),
        "center": ["hup", "lzu", "multicenter", "pediatric"],
        "subject_id": ["hup:a", "lzu:b", "multicenter:c", "pediatric:d"],
        "canonical_channels": [[f"c{i}" for i in range(6)] for _ in range(4)],
    }
    batch["seizure_channel_mask"][0, 2:] = False
    batch["seizure_channel_mask"][3, 1:] = False
    batch["causal_propagation_valid"][0, 3:] = False
    if labels:
        batch["labels_ez"] = torch.tensor([
            [0,1,0,1,-1,-1], [0,0,1,1,0,-1], [1,0,1,0,1,0], [0,1,1,-1,-1,-1]
        ], dtype=torch.float32)
    return batch


def cane_model() -> NeuroEZCModel:
    torch.manual_seed(11)
    return NeuroEZCModel(cane_args()).eval()
