from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path(r"E:\DRE-nips\dataest")
DEFAULT_QUALITY_REPORT = Path(
    r"E:\DRE-nips\new-pipeline\new-5-29\hup_strict_interictal_experiments\slope_quality_report_HUP_S.xlsx"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs_hup_strict_interictal"


VERSIONS = (
    "B0_HUP_Filtered",
    "M1_HUP_StrictInterNorm",
    "M2_HUP_InterPhysNode",
    "M3_HUP_InterPhysGraph",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HUP-only strict ictal/interictal NeuroEZ experiments.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--edf_quality_report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--allowed_edf_quality", type=str, default="good")
    parser.add_argument("--preview_quality_aliases", type=str, default="review")
    parser.add_argument("--interictal_quality_policy", choices=["exact_only", "subject_if_any_kept"], default="subject_if_any_kept")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--target_sfreq", type=float, default=1000.0)
    parser.add_argument("--raw_temporal_sfreq", type=float, default=1000.0)
    parser.add_argument("--prepost_context_sec", type=float, default=30.0)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--step_sec", type=float, default=1.0)
    parser.add_argument("--min_ictal_windows", type=int, default=2)
    parser.add_argument("--interictal_min_windows", type=int, default=10)
    parser.add_argument("--max_interictal_windows_per_run", type=int, default=600)
    parser.add_argument("--interictal_pre_buffer_sec", type=float, default=1800.0)
    parser.add_argument("--interictal_post_buffer_sec", type=float, default=1800.0)
    parser.add_argument("--interictal_fallback_buffer_sec", type=float, default=600.0)

    parser.add_argument("--spectral_min_freq", type=float, default=10.0)
    parser.add_argument("--spectral_max_freq", type=float, default=300.0)
    parser.add_argument("--graph_edge_quantile", type=float, default=0.70)
    parser.add_argument("--graph_min_edge_weight", type=float, default=0.10)
    parser.add_argument("--ez_definition", type=str, default="soz_or_resected")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_epochs_before_early_stop", type=int, default=50)
    parser.add_argument("--patient_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_num_workers", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--reuse_caches", action="store_true")
    return parser
