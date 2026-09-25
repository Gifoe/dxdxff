from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from exp_ez_localization import Exp_EZLocalization
from report_threshold import summarize_cv_results


def _log(message: str) -> None:
    print(f"[TeChEZ-Ranker] {message}", flush=True)


def _describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    try:
        return f"{device} ({torch.cuda.get_device_name(device)})"
    except Exception:
        return str(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patient-wise TeChEZ ranker with all features and oracle ranking diagnostics"
    )
    parser.add_argument("--dataset_dir", type=str, default=r"E:\DRE-nips\dataest")
    parser.add_argument(
        "--participants_path",
        type=str,
        default=r"E:\DRE-nips\dataest\participants.tsv",
    )
    parser.add_argument("--subject_filter", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"E:\DRE-nips\new-pipeline\new-4-24\outputs",
    )
    parser.add_argument("--sample_cache_path", "--window_cache_path", dest="sample_cache_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="PatientChannelRanker")

    parser.add_argument("--split_strategy", type=str, default="5fold", choices=["5fold", "lopo"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--feature_num_workers", type=int, default=8)
    parser.add_argument("--target_sfreq", type=float, default=512.0)
    parser.add_argument("--feature_scales_sec", type=str, default="3,5,10")
    parser.add_argument("--raw_temporal_duration_sec", type=float, default=5.0)
    parser.add_argument("--raw_temporal_sfreq", type=float, default=256.0)
    parser.add_argument("--ez_definition", type=str, default="soz_or_resected")
    parser.add_argument("--success_only", action="store_true", default=True)
    parser.add_argument("--force_rebuild_cache", action="store_true")

    parser.add_argument("--spectral_min_freq", type=float, default=1.0)
    parser.add_argument("--spectral_max_freq", type=float, default=150.0)
    parser.add_argument("--graph_edge_quantile", type=float, default=0.70)
    parser.add_argument("--graph_min_edge_weight", type=float, default=0.10)

    parser.add_argument("--raw_feature_weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--channel_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--local_logit_weight", type=float, default=0.35)
    parser.add_argument("--count_blend_weight", type=float, default=0.35)
    parser.add_argument("--rank_margin", type=float, default=0.25)
    parser.add_argument("--rank_lambda", type=float, default=1.0)
    parser.add_argument("--listwise_lambda", type=float, default=1.0)
    parser.add_argument("--bce_lambda", type=float, default=0.25)
    parser.add_argument("--count_lambda", type=float, default=0.20)
    parser.add_argument("--mass_lambda", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--ranker_shaft_smooth_weight", type=float, default=0.0)
    parser.add_argument("--ranker_consistency_weight", type=float, default=0.0)
    parser.add_argument("--ranker_presence_weight", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _log("Step 1/5 - Parsed command-line arguments.")
    _log(
        "Experiment config: "
        f"dataset_dir={args.dataset_dir} | output_dir={args.output_dir} | "
        f"device={args.device} | CUDA available={torch.cuda.is_available()} | "
        f"feature_scales_sec={args.feature_scales_sec} | "
        f"epochs={args.epochs} | d_model={args.d_model}"
    )

    _log("Step 2/5 - Initializing patient-wise ranking experiment.")
    exp = Exp_EZLocalization(args)
    _log(
        "Step 3/5 - Initialization complete. "
        f"Using device: {_describe_device(exp.device)} | "
        f"patients: {len(exp.patient_index)} | folds: {len(exp.outer_splits)}"
    )

    _log("Step 4/5 - Starting cross-validation training and evaluation.")
    predictions = exp.run()

    _log("Step 5/5 - Writing per-patient reports and summary files.")
    summary = summarize_cv_results(predictions, args.output_dir)
    print(
        "TeChEZ patient-wise ranker pipeline completed. Summary written to: "
        f"{Path(args.output_dir) / 'summary_metrics_threshold.json'}"
    )
    print(summary)


if __name__ == "__main__":
    main()
