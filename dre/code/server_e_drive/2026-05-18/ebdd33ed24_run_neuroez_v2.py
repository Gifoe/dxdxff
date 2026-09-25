from __future__ import annotations

import argparse
import json
from pathlib import Path

def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DG-NeuroEZ-NEZ+ patient-level inverse EZ ranking.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--participants_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs_neuroez_v2")
    parser.add_argument("--subject_filter", type=str, default=None)
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--sample_cache_path", type=str, default=None)
    parser.add_argument("--window_cache_path", type=str, default=None)

    parser.add_argument("--split_strategy", type=str, default="5fold", choices=["5fold", "lopo"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--target_sfreq", type=float, default=1000.0)
    parser.add_argument("--prepost_context_sec", type=float, default=30.0)
    parser.add_argument("--raw_temporal_sfreq", type=float, default=1000.0)
    parser.add_argument("--spectral_min_freq", type=float, default=10.0)
    parser.add_argument("--spectral_max_freq", type=float, default=300.0)
    parser.add_argument("--graph_edge_quantile", type=float, default=0.70)
    parser.add_argument("--graph_min_edge_weight", type=float, default=0.10)
    parser.add_argument("--sliding_window_sec", type=float, default=2.0)
    parser.add_argument("--sliding_step_sec", type=float, default=1.0)
    parser.add_argument("--ez_definition", type=str, default="soz_or_resected")
    parser.add_argument("--extract_static_features", action="store_true")
    parser.add_argument("--extract_window_tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_num_workers", type=int, default=8)

    parser.add_argument("--positive_label", type=str, default="nez", choices=["nez", "ez"])
    parser.add_argument("--save_both_label_views", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--score_semantics", type=str, default="nez_probability", choices=["nez_probability"])
    parser.add_argument("--selection_target", type=str, default="ez", choices=["ez"])
    parser.add_argument("--selection_score", type=str, default="ez_from_nez", choices=["ez_from_nez", "score_ez", "score_nez"])
    parser.add_argument("--decision_rule", type=str, default="threshold_nez")
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--tune_decision_rule", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--threshold_tuning_metric", type=str, default="patient_macro_f1")
    parser.add_argument("--threshold_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")

    parser.add_argument("--feature_view", type=str, default="self_comparison", choices=["self_comparison", "absolute", "raw", "none"])
    parser.add_argument("--self_compare_baseline", type=str, default="pre_onset", choices=["pre_onset"])
    parser.add_argument("--self_compare_eps", type=float, default=1e-5)
    parser.add_argument("--self_compare_include_abs", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--self_compare_include_pre_mean", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--self_compare_include_delta", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--self_compare_include_zdelta", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--self_compare_include_ratio", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--self_compare_include_channel_rank", type=_str_to_bool, nargs="?", const=True, default=False)

    parser.add_argument("--adjacency_view", type=str, default="mixed_abs_delta", choices=["mixed_abs_delta", "absolute", "abs", "raw", "none"])
    parser.add_argument("--delta_adjacency_baseline", type=str, default="pre_onset", choices=["pre_onset"])
    parser.add_argument("--delta_adjacency_alpha", type=float, default=0.5)

    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--temporal_encoder", type=str, default="mean", choices=["tcn", "bigru", "mean"])
    parser.add_argument("--channel_layers", type=int, default=1)
    parser.add_argument("--seizure_pooling", type=str, default="attention", choices=["attention", "mean", "avg", "average"])
    parser.add_argument("--use_adjacency_message_passing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_channel_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_raw_cnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--raw_view", type=str, default="baseline_normalized", choices=["baseline_normalized", "zscore"])
    parser.add_argument("--raw_cnn_weight", type=float, default=0.25)
    parser.add_argument("--raw_cnn_channels", type=int, default=48)
    parser.add_argument("--raw_cnn_max_samples", type=int, default=8192)
    parser.add_argument("--fusion", type=str, default="gated")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_epochs_before_early_stop", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patient_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=1.0)
    parser.add_argument("--rank_target", type=str, default="nez_higher", choices=["nez_higher", "ez_higher"])
    parser.add_argument("--rank_loss_weight", type=float, default=0.05)
    parser.add_argument("--count_target", type=str, default="both", choices=["both", "ez", "nez"])
    parser.add_argument("--ez_count_loss_weight", type=float, default=1.0)
    parser.add_argument("--nez_count_loss_weight", type=float, default=0.5)
    parser.add_argument("--count_loss_weight", type=float, default=0.0)
    parser.add_argument("--class_weight_mode", type=str, default="ez_negative", choices=["ez_negative", "none"])
    parser.add_argument("--ez_negative_weight", type=str, default="2")
    parser.add_argument("--ez_negative_weight_cap", type=float, default=20.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="patient_macro_f1",
        choices=[
            "patient_macro_f1",
            "patient_balanced_accuracy",
            "patient_weighted_f1",
            "pooled_macro_f1",
            "pooled_balanced_accuracy",
            "patient_macro_ez_f1",
            "patient_macro_nez_f1",
            "patient_macro_f1_ez",
            "patient_macro_f1_nez",
            "ez_auprc",
            "ez_mrr",
            "ez_recall_at_true_count",
            "val_loss",
        ],
    )

    parser.add_argument("--use_supervised_contrastive", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--supcon_weight", type=float, default=0.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.20)
    parser.add_argument("--supcon_label_view", type=str, default="ez_nez")
    parser.add_argument("--supcon_ez_class_weight", type=float, default=2.0)
    parser.add_argument("--supcon_nez_class_weight", type=float, default=1.0)
    parser.add_argument("--use_patient_adversarial", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--adv_weight", type=float, default=0.0)
    parser.add_argument("--adv_warmup_epochs", type=int, default=5)
    parser.add_argument("--adv_max_weight", type=float, default=0.02)
    parser.add_argument("--patient_discriminator_hidden", type=int, default=64)
    parser.add_argument("--use_disentanglement", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--disentangle_weight", type=float, default=0.0)
    parser.add_argument("--orthogonality_weight", type=float, default=0.0)
    parser.add_argument("--patient_projection_dim", type=int, default=32)
    parser.add_argument("--task_projection_dim", type=int, default=32)
    parser.add_argument("--use_graph_edge_dropout", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--graph_edge_dropout", type=float, default=0.0)
    parser.add_argument("--use_graph_sparsity_loss", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--graph_sparsity_weight", type=float, default=0.0)
    parser.add_argument("--use_learnable_edge_gate", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--edge_gate_temperature", type=float, default=1.0)
    parser.add_argument("--use_conditional_alignment", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--alignment_type", type=str, default="coral")
    parser.add_argument("--alignment_weight", type=float, default=0.0)
    parser.add_argument("--alignment_label_view", type=str, default="ez_nez")
    parser.add_argument("--alignment_min_channels_per_class", type=int, default=2)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_interval", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_args_neuroez_v2.json", "w", encoding="utf-8") as fout:
        json.dump(vars(args), fout, indent=2, ensure_ascii=False, sort_keys=True)
    from exp_ez_hybrid import Exp_EZHybridLocalization

    experiment = Exp_EZHybridLocalization(args)
    experiment.run()


if __name__ == "__main__":
    main()
