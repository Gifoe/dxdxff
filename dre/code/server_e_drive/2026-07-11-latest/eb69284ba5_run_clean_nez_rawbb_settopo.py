from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import json_safe
from neuroez_c.settopo_reranker import run_settopo_reranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fold-safe CleanNEZ RawBrainBERT SetTopo reranker.")
    parser.add_argument("--clean-nez-distance-ledger", required=True)
    parser.add_argument("--v3-ledger", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-rule", default="union_top30_feature_raw_neighbors")
    parser.add_argument("--alpha-list", default="0.10")
    parser.add_argument("--train-alpha", type=float, default=0.10)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lambda-clean-nez", type=float, default=1.0)
    parser.add_argument("--lambda-noisy-pos-rank", type=float, default=0.20)
    parser.add_argument("--lambda-topology", type=float, default=0.01)
    parser.add_argument("--lambda-residual", type=float, default=0.001)
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument("--w-feature", type=float, default=1.0)
    parser.add_argument("--w-raw-onset", type=float, default=1.0)
    parser.add_argument("--w-raw-all", type=float, default=0.5)
    parser.add_argument("--w-raw-preictal", type=float, default=0.25)
    parser.add_argument("--model-type", choices=["real_settopo", "ridge_residual_baseline"], default="real_settopo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, audit = run_settopo_reranker(
        args.clean_nez_distance_ledger,
        args.output_dir,
        candidate_rule=args.candidate_rule,
        alpha_list=args.alpha_list,
        train_alpha=args.train_alpha,
        model_type=args.model_type,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        lambda_clean_nez=args.lambda_clean_nez,
        lambda_noisy_pos_rank=args.lambda_noisy_pos_rank,
        lambda_topology=args.lambda_topology,
        lambda_residual=args.lambda_residual,
        ranking_margin=args.ranking_margin,
        w_feature=args.w_feature,
        w_raw_onset=args.w_raw_onset,
        w_raw_all=args.w_raw_all,
        w_raw_preictal=args.w_raw_preictal,
        device=args.device,
        amp=args.amp,
        patience=args.patience,
        seed=args.seed,
        allowed_subjects_ledger=args.allowed_subjects_ledger,
        allowed_subjects_file=args.allowed_subjects_file,
        require_n_patients=args.require_n_patients,
        label_encoding_mode=args.label_encoding_mode,
    )
    print(json.dumps(json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
