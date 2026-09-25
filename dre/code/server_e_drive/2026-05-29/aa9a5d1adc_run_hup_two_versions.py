from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


OVERLAY_DIR = Path(__file__).resolve().parent
REPO_ROOT = OVERLAY_DIR.parent
RUN_SCRIPT = OVERLAY_DIR / "run_neuroez_v2.py"
DEFAULT_OUTPUT_ROOT = Path(r"/root/nips-E/hup_phys_compare_results")
DEFAULT_CACHE_PATH = Path(r"/root/nips-E/outputs_dgneuroez_nez_v3_small/cache/techez_prepost_dynamic_cache_v5_9cb761e87145.pkl")


def _common_args(args: argparse.Namespace, output_dir: Path, seed: int) -> list[str]:
    return [
        str(RUN_SCRIPT),
        "--dataset_dir",
        str(args.dataset_dir),
        "--success_only",
        "--prepost_context_sec",
        "30.0",
        "--split_strategy",
        "5fold",
        "--n_splits",
        "5",
        "--positive_label",
        "nez",
        "--score_semantics",
        "nez_probability",
        "--feature_view",
        "self_comparison",
        "--self_compare_include_abs",
        "true",
        "--self_compare_include_delta",
        "true",
        "--self_compare_include_zdelta",
        "true",
        "--self_compare_include_ratio",
        "true",
        "--self_compare_include_channel_rank",
        "false",
        "--adjacency_view",
        "mixed_abs_delta",
        "--delta_adjacency_alpha",
        "0.5",
        "--graph_edge_quantile",
        "0.70",
        "--graph_min_edge_weight",
        "0.10",
        "--model_dim",
        "32",
        "--num_heads",
        "2",
        "--dropout",
        "0.40",
        "--temporal_encoder",
        "mean",
        "--channel_layers",
        "1",
        "--seizure_pooling",
        "attention",
        "--use_adjacency_message_passing",
        "--use_channel_attention",
        "--class_weight_mode",
        "ez_negative",
        "--ez_negative_weight",
        "2",
        "--rank_target",
        "nez_higher",
        "--rank_loss_weight",
        "0.05",
        "--count_loss_weight",
        "0.0",
        "--decision_rule",
        "threshold_nez",
        "--tune_decision_rule",
        "true",
        "--threshold_tuning_metric",
        "patient_macro_f1",
        "--early_stop_metric",
        "patient_macro_f1",
        "--learning_rate",
        "1e-4",
        "--weight_decay",
        "1e-3",
        "--epochs",
        "80",
        "--patience",
        "6",
        "--min_epochs_before_early_stop",
        "50",
        "--patient_batch_size",
        "2",
        "--num_workers",
        "0",
        "--feature_num_workers",
        "8",
        "--device",
        "auto",
        "--log_interval",
        "1",
        "--no-use_raw_cnn",
        "--use_graph_edge_dropout",
        "false",
        "--graph_edge_dropout",
        "0.0",
        "--use_supervised_contrastive",
        "false",
        "--supcon_weight",
        "0.0",
        "--use_disentanglement",
        "false",
        "--orthogonality_weight",
        "0.0",
        "--disentangle_weight",
        "0.0",
        "--use_patient_adversarial",
        "false",
        "--adv_max_weight",
        "0.0",
        "--use_conditional_alignment",
        "false",
        "--alignment_weight",
        "0.0",
        "--use_graph_sparsity_loss",
        "false",
        "--graph_sparsity_weight",
        "0.0",
        "--use_learnable_edge_gate",
        "false",
        "--window_cache_path",
        str(args.cache_path),
        "--output_dir",
        str(output_dir),
        "--random_seed",
        str(seed),
    ]


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    python_path_parts = [str(OVERLAY_DIR), str(REPO_ROOT)]
    existing_python_path = env.get("PYTHONPATH")
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    print(" ".join(f'"{part}"' if " " in part else part for part in [sys.executable, *cmd]), flush=True)
    with open(log_path, "w", encoding="utf-8") as fout:
        process = subprocess.Popen(
            [sys.executable, *cmd],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            fout.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, [sys.executable, *cmd])


def _aggregate(output_root: Path) -> None:
    cmd = [str(REPO_ROOT / "aggregate_neuroez_results.py"), "--output_root", str(output_root)]
    _run(cmd, output_root / "aggregate.log")


def run_variant(args: argparse.Namespace, name: str, extra_args: list[str]) -> None:
    variant_root = args.output_root / name
    for seed in args.seeds:
        run_name = f"Baseline_seed{seed}"
        out_dir = variant_root / run_name
        print(f"===== START {name} {run_name} =====", flush=True)
        _run([*_common_args(args, out_dir, seed), *extra_args], out_dir / "run.log")
        print(f"===== DONE {name} {run_name} =====", flush=True)
        _aggregate(variant_root)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.replace(";", ",").split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HUP PhysFeat and Phys-NeuroEZ-B variants from one external cache.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--seeds", type=_parse_seeds, default=[42])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.cache_path.exists():
        raise FileNotFoundError(f"Missing cache: {args.cache_path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_variant(args, "NeuroEZ-B-PhysFeat", ["--use_physics_feature_concat", "true"])
    run_variant(
        args,
        "Phys-NeuroEZ-B",
        [
            "--use_physics_prior",
            "true",
            "--physics_logit_bias_beta",
            "0.15",
            "--physics_fusion_weight",
            "0.25",
            "--physics_rank_loss_weight",
            "0.02",
            "--physics_budget_loss_weight",
            "0.02",
        ],
    )


if __name__ == "__main__":
    main()
