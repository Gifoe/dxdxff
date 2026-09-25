from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class A9v14CandidateConfig:
    """Registry entry for one A9v14 candidate experiment."""

    name: str
    module_tags: tuple[str, ...]
    args: dict[str, Any]
    diagnostic_only: bool = False


def _base_disabled() -> dict[str, Any]:
    return {
        "use_patient_context_reranker": False,
        "reranker_use_rank_features": False,
        "reranker_use_patient_gate": False,
        "use_multi_seizure_consistency": False,
        "use_shaft_local_residual": False,
        "use_clinical_mixture_head": False,
        "run_source_propagation_diagnostic": False,
    }


def _entry(name: str, module_tags: tuple[str, ...], overrides: dict[str, Any], *, diagnostic_only: bool = False) -> A9v14CandidateConfig:
    args = _base_disabled()
    args.update(overrides)
    return A9v14CandidateConfig(name=name, module_tags=module_tags, args=args, diagnostic_only=diagnostic_only)


CONFIGS: dict[str, A9v14CandidateConfig] = {
    "A9v3_Reproduce": _entry("A9v3_Reproduce", ("baseline",), {}),
    "A9v14_M1_DeepsetReranker": _entry(
        "A9v14_M1_DeepsetReranker",
        ("M1",),
        {
            "use_patient_context_reranker": True,
            "reranker_type": "deepset",
            "reranker_use_rank_features": False,
        },
    ),
    "A9v14_M1_SetTransformerRank": _entry(
        "A9v14_M1_SetTransformerRank",
        ("M1",),
        {
            "use_patient_context_reranker": True,
            "reranker_type": "set_transformer",
            "reranker_use_rank_features": True,
        },
    ),
    "A9v14_M2_ConsistencyOnly": _entry(
        "A9v14_M2_ConsistencyOnly",
        ("M2",),
        {"use_multi_seizure_consistency": True},
    ),
    "A9v14_M3_LocalOnly": _entry(
        "A9v14_M3_LocalOnly",
        ("M3",),
        {"use_shaft_local_residual": True, "local_window": 1},
    ),
    "A9v14_M4_MixtureOnly": _entry(
        "A9v14_M4_MixtureOnly",
        ("M4",),
        {"use_clinical_mixture_head": True},
    ),
    "A9v14_M1M2_RankConsistency": _entry(
        "A9v14_M1M2_RankConsistency",
        ("M1", "M2"),
        {
            "use_patient_context_reranker": True,
            "reranker_type": "set_transformer",
            "reranker_use_rank_features": True,
            "use_multi_seizure_consistency": True,
        },
    ),
    "A9v14_M1M2M3_RankConsistencyLocal": _entry(
        "A9v14_M1M2M3_RankConsistencyLocal",
        ("M1", "M2", "M3"),
        {
            "use_patient_context_reranker": True,
            "reranker_type": "set_transformer",
            "reranker_use_rank_features": True,
            "use_multi_seizure_consistency": True,
            "use_shaft_local_residual": True,
            "local_window": 1,
        },
    ),
    "A9v14_M1M2M4_RankConsistencyMixture": _entry(
        "A9v14_M1M2M4_RankConsistencyMixture",
        ("M1", "M2", "M4"),
        {
            "use_patient_context_reranker": True,
            "reranker_type": "set_transformer",
            "reranker_use_rank_features": True,
            "use_multi_seizure_consistency": True,
            "use_clinical_mixture_head": True,
        },
    ),
    "A9v14_AutoTop2_Combo": _entry(
        "A9v14_AutoTop2_Combo",
        ("auto_top2",),
        {
            "use_patient_context_reranker": True,
            "reranker_type": "set_transformer",
            "reranker_use_rank_features": True,
            "use_multi_seizure_consistency": True,
        },
    ),
    "A9v14_M5_SourcePropagationDiagnostic_A9v3": _entry(
        "A9v14_M5_SourcePropagationDiagnostic_A9v3",
        ("M5",),
        {"run_source_propagation_diagnostic": True},
        diagnostic_only=True,
    ),
    "A9v14_M5_SourcePropagationDiagnostic_BestCandidate": _entry(
        "A9v14_M5_SourcePropagationDiagnostic_BestCandidate",
        ("M5",),
        {"run_source_propagation_diagnostic": True},
        diagnostic_only=True,
    ),
}


DEFAULT_A9V14_ARGS: dict[str, Any] = {
    "positive_label": "ez",
    "drop_high_ez_fraction_lzu": False,
    "enforce_fixed_all90_protocol": True,
    "split_strategy": "5fold",
    "n_splits": 5,
    "random_seed": 42,
    "reranker_hidden_dim": 64,
    "reranker_num_layers": 1,
    "reranker_dropout": 0.4,
    "reranker_residual_scale": 0.2,
    "reranker_delta_l2_weight": 0.001,
    "reranker_gate_l2_weight": 0.001,
    "consistency_hidden_dim": 64,
    "consistency_dropout": 0.1,
    "consistency_residual_scale": 0.15,
    "consistency_delta_l2_weight": 0.001,
    "local_window": 1,
    "local_hidden_dim": 64,
    "local_dropout": 0.1,
    "local_residual_scale": 0.10,
    "local_delta_l2_weight": 0.001,
    "mixture_core_loss_weight": 0.2,
    "mixture_broad_loss_weight": 0.5,
    "mixture_final_loss_weight": 1.0,
    "freeze_a9v3_backbone": True,
    "base_aux_loss_weight": 0.2,
    "final_loss_weight": 1.0,
    "module_dropout": 0.1,
    "candidate_module_seed": 42,
    "save_record_level_outputs": True,
    "save_prediction_ledger": True,
    "save_module_diagnostics": True,
}


def list_candidate_configs() -> list[str]:
    return list(CONFIGS.keys())


def get_candidate_config(name: str) -> A9v14CandidateConfig:
    try:
        return CONFIGS[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown A9v14 candidate config {name!r}. Available: {list_candidate_configs()}") from exc


def build_run_args(config_name: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    run_args = dict(DEFAULT_A9V14_ARGS)
    run_args.update(get_candidate_config(config_name).args)
    if extra:
        run_args.update(extra)
    return run_args


def as_cli_args(config_name: str, extra: dict[str, Any] | None = None) -> list[str]:
    args = build_run_args(config_name, extra)
    cli: list[str] = []
    for key in sorted(args):
        value = args[key]
        if value is None:
            continue
        if isinstance(value, bool):
            cli.append(f"--{key}" if value else f"--no-{key}")
        else:
            cli.extend([f"--{key}", str(value)])
    return cli


def main() -> None:
    parser = argparse.ArgumentParser(description="List or dump A9v14 candidate module configs.")
    parser.add_argument("--list", action="store_true", default=False)
    parser.add_argument("--config_name", type=str, default="")
    args = parser.parse_args()
    if args.list:
        print("\n".join(list_candidate_configs()))
        return
    if args.config_name:
        print(json.dumps(build_run_args(args.config_name), indent=2, ensure_ascii=False, sort_keys=True))
        return
    print(json.dumps({name: build_run_args(name) for name in list_candidate_configs()}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
