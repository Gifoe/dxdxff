"""Run the pre-registered A12-VCSN suite on explicit frozen V3 and cache inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a12_vcsn.config import A12Config
from a12_vcsn.io import load_hnc_oof_ledger, load_v3_ledger, load_window_feature_store
from a12_vcsn.protocol import audit_cache_subject_coverage
from a12_vcsn.provenance import git_state
from a12_vcsn.suite import ALL_VARIANTS, run_a12_suite
from a12_vcsn.utils import environment_snapshot, write_json


def _summary_metric(path: str | None) -> float | None:
    if not path: return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"patient_macro_f1", "old_v3_patient_macro_f1"} and isinstance(item, (int, float)): return float(item)
                found = walk(item)
                if found is not None: return found
        return None
    metric = walk(data)
    if metric is None: raise ValueError("old-v3-summary lacks patient_macro_f1")
    return metric


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional JSON/YAML runtime config; CLI and A12_* environment values take precedence.")
    parser.add_argument("--old-v3-ledger", default=os.environ.get("A12_OLD_V3_LEDGER"))
    parser.add_argument("--window-cache-path", default=os.environ.get("A12_WINDOW_CACHE_PATH"))
    parser.add_argument("--cache-invalid-record-policy", "--invalid-record-policy", dest="invalid_record_policy", choices=["fail", "drop"], default="drop")
    parser.add_argument("--output-dir", default=os.environ.get("A12_OUTPUT_DIR"))
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--hnc-ledger", default=os.environ.get("A12_HNC_LEDGER"))
    parser.add_argument("--hnc-score-semantics", choices=["p_nez", "p_ez", "residual_nez", "residual_ez"], default=None)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", type=_csv_ints, default=(42, 43, 44))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict-device", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-rebuild-pair-cache", action="store_true", default=False)
    parser.add_argument("--candidate-config", default=None, help="Optional JSON config file; unknown keys are rejected.")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--max-swaps", type=int, default=1)
    parser.add_argument("--max-outer-folds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--audit-only", action="store_true", default=False)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-old-v3-macro-f1", type=float, default=None)
    parser.add_argument("--old-v3-summary", default=None)
    parser.add_argument("--anchor-parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--risk-lambda", type=float, default=.10)
    parser.add_argument("--uncertainty-lambda", type=float, default=0.)
    parser.add_argument("--minimum-action-coverage", type=float, default=0.)
    parser.add_argument("--patient-harm-limit", type=float, default=.15)
    parser.add_argument("--max-eject-candidates", type=int, default=12)
    parser.add_argument("--max-add-candidates", type=int, default=30)
    parser.add_argument("--max-pairs-per-patient", type=int, default=360)
    parser.add_argument("--anchor-quantile", type=float, default=.50)
    parser.add_argument("--anchor-min-channels", type=int, default=8)
    parser.add_argument("--minimum-anchor-channel-coverage", type=float, default=0.)
    parser.add_argument("--minimum-anchor-feature-finite-ratio", type=float, default=0.)
    parser.add_argument("--trajectory-mode", choices=["trajectory_compact", "trajectory_full"], default="trajectory_compact")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--gradient-clip", type=float, default=1.)
    parser.add_argument("--catboost-depth", type=int, default=3)
    parser.add_argument("--catboost-learning-rate", type=float, default=.05)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=20.)
    parser.add_argument("--catboost-max-iterations", type=int, default=800)
    parser.add_argument("--catboost-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--smoke-mode", action="store_true", default=False)
    return parser


def _load_runtime_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; JSON config remains available") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("runtime config must be a JSON/YAML mapping")
    return payload


def _resolve_runtime_paths(args: argparse.Namespace) -> argparse.Namespace:
    payload = _load_runtime_config(args.config)
    for name in ("old_v3_ledger", "window_cache_path", "hnc_ledger", "old_v3_summary", "output_dir"):
        if getattr(args, name, None) is None and payload.get(name) is not None:
            setattr(args, name, str(payload[name]))
    missing = [name for name in ("old_v3_ledger", "window_cache_path", "output_dir") if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"missing runtime paths {missing}; provide CLI, JSON/YAML config, or A12_* environment variables")
    if args.max_swaps < 0:
        raise ValueError("max-swaps must be nonnegative")
    return args


def _filter_allowed_subjects(ledger: pd.DataFrame, path: str | None) -> pd.DataFrame:
    if path is None:
        return ledger
    allowed = pd.read_csv(path)
    if "subject_id" not in allowed.columns:
        raise ValueError("allowed-subjects-ledger must contain subject_id")
    result = ledger[ledger["subject_id"].astype(str).isin(set(allowed["subject_id"].astype(str)))].copy()
    if result.empty:
        raise ValueError("allowed-subjects-ledger removed all V3 subjects")
    return result


def _attach_hnc(ledger: pd.DataFrame, path: str | None, *, semantics: str | None = None, strict: bool = True) -> tuple[pd.DataFrame, bool, dict[str, object] | None]:
    if path is None:
        return ledger, False, None
    try:
        hnc, audit = load_hnc_oof_ledger(path, semantics=semantics)
    except Exception as exc:
        if strict:
            raise
        return ledger, False, {"status": "skipped", "reason": str(exc), "path": str(Path(path).resolve())}
    joined = ledger.merge(hnc, left_on=["subject_id", "outer_fold", "channel_name_norm"], right_on=["subject_id", "fold_idx", "channel_name_norm"], how="left", validate="one_to_one", suffixes=("", "_hnc"))
    if joined["optional_hnc_score"].isna().any():
        if strict:
            raise ValueError("HNC OOF ledger does not cover every frozen V3 subject-channel key")
        return ledger, False, {**audit, "status": "skipped", "reason": "HNC OOF ledger does not cover every normalized frozen V3 subject-channel key"}
    return joined.drop(columns=["fold_idx", "channel_name_original_hnc"], errors="ignore"), True, audit


def main(argv: list[str] | None = None) -> None:
    args = _resolve_runtime_paths(build_parser().parse_args(argv))
    output = Path(args.output_dir)
    ledger, resolved = load_v3_ledger(args.old_v3_ledger, strict=args.strict)
    ledger = _filter_allowed_subjects(ledger, args.allowed_subjects_ledger)
    channel_features, trajectories, cache_audit = load_window_feature_store(args.window_cache_path, strict=args.strict, invalid_record_policy=args.invalid_record_policy)
    cache_coverage = audit_cache_subject_coverage(ledger, channel_features, strict=args.strict)
    ledger, hnc_available, hnc_audit = _attach_hnc(ledger, args.hnc_ledger, semantics=args.hnc_score_semantics, strict=args.strict)
    variants = ALL_VARIANTS if args.variants.strip().lower() == "all" else tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"unknown A12 variants: {unknown}")
    expected = args.expected_old_v3_macro_f1 if args.expected_old_v3_macro_f1 is not None else _summary_metric(args.old_v3_summary)
    config = A12Config(seeds=args.seeds, inner_folds=args.inner_folds, max_swaps=args.max_swaps, strict=args.strict, resume=args.resume, device=args.device, strict_device=args.strict_device, num_workers=args.num_workers, variants=tuple(variants), expected_old_v3_macro_f1=expected, old_v3_summary=args.old_v3_summary, anchor_parity_tolerance=args.anchor_parity_tolerance, risk_lambda=args.risk_lambda, uncertainty_lambda=args.uncertainty_lambda, minimum_action_coverage=args.minimum_action_coverage, patient_harm_limit=args.patient_harm_limit, max_eject_candidates=args.max_eject_candidates, max_add_candidates=args.max_add_candidates, max_pairs_per_patient=args.max_pairs_per_patient, force_rebuild_pair_cache=args.force_rebuild_pair_cache, anchor_quantile=args.anchor_quantile, anchor_min_channels=args.anchor_min_channels, minimum_anchor_channel_coverage=args.minimum_anchor_channel_coverage, minimum_anchor_feature_finite_ratio=args.minimum_anchor_feature_finite_ratio, trajectory_mode=args.trajectory_mode, cache_invalid_record_policy=args.invalid_record_policy, batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience, learning_rate=args.learning_rate, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, dropout=args.dropout, gradient_clip=args.gradient_clip, catboost_depth=args.catboost_depth, catboost_learning_rate=args.catboost_learning_rate, catboost_l2_leaf_reg=args.catboost_l2_leaf_reg, catboost_max_iterations=args.catboost_max_iterations, catboost_early_stopping_rounds=args.catboost_early_stopping_rounds, smoke_mode=args.smoke_mode)
    if args.candidate_config:
        overrides = json.loads(Path(args.candidate_config).read_text(encoding="utf-8"))
        unknown_config = sorted(set(overrides) - set(config.to_dict()))
        if unknown_config:
            raise ValueError(f"unknown candidate-config fields: {unknown_config}")
        config = replace(config, **overrides)
    write_json(output / "audit" / "resolved_schema.json", resolved)
    write_json(output / "environment.json", environment_snapshot())
    write_json(output / "git_state.json", git_state())
    write_json(output / "audit" / "cache_audit.json", cache_audit)
    filter_audit = cache_audit.get("filter_audit", {})
    excluded = pd.DataFrame(filter_audit.get("excluded_records_detail", []))
    retained = pd.DataFrame(filter_audit.get("retained_records_detail", []))
    patient_coverage = pd.DataFrame(filter_audit.get("patient_coverage", []))
    excluded.to_csv(output / "audit" / "excluded_cache_records.csv", index=False)
    retained.to_csv(output / "audit" / "retained_cache_records.csv", index=False)
    patient_coverage.to_csv(output / "audit" / "patient_cache_coverage_after_filter.csv", index=False)
    pd.concat([retained.assign(status="retained"), excluded.assign(status="excluded")], ignore_index=True).to_csv(output / "audit" / "cache_record_filter_audit.csv", index=False)
    write_json(output / "audit" / "cache_record_filter_summary.json", {key: value for key, value in filter_audit.items() if not key.endswith("_detail") and key != "patient_coverage"})
    write_json(output / "audit" / "cache_subject_coverage.json", cache_coverage)
    if hnc_audit:
        write_json(output / "audit" / "hnc_oof_audit.json", hnc_audit)
    if args.audit_only or args.dry_run:
        print(json.dumps({"mode": "audit-only" if args.audit_only else "dry-run", "n_subjects": int(ledger.subject_id.nunique()), "cache_feature_dim": cache_audit["feature_dim"], "output_dir": str(output.resolve())}, ensure_ascii=False))
        return
    folds = sorted(int(item) for item in ledger.outer_fold.unique())
    evaluation_folds = set(folds[: args.max_outer_folds]) if args.max_outer_folds else None
    result = run_a12_suite(ledger, output_dir=output, config=config, cache_audit=cache_audit, hnc_available=hnc_available, hnc_audit=hnc_audit, resolved_schema=resolved, channel_features=channel_features, trajectories=trajectories, evaluation_folds=evaluation_folds)
    print(json.dumps({"output_dir": result["output_dir"], "variants": {key: value["status"] for key, value in result["variants"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
