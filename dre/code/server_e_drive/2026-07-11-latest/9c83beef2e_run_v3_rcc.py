#!/usr/bin/env python3
"""Run one frozen V3-RCC profile using existing B0 training infrastructure."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.v3_rcc_protocol import file_sha256, read_allowed_subjects, validate_v3_rcc_protocol
from neuroez_c.v3_rcc_profiles import get_v3_rcc_profile
from neuroez_c.v3_rcc_reporting import build_v3_rcc_reports


def _build_training_command(effective: dict[str, object]) -> tuple[list[str], list[str]]:
    """Serialize only arguments accepted by the installed training CLI.

    Base run-args JSON files are historical artifacts and can contain fields
    removed from the current parser.  Passing those through makes a frozen
    RCC run fail before validation or training starts.
    """
    from run_neuroez_c import build_parser

    parser = build_parser()
    actions = {action.dest: action for action in parser._actions}
    aliases = {
        "allowed_subjects_ledger": "allowed-subjects-ledger",
        "allowed_subjects_file": "allowed-subjects-file",
        "require_n_patients": "require-n-patients",
    }
    cmd = [sys.executable, str(REPO / "run_neuroez_c.py")]
    ignored: list[str] = []
    for key, value in sorted(effective.items()):
        action = actions.get(key)
        if action is None:
            ignored.append(key)
            continue
        if value is None or value == "":
            continue
        flag = "--" + aliases.get(key, key)
        if isinstance(value, bool):
            # BooleanOptionalAction accepts --no-foo.  Legacy bool arguments
            # such as drop_high_ez_fraction_lzu require an explicit value.
            if hasattr(action, "option_strings") and f"--no-{key}" in action.option_strings:
                cmd.append(flag if value else f"--no-{key}")
            else:
                cmd.extend([flag, str(value).lower()])
        else:
            cmd.extend([flag, str(value)])
    return cmd, ignored

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-run-args", required=True)
    p.add_argument("--v3_rcc_profile", choices=["R0_BASE", "R1_RANK_COVERAGE", "R2_HYBRID_CALIBRATION"], required=True)
    p.add_argument("--window_cache_path", required=True); p.add_argument("--allowed_subjects_ledger", required=True)
    p.add_argument("--fixed_fold_manifest", required=True); p.add_argument("--require_n_patients", type=int, default=80)
    p.add_argument("--output_dir", required=True); p.add_argument("--baseline_ledger", default=""); p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--max_outer_folds", type=int, default=0); p.add_argument("--dry_run", action="store_true")
    a = p.parse_args(); output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(a.base_run_args).read_text(encoding="utf-8")); profile = get_v3_rcc_profile(a.v3_rcc_profile)
    protocol = validate_v3_rcc_protocol(patient_ids=read_allowed_subjects(a.allowed_subjects_ledger), allowed_subjects_path=a.allowed_subjects_ledger, outer_fold_manifest_path=a.fixed_fold_manifest, require_n_patients=a.require_n_patients, seed=a.random_seed)
    effective = {**base, "config_name": f"V3_RCC_{profile.name}_seed{a.random_seed}", "output_dir": str(output), "window_cache_path": a.window_cache_path, "allowed_subjects_ledger": a.allowed_subjects_ledger, "require_n_patients": a.require_n_patients, "random_seed": a.random_seed, "split_strategy": "5fold", "n_splits": 5, "max_outer_folds": a.max_outer_folds, "positive_label": "ez", "drop_high_ez_fraction_lzu": False, "loss_mode": "patient_balanced_bce", "patient_loss_weighting": "uniform", "use_v3_qbc": False, "use_v3_rcc": True, "v3_rcc_profile": profile.name, "v3_rcc_outer_fold_manifest": a.fixed_fold_manifest, "use_negative_anchor_head": True, "negative_anchor_loss_weight": 0.02, "use_ez_ranking_loss": True, "ez_ranking_loss_weight": 0.05, "use_hard_topk_loss": False, "use_broad_ez_mil_loss": False, "use_a9v8_lcbo": False, "use_two_expert_router": False, "use_feature_separated_two_expert": False, "use_n6_dual_view_ema": False, "use_diffusion_residual": False, "use_view_gated_fusion": False, "train_subject_dropout_count": 0, "dry_run_config_only": a.dry_run}
    audit = {**protocol, "profile": profile.name, "q10": False, "boundary_loss": False, "hnc": False, "robust_tail": False, "scope": False, "balanced_bce_weight": 0.75 if profile.use_hybrid_bce else 1.0, "unweighted_bce_weight": 0.25 if profile.use_hybrid_bce else 0.0, "coverage_weight": 0.03 if profile.use_coverage else 0.0, "pairwise_weight": 0.05, "base_args_hash": file_sha256(a.base_run_args)}
    (output / "v3_rcc_config_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); (output / "fold_protocol_audit.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8"); (output / "run_args.json").write_text(json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8")
    cmd, ignored_base_args = _build_training_command(effective)
    audit["ignored_unsupported_base_args"] = ignored_base_args
    (output / "v3_rcc_config_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if a.dry_run: print(json.dumps(audit, indent=2)); return
    subprocess.run(cmd, cwd=REPO, check=True)
    result = build_v3_rcc_reports(output)
    if profile.name == "R0_BASE":
        reproduction = {"status": "blocked_missing_baseline_ledger", "baseline_ledger": a.baseline_ledger}
        if a.baseline_ledger:
            baseline = Path(a.baseline_ledger)
            if not baseline.is_file(): raise FileNotFoundError(f"Baseline V3 ledger not found: {baseline}")
            frame = __import__('pandas').read_csv(baseline)
            required = {"subject_id", "final_nez_logit"}
            if not required.issubset(frame.columns): raise ValueError(f"Baseline ledger missing {sorted(required-set(frame.columns))}")
            # Current R0 true-K metrics are comparable only after matching the same held-out patient/channel ledger.
            same_patients = int(frame.subject_id.nunique()) == int(result["n_patients"])
            baseline_truek = baseline.parent / "truek_summary.csv"; baseline_formal = baseline.parent / "formal_summary.csv"
            if not baseline_truek.is_file() or not baseline_formal.is_file():
                raise FileNotFoundError("Baseline ledger must sit beside truek_summary.csv and formal_summary.csv")
            bt = __import__('pandas').read_csv(baseline_truek).iloc[0]; bf = __import__('pandas').read_csv(baseline_formal).iloc[0]; current = __import__('pandas').read_csv(output/'formal_summary.csv').iloc[0]
            diffs = {"truek_macro_f1": abs(result["truek_patient_macro_f1"]-float(bt.patient_macro_f1)), "ez_auprc": abs(float(current.patient_ez_auprc)-float(bf.patient_ez_auprc)), "ez_mrr": abs(float(current.patient_ez_mrr)-float(bf.patient_ez_mrr))}
            reproduction = {"status": "passed" if same_patients and max(diffs.values())<=.003 else "failed", "baseline_ledger": str(baseline), "same_patient_count": same_patients, "differences": diffs, "tolerance": .003}
        (output/'r0_reproduction_audit.json').write_text(json.dumps(reproduction,indent=2),encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
if __name__ == "__main__": main()
