from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
for path in (REPO, PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neuroez_c.task2.audit import audit_inputs
from neuroez_c.task2.clinical_target import build_clinical_target_lookup
from neuroez_c.task2.data import load_cache
from neuroez_c.task2.p2_adapter import P23Runtime, P2ExportAdapter, load_p2_args, locate_p2_config
from neuroez_c.task2.protocol import manifest_patient_keys
from neuroez_c.task2.training import prepare_examples


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit P2-Q10-NPAM inputs and leakage boundaries")
    value.add_argument("--outcome_table", default=os.getenv("DRE_TASK2_OUTCOME_TABLE", "cache://patient_index"))
    value.add_argument("--feature_cache", default=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH"), required=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH") is None)
    value.add_argument("--raw_cache", default=os.getenv("DRE_TASK1_RAW_CACHE_PATH"), required=os.getenv("DRE_TASK1_RAW_CACHE_PATH") is None)
    value.add_argument("--p2_checkpoint_root", default=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT"))
    value.add_argument("--p2_runtime_root", default=os.getenv("DRE_TASK1_P2_RUNTIME_ROOT", str(REPO / "P23_TRN_NEZ_80")))
    value.add_argument("--p2_config")
    value.add_argument("--fold_manifest", default=os.getenv("DRE_TASK2_FOLD_MANIFEST"))
    value.add_argument("--output_dir", default=os.getenv("DRE_TASK2_OUTPUT_DIR", str(PROJECT / "outputs")))
    value.add_argument("--exclusion_manifest", default=os.getenv("DRE_TASK2_EXCLUSION_MANIFEST", str(PROJECT / "configs" / "data_exclusions.csv")))
    value.add_argument("--device", default="cpu")
    value.add_argument("--skip_embedding_export", action="store_true")
    value.add_argument("--strict", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    feature = load_cache(args.feature_cache)
    raw = load_cache(args.raw_cache)
    summary = audit_inputs(feature, raw, feature_path=args.feature_cache, raw_path=args.raw_cache, outcome_table=args.outcome_table, fold_manifest=args.fold_manifest, p2_checkpoint_root=args.p2_checkpoint_root, output_dir=args.output_dir, exclusion_manifest=args.exclusion_manifest, strict=args.strict)
    target_lookup, target_alignment, target_distribution = build_clinical_target_lookup(feature, sorted(manifest_patient_keys(args.fold_manifest)), strict=args.strict)
    audit_output=Path(args.output_dir); target_alignment.to_csv(audit_output/"clinical_target_alignment_audit.csv",index=False); target_distribution.to_csv(audit_output/"clinical_target_distribution_audit.csv",index=False)
    summary["clinical_target"]={"n_valid_patients":len(target_lookup),"alignment_passed":bool((target_alignment.alignment_status=="ok").all()),"minimum_match_fraction":float(target_alignment.match_fraction.min()),"success_with_mask":int(((target_alignment.outcome_label==1)&(target_alignment.alignment_status=="ok")).sum()),"failure_with_mask":int(((target_alignment.outcome_label==0)&(target_alignment.alignment_status=="ok")).sum())}
    if args.p2_checkpoint_root and not args.skip_embedding_export:
        runtime = P23Runtime(args.p2_runtime_root)
        config = locate_p2_config(args.p2_checkpoint_root, args.p2_config)
        p2_args = load_p2_args(config, feature_cache=args.feature_cache)
        subject = sorted(feature["patient_index"])[0]
        examples, _ = prepare_examples(runtime, feature, p2_args, normalizer_subjects=[subject], output_subjects=[subject])
        batch = runtime.collate(examples)
        checkpoint_audit = Path(args.output_dir) / "task2_npam_checkpoint_audit.csv"
        table = pd.read_csv(checkpoint_audit)
        fold_exports = []
        for row_index, row in table.iterrows():
            fold = int(row["outer_fold"])
            checkpoint = Path(row["checkpoint_path"])
            adapter = P2ExportAdapter(runtime, p2_args, checkpoint, device=args.device)
            shapes = adapter.audit_shapes(adapter(batch))
            metadata_matches = adapter.checkpoint_outer_fold in {0, fold}
            table.loc[row_index, "checkpoint_metadata_outer_fold"] = adapter.checkpoint_outer_fold
            table.loc[row_index, "outer_fold_matches_checkpoint"] = metadata_matches
            table.loc[row_index, "embedding_export_verified"] = True
            fold_exports.append({"outer_fold": fold, "checkpoint": str(checkpoint.resolve()), "checkpoint_metadata_outer_fold": adapter.checkpoint_outer_fold, "outer_fold_matches_checkpoint": metadata_matches, "shapes": shapes})
            if args.strict and not metadata_matches:
                raise ValueError(f"P2 checkpoint metadata fold mismatch: expected {fold}, got {adapter.checkpoint_outer_fold}")
        (Path(args.output_dir) / "task2_npam_p2_export_shape_audit.json").write_text(json.dumps({"subject": subject, "fold_exports": fold_exports}, indent=2), encoding="utf-8")
        table.to_csv(checkpoint_audit, index=False)
        summary["p2_embedding_export_shapes"] = fold_exports[0]["shapes"] if fold_exports else {}
        summary["p2_embedding_export_verified_folds"] = [row["outer_fold"] for row in fold_exports]
        summary["p2_checkpoint_metadata_match_all"] = all(row["outer_fold_matches_checkpoint"] for row in fold_exports)
        (Path(args.output_dir) / "task2_npam_input_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
