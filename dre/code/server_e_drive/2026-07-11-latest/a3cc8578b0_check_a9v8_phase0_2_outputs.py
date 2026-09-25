from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check A9v8 Phase0-2 regenerated output invariants.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--expect_fold_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    root = Path(args.root)
    teacher = root / "teacher" / "a9v3_oof_channel_scores.csv"
    if not teacher.exists():
        raise FileNotFoundError(f"Missing teacher CSV: {teacher}")
    center_audit = root / "teacher" / "center_mapping_audit.csv"
    if not center_audit.exists():
        raise FileNotFoundError(f"Missing center mapping audit: {center_audit}")

    candidates = [
        root / "teacher" / "latent_core_targets_phys",
        root / "teacher" / "latent_core_targets_teacher_only",
        root / "teacher" / "latent_core_targets",
    ]
    target_dir = next((path for path in candidates if (path / "latent_core_audit.json").exists()), None)
    if target_dir is None:
        raise FileNotFoundError("Missing latent_core_audit.json under teacher latent target directories.")
    audit = _read_json(target_dir / "latent_core_audit.json")
    observed = [int(x) for x in audit.get("fold_ids_observed", [])]
    expected = [int(x) for x in args.expect_fold_ids]
    if observed != expected:
        raise ValueError(f"Unexpected fold_ids_observed={observed}; expected {expected}")
    for fold in expected:
        if (target_dir / f"latent_core_targets_fold0_train.csv").exists():
            raise ValueError("fold0 train file must not be generated.")
        path = target_dir / f"latent_core_targets_fold{fold}_train.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold train target: {path}")
        info = audit.get("fold_train_files", {}).get(f"fold{fold}", {})
        if bool(info.get("contains_heldout_fold", True)):
            raise ValueError(f"Fold {fold} train target contains heldout fold according to audit.")
    if audit.get("broad_center_rule") != "normalized_center_string":
        raise ValueError("broad_center_rule must be normalized_center_string.")
    phys_available = bool(audit.get("phys_core_score_available"))
    teacher_only = bool(audit.get("teacher_only_fallback_used"))
    if target_dir.name.endswith("_phys"):
        if not phys_available:
            raise ValueError("Strict phys target directory must have phys_core_score_available=true.")
        if audit.get("feature_name_source_used") not in {"cache", "cli"}:
            raise ValueError("Phys target must record feature_name_source_used as cache or cli.")
        if not audit.get("matched_feature_names"):
            raise ValueError("Phys target must have non-empty matched_feature_names.")
    if not phys_available and not teacher_only:
        raise ValueError("phys unavailable must be explicit teacher_only fallback.")
    print(f"A9v8 Phase0-2 output check passed: {target_dir}")


if __name__ == "__main__":
    main()
