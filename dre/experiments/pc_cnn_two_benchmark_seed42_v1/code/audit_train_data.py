"""Train-only PC-CNN data/normalization audit, with no model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from descriptor_norm import apply, save
from patient_bank import IctalBank, OmniTrainBank


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--official-split", type=Path)
    p.add_argument("--train-val-split", type=Path)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.benchmark == "ictal":
        if not args.manifest or digest(args.manifest) != lock["ictal_fold_manifest_sha256"]:
            raise RuntimeError("Ictal frozen fold manifest mismatch")
        bank = IctalBank(args.cache, args.manifest)
        fit, validation = bank.folds[1]["fit"], bank.folds[1]["validation"]
        moments = bank.fit_moments(1)
        example = next(bank.records(fit[0]))
    else:
        if not args.official_split or not args.train_val_split or \
                digest(args.official_split) != lock["omni_official_split_sha256"] or \
                digest(args.train_val_split) != lock["omni_inner_train_val_split_sha256"]:
            raise RuntimeError("Omni official/inner split mismatch")
        bank = OmniTrainBank(args.cache, args.train_val_split, args.official_split)
        fit, validation = bank.patients("inner_train"), bank.patients("inner_val")
        moments = bank.fit_moments()
        example = next(bank.records(fit[0], 0, all_clips=False))
    frozen = moments.freeze(benchmark=args.benchmark, fold=1,
                            fit_patient_count=len(fit), protocol_sha256=digest(args.protocol))
    normalized = apply(example["descriptors"], example["descriptor_mask"], frozen)
    if not np.isfinite(normalized).all() or np.any(normalized[~example["descriptor_mask"]] != 0):
        raise RuntimeError("Descriptor normalization/mask replay failed")
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / f"{args.benchmark}_fold1_descriptor_norm.json", frozen)
    audit = {"status": "PASS", "benchmark": args.benchmark, "fold": 1,
             "fit_patients": len(fit), "validation_patients": len(validation),
             "fit_val_disjoint": not bool(set(fit) & set(validation)),
             "feature_coordinates": 36,
             "nonempty_fitted_coordinates": int(np.sum(moments.count > 0)),
             "source_role_for_example": "fit_only",
             "validation_or_test_used_for_fit": False,
             "final_test_accessed": False}
    (args.output / f"{args.benchmark}_train_data_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
