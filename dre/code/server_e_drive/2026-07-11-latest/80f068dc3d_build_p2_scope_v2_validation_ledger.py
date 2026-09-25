"""Build the deterministic, stratified fixed-validation ledger for SCOPE-v2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_factory import build_outer_splits
from neuroez_c.cane_path_cohort import build_sensitivity80_cohort, subject_center
from neuroez_c.task1_cohort_contract import read_all90_ledger


def _hash(seed: int, fold: int, subject: str) -> str:
    return hashlib.sha256(f"{seed}|{fold}|{subject}".encode("utf-8")).hexdigest()


def _bins(values: dict[str, float], maximum: int = 3) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    unique = sorted({value for _, value in ordered})
    n_bins = min(maximum, len(unique))
    if n_bins <= 1:
        return {subject: 0 for subject, _ in ordered}
    cuts = np.quantile(np.asarray([value for _, value in ordered], dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return {subject: int(np.searchsorted(cuts, value, side="right")) for subject, value in ordered}


def _meta(patient_index: dict[str, Any], subject: str) -> dict[str, float | str]:
    data = patient_index[subject]
    labels = np.asarray(data.get("labels", []), dtype=float)
    valid = labels[labels >= 0]
    return {
        "center": str(data.get("center", subject_center(subject))).lower(),
        "ez_fraction": float(np.mean(valid > .5)) if valid.size else 0.0,
        "channel_count": float(valid.size),
        "seizure_count": float(data.get("n_seizures", data.get("seizure_count", 1))),
    }


def _select_validation(train: list[str], patient_index: dict[str, Any], *, seed: int, fold: int) -> tuple[list[str], dict[str, str]]:
    metas = {subject: _meta(patient_index, subject) for subject in train}
    ez_bin = _bins({subject: float(meta["ez_fraction"]) for subject, meta in metas.items()})
    channel_bin = _bins({subject: float(meta["channel_count"]) for subject, meta in metas.items()})
    seizure_bin = _bins({subject: float(meta["seizure_count"]) for subject, meta in metas.items()})
    strata: dict[str, list[str]] = defaultdict(list)
    fallback: dict[str, str] = {}
    for subject in train:
        center = str(metas[subject]["center"])
        candidates = (
            f"{center}|{ez_bin[subject]}|{channel_bin[subject]}|{seizure_bin[subject]}",
            f"{center}|{ez_bin[subject]}|{channel_bin[subject]}",
            f"{center}|{ez_bin[subject]}",
            center,
        )
        # Sparse cells fall back progressively. This is a stratum assignment,
        # not a global lexicographic sample.
        selected = candidates[-1]
        for candidate in candidates:
            if sum(1 for other in train if candidate == (f"{metas[other]['center']}|{ez_bin[other]}|{channel_bin[other]}|{seizure_bin[other]}" if candidate.count("|") == 3 else f"{metas[other]['center']}|{ez_bin[other]}|{channel_bin[other]}" if candidate.count("|") == 2 else f"{metas[other]['center']}|{ez_bin[other]}" if candidate.count("|") == 1 else str(metas[other]['center']))) >= 2:
                selected = candidate; break
        strata[selected].append(subject); fallback[subject] = selected
    target = int(round(.20 * len(train))); low = int(np.ceil(.15 * len(train))); high = int(np.floor(.25 * len(train)))
    target = min(max(target, low), high)
    selected: set[str] = set()
    for members in strata.values():
        take = int(round(.20 * len(members)))
        if take == 0 and len(members) >= 2:
            take = 1
        selected.update(sorted(members, key=lambda subject: _hash(seed, fold, subject))[:take])
    # Each viable center gets at least one validation patient.
    for center, members in defaultdict(list, {center: [s for s in train if metas[s]["center"] == center] for center in {m["center"] for m in metas.values()}}).items():
        if members and not any(metas[s]["center"] == center for s in selected):
            selected.add(sorted(members, key=lambda subject: _hash(seed, fold, subject))[0])
    available = sorted(set(train) - selected, key=lambda subject: _hash(seed, fold, subject))
    while len(selected) < target and available:
        selected.add(available.pop(0))
    # Trim only when coverage of a represented center remains intact.
    while len(selected) > target:
        removable = [s for s in sorted(selected, key=lambda subject: _hash(seed, fold, subject), reverse=True) if sum(metas[x]["center"] == metas[s]["center"] for x in selected) > 1]
        if not removable:
            break
        selected.remove(removable[0])
    if not (low <= len(selected) <= high):
        raise RuntimeError(f"Fold {fold} validation count {len(selected)} is outside 15%-25% range")
    return sorted(selected), fallback


def _distribution(subjects: list[str], patient_index: dict[str, Any]) -> dict[str, Any]:
    metas = [_meta(patient_index, subject) for subject in subjects]
    return {"by_center": dict(sorted(Counter(str(meta["center"]) for meta in metas).items())), **{f"{key}_{stat}": float(getattr(np, stat)([float(meta[key]) for meta in metas])) if metas else 0.0 for key in ("ez_fraction", "channel_count", "seizure_count") for stat in ("mean", "std")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True); parser.add_argument("--subjects", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--audit", required=True)
    parser.add_argument("--cohort", choices=("sensitivity80", "primary90"), required=True)
    parser.add_argument("--exclude-subjects-file", default=""); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if int(args.seed) != 42:
        raise ValueError("SCOPE validation ledger is fixed to seed 42")
    with open(args.cache, "rb") as handle:
        payload = pickle.load(handle)
    source_index = payload["patient_index"]
    all90 = read_all90_ledger(args.subjects)
    if set(all90) - set(source_index):
        raise RuntimeError("Feature cache is missing subjects from the frozen All90 ledger")
    all90_index = {subject: source_index[subject] for subject in all90}
    outer = build_outer_splits(all90_index, n_splits=5, random_seed=42)
    if args.cohort == "sensitivity80":
        patient_index, outer, _ = build_sensitivity80_cohort(all90_index, outer, args.exclude_subjects_file)
    else:
        patient_index = all90_index
    rows: list[dict[str, Any]] = []; audits: list[dict[str, Any]] = []
    for split in outer:
        fold = int(split["fold_idx"]); train = sorted(map(str, split["train_subjects"])); test = sorted(map(str, split["test_subjects"]))
        validation, strata = _select_validation(train, patient_index, seed=42, fold=fold); fit = sorted(set(train) - set(validation))
        if set(fit) & set(validation) or set(fit) & set(test) or set(validation) & set(test) or set(fit) | set(validation) != set(train) or set(test) != set(split["test_subjects"]):
            raise RuntimeError(f"Ledger integrity failure in fold {fold}")
        for role, subjects in (("fit", fit), ("validation", validation), ("test", test)):
            for subject in subjects:
                meta = _meta(patient_index, subject)
                rows.append({"outer_fold": fold, "subject_id": subject, "center": meta["center"], "role": role, "stratum": strata.get(subject, "outer_test")})
        audits.append({"outer_fold": fold, "fit_count": len(fit), "validation_count": len(validation), "test_count": len(test), "fit": _distribution(fit, patient_index), "validation": _distribution(validation, patient_index), "test": _distribution(test, patient_index), "outer_train_coverage": True, "outer_test_match": True})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outer_fold", "subject_id", "center", "role", "stratum"]); writer.writeheader(); writer.writerows(rows)
    ledger_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    audit = {"cohort": args.cohort, "seed": 42, "n_patients": len(patient_index), "n_outer_splits": 5, "outer_fold_match": True, "ledger_sha256": ledger_hash, "folds": audits, "status": "passed"}
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True); Path(args.audit).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
