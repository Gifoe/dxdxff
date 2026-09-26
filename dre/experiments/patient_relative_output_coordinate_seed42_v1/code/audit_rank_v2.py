"""Frozen numerical amendment: audit EZ ranking on float64 logit coordinates only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


EXPERIMENT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("COORD_RUNTIME", ""))
ORIGINAL_LOCK_SHA256 = "a28aa5d552e5ff830a9b04eeafe4520a4fd485b3b19420eb79b272f2b6f4a90e"
AMENDMENT_SHA256 = "c13c651d2417d7a6798bcf2772059288bde11a8a99f383a600f505bbcac59d13"
VARIANTS = ("C1_CENTERED", "C2_ROBUSTZ")
METRICS = ("EZ_AUPRC", "EZ_AUROC", "EZ_MRR", "Top1_is_EZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranking_metrics(y_ez: np.ndarray, r_ez: np.ndarray) -> dict[str, float]:
    ordered = np.argsort(-r_ez, kind="stable")
    positive_ranks = np.flatnonzero(y_ez[ordered] == 1)
    return {
        "EZ_AUPRC": float(average_precision_score(y_ez, r_ez)),
        "EZ_AUROC": float(roc_auc_score(y_ez, r_ez)),
        "EZ_MRR": float(1.0 / (int(positive_ranks[0]) + 1)) if positive_ranks.size else 0.0,
        "Top1_is_EZ": float(y_ez[int(ordered[0])] == 1),
    }


def main() -> None:
    if not os.environ.get("COORD_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("Set absolute COORD_RUNTIME")
    if file_sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != ORIGINAL_LOCK_SHA256:
        raise RuntimeError("Original scientific protocol changed")
    if file_sha256(EXPERIMENT / "RANK_AUDIT_PROTOCOL_AMENDMENT.json") != AMENDMENT_SHA256:
        raise RuntimeError("Rank audit amendment changed")
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if source.get("pass") is not True or source.get("outer_test_accessed") is not False:
        raise RuntimeError("SOURCE_A1_VLOO_REPRODUCTION_FAILED")
    if abs(source["reproduced_mean"] - 0.6259962097139906) > 1e-6:
        raise RuntimeError("SOURCE_A1_VLOO_REPRODUCTION_FAILED")
    expected_folds = [0.6549997115717437, 0.6410958089865288, 0.6101185971670673,
                      0.6329791804569156, 0.5907877503876969]
    if len(source["folds"]) != 5 or any(abs(row["reproduced"] - expected_folds[i]) > 1e-6
                                        for i, row in enumerate(source["folds"])):
        raise RuntimeError("SOURCE_A1_VLOO_REPRODUCTION_FAILED")

    cases = 0
    source_probability_tie_cases = 0
    min_spearman = {variant: 1.0 for variant in VARIANTS}
    pairwise_order_violations = {variant: 0 for variant in VARIANTS}
    pairwise_order_changed_cases = {variant: 0 for variant in VARIANTS}
    stable_sort_mismatch_cases = {variant: 0 for variant in VARIANTS}
    metric_changed_cases = {variant: 0 for variant in VARIANTS}
    max_metric_delta = {variant: {metric: 0.0 for metric in METRICS} for variant in VARIANTS}

    for fold in range(1, 6):
        for epoch in range(1, 31):
            payload = json.loads((RUNTIME / "validation_logits_private" / f"fold_{fold}" /
                                  f"epoch_{epoch:02d}.json").read_text(encoding="utf-8"))
            if payload["epoch"] != epoch or len(payload["patients"]) != 13:
                raise RuntimeError("Private source validation logits are incomplete")
            for patient in payload["patients"]:
                a = np.asarray(patient["logits_nez"], dtype=np.float64)
                y_ez = np.asarray(patient["labels_ez"], dtype=np.int8)
                if len(a) < 2 or len(a) != len(y_ez) or not np.isfinite(a).all() or set(np.unique(y_ez)) != {0, 1}:
                    raise RuntimeError("Invalid valid-channel validation record")
                median = np.median(a)
                mad = np.median(np.abs(a - median))
                scale = max(1.4826 * mad, 1e-6)
                scores = {
                    "C0_RAW": -a,
                    "C1_CENTERED": -(a - median),
                    "C2_ROBUSTZ": -((a - median) / scale),
                }
                reference = ranking_metrics(y_ez, scores["C0_RAW"])
                reference_order = np.argsort(-scores["C0_RAW"], kind="stable")
                upper = np.triu_indices(len(a), k=1)
                reference_sign = np.sign(scores["C0_RAW"][:, None] - scores["C0_RAW"][None, :])[upper]
                p_ez_float32 = np.asarray(patient["score_ez_core"], dtype=np.float32)
                if len(np.unique(p_ez_float32)) < len(np.unique(a)):
                    source_probability_tie_cases += 1
                for variant in VARIANTS:
                    r = scores[variant]
                    rho = float(spearmanr(scores["C0_RAW"], r).statistic)
                    min_spearman[variant] = min(min_spearman[variant], rho) if np.isfinite(rho) else -1.0
                    signs = np.sign(r[:, None] - r[None, :])[upper]
                    violations = int(np.count_nonzero(signs != reference_sign))
                    pairwise_order_violations[variant] += violations
                    pairwise_order_changed_cases[variant] += int(violations != 0)
                    stable_sort_mismatch_cases[variant] += int(not np.array_equal(
                        reference_order, np.argsort(-r, kind="stable")))
                    measured = ranking_metrics(y_ez, r)
                    changed = False
                    for metric in METRICS:
                        delta = abs(measured[metric] - reference[metric])
                        max_metric_delta[variant][metric] = max(max_metric_delta[variant][metric], delta)
                        tolerance = 0.0 if metric == "Top1_is_EZ" else 1e-12
                        changed |= delta > tolerance
                    metric_changed_cases[variant] += int(changed)
                cases += 1
        print(f"[RANK V2] fold={fold} audited={cases}", flush=True)

    passed = cases == 1950 and all(
        min_spearman[variant] >= 0.999999999 and
        pairwise_order_violations[variant] == 0 and
        stable_sort_mismatch_cases[variant] == 0 and
        metric_changed_cases[variant] == 0
        for variant in VARIANTS
    )
    audit = {
        "pass": passed,
        "terminal": "RANK_INVARIANCE_V2_PASSED" if passed else "OUTPUT_COORDINATE_RANK_INVARIANCE_V2_FAILED",
        "patient_epoch_cases_per_transform": cases,
        "canonical_rank_score": "float64 EZ-positive negative of raw/centered/robust-z NEZ logits; no sigmoid",
        "minimum_spearman": min_spearman,
        "pairwise_order_violations": pairwise_order_violations,
        "pairwise_order_changed_cases": pairwise_order_changed_cases,
        "stable_sort_mismatch_cases": stable_sort_mismatch_cases,
        "maximum_absolute_metric_delta": max_metric_delta,
        "ranking_metric_changed_cases": metric_changed_cases,
        "source_float32_probability_tie_cases_with_distinct_logits_diagnostic_only": source_probability_tie_cases,
        "tolerance": {"spearman_min": 0.999999999, "ranking_metrics_max_abs": 1e-12,
                      "Top1_is_EZ": "exact", "pairwise_order": "exact", "stable_sort": "exact"},
        "original_protocol_sha256": ORIGINAL_LOCK_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "outer_test_accessed": False,
    }
    (EXPERIMENT / "RANK_INVARIANCE_AUDIT_V2.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if not passed:
        raise RuntimeError("OUTPUT_COORDINATE_RANK_INVARIANCE_V2_FAILED")


if __name__ == "__main__":
    main()
