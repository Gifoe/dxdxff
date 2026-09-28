"""Exact A1 vs fixed-config S4 matched query evaluation after score freeze.

This is exploratory: A1 historical target outcomes were previously viewed, and
A1 uses historic VLOO checkpoint selection while S4 uses FIT-only selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from run_s4_feasibility import AMENDMENT_SHA, LOCK_SHA, VARIANT
from stage0_source import file_sha, write_json


METRICS = ("ap", "auc", "mrr", "top1", "ndcg", "macro_f1", "ez_f1", "ba", "predicted_ez_fraction")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    valid = array[np.isfinite(array)]
    return float(valid.mean()) if len(valid) else float("nan")


def cluster_stat(values: np.ndarray, ids: list[str], row_ids: list[str], counts: np.ndarray) -> dict:
    index = {sid: i for i, sid in enumerate(ids)}
    numer = np.zeros(len(ids), dtype=float)
    denom = np.zeros(len(ids), dtype=float)
    for value, sid in zip(values, row_ids, strict=True):
        if np.isfinite(value):
            numer[index[sid]] += value
            denom[index[sid]] += 1
    totals = counts @ denom
    draws = np.divide(counts @ numer, totals, out=np.full(len(counts), np.nan), where=totals > 0)
    if np.isfinite(draws).mean() < .99 or denom.sum() == 0:
        raise RuntimeError("Degenerate patient-ID bootstrap")
    return {"mean": float(numer.sum()/denom.sum()),
            "ci_low": float(np.nanquantile(draws, .025)), "ci_high": float(np.nanquantile(draws, .975)),
            "n_estimable_query_rows": int(denom.sum())}


def center(sid: str) -> str:
    prefix = sid.split(":", 1)[0].lower()
    return {"hup": "HUP", "lzu": "LZU", "multicenter": "multi-site", "pediatric": "pediatric_unverified_Fudan"}.get(prefix, "unknown")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--amendment", type=Path, required=True)
    p.add_argument("--freeze-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--srgi-runtime", type=Path, required=True)
    a = p.parse_args()
    if file_sha(a.lock) != LOCK_SHA or file_sha(a.amendment) != AMENDMENT_SHA:
        raise RuntimeError("S4 feasibility protocol changed")
    freeze = json.loads(a.freeze_audit.read_text(encoding="utf-8"))
    manifest_path = a.runtime / "S4_FEASIBILITY_SCORE_FREEZE_PRIVATE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not freeze["pass"] or freeze["amendment_sha256"] != AMENDMENT_SHA or
            freeze["private_manifest_sha256"] != file_sha(manifest_path) or
            manifest["target_label_keys_indexed"] or len(manifest["files"]) != 5):
        raise RuntimeError("Verified five-fold score freeze required before label read")
    for rel, digest in manifest["files"].items():
        if file_sha(a.runtime / "frozen_target_scores" / rel) != digest:
            raise RuntimeError("Frozen target score changed")
    source = a.lock.parent.parent
    os.environ["SRGI_RUNTIME"] = str(a.srgi_runtime)
    sys.path[:0] = [str(source / "active_fewshot_patient_calibration_seed42_v1" / "code")]
    import common as afc
    import run as afr

    if not os.environ.get("SRGI_RUNTIME") or not afc.PRIOR_RUNTIME.is_absolute():
        raise RuntimeError("Exact historical A1 private VLOO runtime unavailable")
    rows = []
    patient_diagnostics = []
    for fold in range(1, 6):
        historical = afc.read_csv(afc.PRIOR_RUNTIME / "private" / f"fold_{fold}" / "A1_VLOO_PRIVATE.csv")
        if len(historical) != 13:
            raise RuntimeError("Historical A1 VLOO fold missing 13 patients")
        with (a.runtime / "frozen_target_scores" / VARIANT / f"fold_{fold}.pkl").open("rb") as stream:
            spectral = pickle.load(stream)
        if set(spectral) != {r["subject_id"] for r in historical}:
            raise RuntimeError("S4/A1 target identity mismatch")
        for reference in historical:
            sid = reference["subject_id"]
            epoch = int(reference["selected_epoch"])
            tau = float(reference["selected_threshold"])
            payload_path = afc.PRIOR_RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}_representations.pkl"
            with payload_path.open("rb") as stream:
                payload = pickle.load(stream)
            if payload["fold"] != fold or payload["epoch"] != epoch or payload["r4_dim"] != 64 or payload["r4_error"] > 1e-6:
                raise RuntimeError("Historical A1 source representation provenance mismatch")
            item = payload["val"][sid]
            y = np.asarray(item["y"], dtype=np.int8)
            margin_a1 = afr.margin_for(item, tau)
            afr.verify_b0(item, tau, margin_a1)
            s4 = spectral[sid]
            if s4["n_channels"] != len(y) or len(s4["logit_nez"]) != len(y):
                raise RuntimeError("S4/A1 channel order or count mismatch")
            margin_s4 = -np.asarray(s4["logit_nez"], dtype=np.float64)
            if not np.isfinite(margin_s4).all():
                raise RuntimeError("Nonfinite S4 target margin")
            ap_a1 = afc.query_metrics(y, margin_a1)["ap"]
            ap_s4 = afc.query_metrics(y, margin_s4)["ap"]
            ranks_a1 = np.argsort(-margin_a1, kind="stable")
            ranks_s4 = np.argsort(-margin_s4, kind="stable")
            patient_diagnostics.append({"fold": fold, "sid": sid, "a1_ap": ap_a1, "s4_ap": ap_s4,
                                        "a1_top1": int(y[ranks_a1[0]] == 1), "s4_top1": int(y[ranks_s4[0]] == 1),
                                        "rank_disagreement_fraction": float(np.mean(ranks_a1 != ranks_s4))})
            for rep in range(20):
                _, query = afc.split_indices(len(y), 42, fold, sid, rep)
                rows.append({"fold": fold, "sid": sid, "rep": rep, "model": "B0_A1",
                             **afc.query_metrics(y[query], margin_a1[query])})
                rows.append({"fold": fold, "sid": sid, "rep": rep, "model": VARIANT,
                             **afc.query_metrics(y[query], margin_s4[query])})
        print(f"MATCHED_QUERY_EVAL fold={fold}/5", flush=True)
    if len(rows) != 2600 or len({r["sid"] for r in rows}) != 47:
        raise RuntimeError("Expected 2 x 65 x 20 rows and 47 unique patients")
    baseline = {(r["fold"], r["sid"], r["rep"]): r for r in rows if r["model"] == "B0_A1"}
    spectral_rows = [r for r in rows if r["model"] == VARIANT]
    ids = sorted({r["sid"] for r in rows})
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    counts = np.zeros((10000, len(ids)), dtype=np.int16)
    np.add.at(counts, (np.arange(10000)[:, None], draws), 1)
    counts = counts.astype(np.float32)
    delta_ap = np.asarray([r["ap"] - baseline[(r["fold"], r["sid"], r["rep"])]["ap"] for r in spectral_rows])
    ci = cluster_stat(delta_ap, ids, [r["sid"] for r in spectral_rows], counts)
    matrix = []
    for name in ("B0_A1", VARIANT):
        part = [r for r in rows if r["model"] == name]
        matrix.append({"model": name, "n_target_cells": 65, "n_unique_patients": 47, "n_fixed_queries": 1300,
                       **{metric: mean(r[metric] for r in part) for metric in METRICS}})
    a1_ap = matrix[0]["ap"]
    if not math.isclose(a1_ap, 0.5767434626151353, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError(f"Historical exact A1 replay mismatch: {a1_ap}")
    fold_rows = []
    for fold in range(1, 6):
        part = [r for r in spectral_rows if r["fold"] == fold]
        fold_rows.append({"fold": fold, "s4_ap": mean(r["ap"] for r in part),
                          "a1_ap": mean(baseline[(r["fold"], r["sid"], r["rep"])]["ap"] for r in part),
                          "delta_ap": mean(r["ap"] - baseline[(r["fold"], r["sid"], r["rep"])]["ap"] for r in part)})
    center_rows = []
    for site in sorted({center(r["sid"]) for r in spectral_rows}):
        part = [r for r in spectral_rows if center(r["sid"]) == site]
        center_rows.append({"center": site, "unique_patients": len({r["sid"] for r in part}),
                            "s4_ap": mean(r["ap"] for r in part),
                            "delta_vs_a1_ap": mean(r["ap"] - baseline[(r["fold"], r["sid"], r["rep"])]["ap"] for r in part)})
    a.output.mkdir(parents=True, exist_ok=True)
    write_csv(a.output / "PRIMARY_MODEL_MATRIX.csv", matrix)
    write_csv(a.output / "FOLD_CONSISTENCY.csv", fold_rows)
    write_csv(a.output / "CENTER_STRATIFIED_METRICS.csv", center_rows)
    write_csv(a.output / "PATIENT_CLUSTER_BOOTSTRAP.csv", [{"comparison": "S4_MINUS_A1", **ci,
               "patient_clusters": 47, "draws": 10000, "seed": 42,
               "delta_mrr": mean(r["mrr"] - baseline[(r["fold"], r["sid"], r["rep"])]["mrr"] for r in spectral_rows),
               "delta_top1": mean(r["top1"] - baseline[(r["fold"], r["sid"], r["rep"])]["top1"] for r in spectral_rows),
               "positive_folds": sum(row["delta_ap"] > 0 for row in fold_rows)}])
    correlation = np.corrcoef([r["a1_ap"] for r in patient_diagnostics],
                              [r["s4_ap"] for r in patient_diagnostics])[0, 1]
    diagnostic = {"patient_ap_correlation": float(correlation) if np.isfinite(correlation) else None,
                  "s4_top1_rescues": sum(r["a1_top1"] == 0 and r["s4_top1"] == 1 for r in patient_diagnostics),
                  "s4_top1_destructions": sum(r["a1_top1"] == 1 and r["s4_top1"] == 0 for r in patient_diagnostics),
                  "mean_rank_position_disagreement": mean(r["rank_disagreement_fraction"] for r in patient_diagnostics),
                  "n_target_cells": 65, "posthoc_only": True}
    write_json(a.output / "ERROR_COMPLEMENTARITY.json", diagnostic)
    s4_ap = matrix[1]["ap"]
    decision = ("STOP_NO_FUSION" if s4_ap < .55 else
                "INSPECT_COMPLEMENTARITY" if s4_ap < .57 else
                "FUSION_ONLY_IF_VALID_FIT_OOF_A1_EXISTS" if s4_ap < .59 else
                "PROMISING_FUSION_REQUIRES_VALID_FIT_OOF_A1")
    write_json(a.output / "DRST_FEASIBILITY_GATES.json", {"s4_ap": s4_ap, "a1_ap": a1_ap,
               "delta_ap": ci["mean"], "delta_ap_ci": [ci["ci_low"], ci["ci_high"]],
               "positive_folds": sum(row["delta_ap"] > 0 for row in fold_rows),
               "decision": decision, "A1_retrained": False, "S1_run_in_this_round": False,
               "new_target_outcomes_used_for_training_or_selection": False, "outer_test_accessed": False})
    write_json(a.output / "SOURCE_REPRODUCTION.json", {"exact_A1_fixed_query_AP": a1_ap,
               "expected": 0.5767434626151353, "pass": True,
               "A1_selection": "historical VLOO", "S4_selection": "FIT-only",
               "historical_A1_target_outcomes_previously_viewed": True})
    write_json(a.output / "LABEL_USAGE_AUDIT.json", {"spectral_score_freeze_manifest_verified_before_label_read": True,
               "legacy_loader_materialized_target_label_tensors_during_scoring": True,
               "target_labels_used_for_training_or_checkpoint_selection": False,
               "target_labels_used_for_evaluation_after_freeze": True,
               "historical_A1_target_outcomes_previously_viewed": True,
               "outer_test_accessed": False})
    report = (f"# DRST-PR S4-only feasibility (exploratory)\n\n"
              f"Exact historical A1 replay: AP {a1_ap:.6f} (pass). No A1 retraining.\n\n"
              f"S4 fixed lr=3e-4, wd=1e-3: AP {s4_ap:.6f}; Δ vs A1 {ci['mean']:+.6f} "
              f"(95% patient-ID bootstrap [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}]); "
              f"positive folds {sum(row['delta_ap'] > 0 for row in fold_rows)}/5.\n\n"
              f"Decision: `{decision}`. Error complementarity: A1-wrong/S4-right top1 "
              f"{diagnostic['s4_top1_rescues']}, reverse {diagnostic['s4_top1_destructions']}.\n\n"
              "No S1/S2/S3 attribution, A1 replicate, ensemble control, or unique-complementarity claim. "
              "A1 used historical VLOO selection while S4 used FIT-only selection. Historical A1 target "
              "outcomes had already been viewed; this is not sealed confirmation. No outer test was accessed.\n")
    (a.output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(f"S4_FEASIBILITY_EVALUATION_COMPLETE AP={s4_ap:.6f} decision={decision}", flush=True)


if __name__ == "__main__":
    main()
