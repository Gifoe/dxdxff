"""Hash-freeze score grids, then run exact matched A1 VLOO/fixed-query evaluation.

Patient and channel tables remain under the private runtime. Public files are
aggregate-only. Nothing in this script opens an outer-test prediction file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

VARIANTS = ("M1_RAWTINY_NOPR", "M2_RAWTINY_PR", "M3_HYBRID_PR")
ALL = ("M0_A1",) + VARIANTS
PAIRS = (("M1_RAWTINY_NOPR", "M0_A1"), ("M2_RAWTINY_PR", "M1_RAWTINY_NOPR"),
         ("M2_RAWTINY_PR", "M0_A1"), ("M3_HYBRID_PR", "M2_RAWTINY_PR"),
         ("M3_HYBRID_PR", "M0_A1"))
LOCK_SHA = "028a951709f095459c39986f3823110fa9826078f24290d6afe9017c2ed621d2"
METRICS = ("ap", "auc", "mrr", "top1", "ndcg", "macro_f1", "ez_f1", "ba", "predicted_ez_fraction")


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def avg(values) -> float | None:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    return float(data.mean()) if len(data) else None


def dependencies(root: Path):
    project = root.parent
    sys.path.insert(0, str(project / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
    sys.path.insert(0, str(project / "active_fewshot_patient_calibration_seed42_v1" / "code"))
    import common as afc
    import run as afr
    from development_metrics import epoch_grid, finalize_fold
    return afc, afr, epoch_grid, finalize_fold


def score_path(runtime: Path, variant: str, fold: int, epoch: int) -> Path:
    return runtime / variant / f"fold_{fold}" / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"


def freeze(runtime: Path, output: Path, afc) -> None:
    files = {}
    for variant in VARIANTS:
        for fold in range(1, 6):
            folder = runtime / variant / f"fold_{fold}"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if summary["variant"] != variant or summary["fold"] != fold or len(summary["epoch_rows"]) != 30:
                raise RuntimeError(f"Incomplete training grid: {variant} fold={fold}")
            for epoch, row in enumerate(summary["epoch_rows"], 1):
                path = score_path(runtime, variant, fold, epoch)
                digest = sha(path)
                if row["score_sha256"] != digest:
                    raise RuntimeError(f"Score snapshot mismatch: {path}")
                files[str(path.relative_to(runtime))] = digest
    if len(files) != 450:
        raise RuntimeError("Expected exactly 450 score-only files")
    init = json.loads((runtime / "fold_1_raw_initializations.json").read_text(encoding="utf-8"))
    for fold in range(1, 6):
        row = json.loads((runtime / f"fold_{fold}_raw_initializations.json").read_text(encoding="utf-8"))
        if row[VARIANTS[0]] != row[VARIANTS[1]]:
            raise RuntimeError(f"M1/M2 raw initialization mismatch in fold {fold}")
    private = {"lock_sha256": LOCK_SHA, "n_score_files": 450, "files": files,
               "patient_labels_indexed_during_this_freeze": False}
    write_json(runtime / "SCORE_FREEZE_BEFORE_TARGET_LABELS.json", private)
    public = {"pass": True, "lock_sha256": LOCK_SHA, "n_variants": 3, "folds": 5,
              "epochs": 30, "n_score_files": 450,
              "private_manifest_sha256": sha(runtime / "SCORE_FREEZE_BEFORE_TARGET_LABELS.json"),
              "target_labels_indexed_during_freeze": False,
              "legacy_loader_materialized_validation_labels_during_training": True}
    write_json(output / "SCORE_FREEZE_AUDIT.json", public)
    print("SCORE_FREEZE_PASS 450/450", flush=True)


@lru_cache(maxsize=40)
def snap(runtime_str: str, variant: str, fold: int, epoch: int) -> dict:
    with score_path(Path(runtime_str), variant, fold, epoch).open("rb") as h:
        return pickle.load(h)


def source_rows(afc, fold: int) -> list[dict]:
    rows = afc.read_csv(afc.PRIOR_RUNTIME / "private" / f"fold_{fold}" / "A1_VLOO_PRIVATE.csv")
    if len(rows) != 13:
        raise RuntimeError("Source A1 VLOO fold must have 13 rows")
    return rows


@lru_cache(maxsize=80)
def source_payload(runtime: str, fold: int, epoch: int):
    path = Path(runtime) / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}_representations.pkl"
    with path.open("rb") as h:
        value = pickle.load(h)
    if value["fold"] != fold or value["epoch"] != epoch or value["r4_dim"] != 64 or value["r4_error"] > 1e-6:
        raise RuntimeError("Exact A1 representation provenance mismatch")
    return value


def labels(afc, fold: int, sid: str) -> np.ndarray:
    selected = next(r for r in source_rows(afc, fold) if r["subject_id"] == sid)
    epoch = int(selected["selected_epoch"])
    return np.asarray(source_payload(str(afc.PRIOR_RUNTIME), fold, epoch)["val"][sid]["y"], dtype=np.int8)


def evaluate(runtime: Path, output: Path, afc, afr, epoch_grid, finalize_fold) -> None:
    freeze_path = runtime / "SCORE_FREEZE_BEFORE_TARGET_LABELS.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["lock_sha256"] != LOCK_SHA or frozen["n_score_files"] != 450:
        raise RuntimeError("Complete pre-label score freeze required")
    rows = []
    for fold in range(1, 6):
        baseline = source_rows(afc, fold)
        for src in baseline:
            sid = src["subject_id"]
            epoch = int(src["selected_epoch"])
            tau = float(src["selected_threshold"])
            item = source_payload(str(afc.PRIOR_RUNTIME), fold, epoch)["val"][sid]
            y = np.asarray(item["y"], dtype=np.int8)
            margin = afr.margin_for(item, tau)
            afr.verify_b0(item, tau, margin)
            for rep in range(20):
                _, query = afc.split_indices(len(y), 42, fold, sid, rep)
                rows.append({"fold": fold, "sid": sid, "rep": rep, "variant": "M0_A1",
                             **afc.query_metrics(y[query], margin[query])})
        for variant in VARIANTS:
            payloads = []
            for epoch in range(1, 31):
                path = score_path(runtime, variant, fold, epoch)
                if sha(path) != frozen["files"][str(path.relative_to(runtime))]:
                    raise RuntimeError("Frozen score file changed after label unlock")
                snapshot = snap(str(runtime), variant, fold, epoch)
                records = []
                for sid in sorted(snapshot):
                    row = snapshot[sid]
                    y = labels(afc, fold, sid)
                    if len(y) != row["n_channels"]:
                        raise RuntimeError("Source/raw channel order mismatch")
                    records.append({"subject_id": sid, "channel_mask": np.ones(len(y), bool),
                                    "labels_ez": y, "labels_nez": 1-y, "labels": 1-y,
                                    "score_ez": row["score_ez"], "score_nez": row["score_nez"]})
                payloads.append(epoch_grid(records, epoch))
            selection_path = runtime / "private" / variant / f"fold_{fold}_VLOO_PATIENT_PRIVATE.csv"
            public, full = finalize_fold(payloads, variant, fold, selection_path)
            write_json(runtime / "private" / variant / f"fold_{fold}_VLOO_SUMMARY.json",
                       {"public": public, "full": full})
            selection = afc.read_csv(selection_path)
            if len(selection) != 13:
                raise RuntimeError("VLOO selected wrong number of target cells")
            for selected in selection:
                sid = selected["subject_id"]
                epoch = int(selected["selected_epoch"])
                tau = float(selected["selected_threshold"])
                item = snap(str(runtime), variant, fold, epoch)[sid]
                y = labels(afc, fold, sid)
                margin = math.log(tau/(1-tau)) - np.asarray(item["logit_nez"], dtype=np.float64)
                if len(margin) != len(y) or not np.isfinite(margin).all():
                    raise RuntimeError("Invalid target score")
                for rep in range(20):
                    _, query = afc.split_indices(len(y), 42, fold, sid, rep)
                    rows.append({"fold": fold, "sid": sid, "rep": rep, "variant": variant,
                                 **afc.query_metrics(y[query], margin[query])})
            print(f"[VLOO] {variant} fold={fold} cells=13", flush=True)
    if len(rows) != 5200 or len({r["sid"] for r in rows}) != 47:
        raise RuntimeError("Expected 4 models x 65 cells x 20 reps and 47 unique patients")
    private = runtime / "private" / "all_query_rows.pkl"
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open("wb") as h:
        pickle.dump(rows, h, protocol=5)
    finalize(output, rows)


def center_of(sid: str) -> str:
    token = sid.split(":", 1)[0].lower()
    if token == "hup": return "HUP"
    if token in {"lzu", "fudan"}: return token.upper()
    return "multi-site"


def bootstrap(deltas: np.ndarray, part: list[dict], ids: list[str], counts: np.ndarray) -> dict:
    index = {sid: i for i, sid in enumerate(ids)}
    numer = np.zeros(len(ids))
    denom = np.zeros(len(ids))
    for value, row in zip(deltas, part):
        if np.isfinite(value):
            i = index[row["sid"]]
            numer[i] += value
            denom[i] += 1
    draws = np.divide(counts @ numer, counts @ denom,
                      out=np.full(len(counts), np.nan), where=(counts @ denom) > 0)
    good = draws[np.isfinite(draws)]
    if len(good) < 0.99 * len(draws):
        raise RuntimeError("Degenerate patient-cluster bootstrap")
    return {"mean": float(numer.sum()/denom.sum()), "ci_low": float(np.quantile(good, .025)),
            "ci_high": float(np.quantile(good, .975)), "n_estimable": int(denom.sum())}


def finalize(output: Path, rows: list[dict]) -> None:
    exact = {(r["fold"], r["sid"], r["rep"], r["variant"]): r for r in rows}
    if len(exact) != len(rows):
        raise RuntimeError("Duplicate target query cells")
    ids = sorted({r["sid"] for r in rows})
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))
    counts = np.zeros((10000, len(ids)), dtype=np.int16)
    np.add.at(counts, (np.arange(10000)[:, None], draws), 1)
    counts = counts.astype(np.float32)
    matrix, fold_rows, center_rows, pair_rows = [], [], [], []
    for variant in ALL:
        part = [r for r in rows if r["variant"] == variant]
        if len(part) != 1300:
            raise RuntimeError("Incomplete fixed query repetitions")
        base = [exact[(r["fold"], r["sid"], r["rep"], "M0_A1")] for r in part]
        difference = np.asarray([r["ap"]-b["ap"] for r, b in zip(part, base)], dtype=float)
        delta = bootstrap(difference, part, ids, counts)
        matrix.append({"model": variant, "n_cells": 65, "n_unique_patient_ids": 47,
                       "n_repetitions": 1300, "n_estimable_ap": sum(np.isfinite(r["ap"]) for r in part),
                       **{metric: avg(r[metric] for r in part) for metric in METRICS},
                       "delta_ap_vs_M0": delta["mean"],
                       "delta_ap_ci_low": delta["ci_low"], "delta_ap_ci_high": delta["ci_high"]})
        for fold in range(1, 6):
            sub = [r for r in part if r["fold"] == fold]
            fold_rows.append({"model": variant, "fold": fold, "n_cells": 13,
                              "ap": avg(r["ap"] for r in sub),
                              "delta_ap_vs_M0": avg(r["ap"]-exact[(fold, r["sid"], r["rep"], "M0_A1")]["ap"] for r in sub),
                              "mrr": avg(r["mrr"] for r in sub), "top1": avg(r["top1"] for r in sub)})
        for center in ("HUP", "multi-site", "LZU", "FUDAN"):
            sub = [r for r in part if center_of(r["sid"]) == center]
            if sub:
                center_rows.append({"model": variant, "center": center, "n_unique_patients": len({r["sid"] for r in sub}),
                                    "ap": avg(r["ap"] for r in sub),
                                    "delta_ap_vs_M0": avg(r["ap"]-exact[(r["fold"], r["sid"], r["rep"], "M0_A1")]["ap"] for r in sub)})
    for a, b in PAIRS:
        part = [r for r in rows if r["variant"] == a]
        delta = np.asarray([r["ap"]-exact[(r["fold"], r["sid"], r["rep"], b)]["ap"] for r in part], dtype=float)
        stat = bootstrap(delta, part, ids, counts)
        signs = [avg(r["ap"]-exact[(fold, r["sid"], r["rep"], b)]["ap"]
                     for r in part if r["fold"] == fold) for fold in range(1, 6)]
        pair_rows.append({"model_a": a, "model_b": b, **stat, "positive_folds": sum(v is not None and v > 0 for v in signs),
                          "delta_mrr": avg(r["mrr"]-exact[(r["fold"], r["sid"], r["rep"], b)]["mrr"] for r in part),
                          "delta_top1": avg(r["top1"]-exact[(r["fold"], r["sid"], r["rep"], b)]["top1"] for r in part),
                          "n_patient_clusters": 47, "bootstrap_resamples": 10000, "seed": 42})
    base = next(r for r in matrix if r["model"] == "M0_A1")
    if abs(base["ap"] - 0.5767434626151353) > 1e-9:
        raise RuntimeError(f"Exact A1 fixed-query replay failed: {base['ap']}")
    write_csv(output / "PRIMARY_MODEL_MATRIX.csv", matrix)
    write_csv(output / "PATIENT_CLUSTER_BOOTSTRAP.csv", pair_rows)
    write_csv(output / "FOLD_CONSISTENCY.csv", fold_rows)
    write_csv(output / "CENTER_STRATIFIED_METRICS.csv", center_rows)
    by_pair = {(r["model_a"], r["model_b"]): r for r in pair_rows}
    m = {r["model"]: r for r in matrix}
    p10 = by_pair[("M1_RAWTINY_NOPR", "M0_A1")]
    p21 = by_pair[("M2_RAWTINY_PR", "M1_RAWTINY_NOPR")]
    p32 = by_pair[("M3_HYBRID_PR", "M2_RAWTINY_PR")]
    best = max(matrix, key=lambda r: r["ap"])
    pbest = by_pair.get((best["model"], "M0_A1"))
    gates = {
        "RAW_INPUT_INFORMATION_SUPPORTED": p10["mean"] > 0 and p10["ci_low"] > 0 and p10["positive_folds"] >= 4,
        "RAW_PATIENT_RELATIVE_SUPPORTED": p21["mean"] >= .010 and p21["ci_low"] > 0 and p21["positive_folds"] >= 4 and p21["delta_mrr"] >= 0 and p21["delta_top1"] >= 0,
        "RAW_ENGINEERED_COMPLEMENTARITY_SUPPORTED": p32["mean"] > 0 and p32["ci_low"] > 0 and p32["positive_folds"] >= 4,
        "RAWTINY_ZERO_SHOT_REACHES_CURRENT_B8": best["ap"] >= .5996322681940147 and (pbest is not None and pbest["ci_low"] > 0),
        "RAWTINY_ZERO_SHOT_REACHES_BEST_B8": best["ap"] >= .6052912572363056 and (pbest is not None and pbest["ci_low"] > 0 and pbest["positive_folds"] >= 4 and pbest["delta_mrr"] >= 0 and pbest["delta_top1"] >= 0),
        "best_model": best["model"], "best_ap": best["ap"],
        "outer_test_accessed": False,
    }
    write_json(output / "RAWTINY_GATES.json", gates)
    print(f"EVALUATION_COMPLETE best={best['model']} ap={best['ap']:.6f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    args = p.parse_args()
    if sha(args.lock) != LOCK_SHA:
        raise RuntimeError("Protocol lock changed")
    root = args.lock.parent
    afc, afr, epoch_grid, finalize_fold = dependencies(root)
    if not os.environ.get("SRGI_RUNTIME") or not Path(os.environ["SRGI_RUNTIME"]).is_absolute():
        raise RuntimeError("Exact A1 private representation runtime missing")
    if args.stage == "freeze":
        freeze(args.runtime, args.output, afc)
    else:
        evaluate(args.runtime, args.output, afc, afr, epoch_grid, finalize_fold)


if __name__ == "__main__":
    main()
