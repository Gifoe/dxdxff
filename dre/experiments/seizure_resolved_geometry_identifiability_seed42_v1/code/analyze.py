"""Fit-only geometry and label-free seizure contexts for fixed 65 VLOO targets."""
from __future__ import annotations

import argparse
import hashlib
import math
import pickle
from itertools import combinations

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import CONTEXTS, LOCK_SHA, RUNTIME, preflight, read_csv, write_json


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n <= 1e-12:
        raise RuntimeError("Undefined direction norm")
    return x / n


def cos(a, b):
    return float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))


def oracle(x, y):
    y = np.asarray(y, dtype=np.int8)
    k = min(5, int(np.sum(y == 1)), int(np.sum(y == 0)))
    if k < 2:
        raise RuntimeError("Oracle needs at least two EZ and NEZ channels")
    directions = []
    for train, _heldout in StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x, y):
        model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        model.fit(x[train], y[train])
        if list(model.classes_) != [0, 1]:
            raise RuntimeError("Oracle EZ-positive semantics changed")
        directions.append(unit(model.coef_[0]))
    w = unit(np.mean(directions, axis=0))
    stability = float(np.mean([cos(directions[a], directions[b]) for a, b in combinations(range(k), 2)]))
    return {"w": w, "stability": stability, "k": k}


def rank_metrics(x, y, w=None, score=None):
    s = np.asarray(x @ w if score is None else score, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int8)
    if len(s) != len(y) or not np.isfinite(s).all() or set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Invalid score/label alignment")
    order = np.argsort(s)[::-1]
    ez_rank = np.flatnonzero(y[order] == 1)
    discounts = np.log2(np.arange(2, len(y) + 2))
    ndcg = float(np.sum(y[order] / discounts) / np.sum(np.sort(y)[::-1] / discounts))
    return {"ap": float(average_precision_score(y, s)), "auc": float(roc_auc_score(y, s)),
            "mrr": float(1 / (ez_rank[0] + 1)), "top1": float(y[np.argmax(s)] == 1), "ndcg": ndcg}


def static_context(row, pca):
    z = pca.transform(row["R4"])
    return np.r_[z.mean(axis=0), z.std(axis=0, ddof=0)]


def seizure_arrays(row, pca):
    e, present = row["E"], row["present"]
    if e.ndim != 3 or present.shape != e.shape[:2] or not present.any(axis=1).all():
        raise RuntimeError("Seizure tensor/mask shape mismatch")
    return [pca.transform(e[s, present[s]]) for s in range(len(e))]


def dist_context(arrays):
    signatures = []
    for z in arrays:
        signatures.append(np.r_[z.mean(axis=0), z.std(axis=0, ddof=0),
                                np.quantile(z, .25, axis=0), np.quantile(z, .50, axis=0),
                                np.quantile(z, .75, axis=0)])
    sig = np.stack(signatures)
    result = np.r_[sig.mean(axis=0), sig.std(axis=0, ddof=0), math.log1p(len(arrays)), float(len(arrays) >= 2)]
    if result.shape != (82,):
        raise RuntimeError("SR_DIST dimension changed")
    return result


def persist_context(row, arrays):
    present = row["present"]
    n_seizures, n_channels = present.shape
    rank = np.full((n_seizures, n_channels, 8), np.nan, dtype=np.float64)
    for s, z in enumerate(arrays):
        for k in range(8):
            rank[s, present[s], k] = rankdata(z[:, k], method="average") / (len(z) + 1)
    with np.errstate(invalid="ignore"):
        count = np.sum(np.isfinite(rank), axis=0)
        mean = np.divide(np.nansum(rank, axis=0), count, out=np.full((n_channels, 8), np.nan), where=count > 0)
        centered = np.where(np.isfinite(rank), rank - mean[None], 0.0)
        sd = np.sqrt(np.divide(np.sum(centered**2, axis=0), count,
                               out=np.full((n_channels, 8), np.nan), where=count > 0))
    parts = []
    for k in range(8):
        valid = count[:, k] > 0
        if not np.any(valid):
            raise RuntimeError("PCA component absent in all channels")
        parts.extend(np.quantile(mean[valid, k], [.1, .25, .5, .75, .9]))
        parts.extend(np.quantile(sd[valid, k], [.1, .25, .5, .75, .9]))
    pair_values = [[] for _ in range(8)]
    if n_seizures >= 2:
        for s, t in combinations(range(n_seizures), 2):
            common = present[s] & present[t]
            for k in range(8):
                if int(common.sum()) < 3:
                    rho = 0.0
                else:
                    rho = spearmanr(rank[s, common, k], rank[t, common, k]).statistic
                    if not np.isfinite(rho):
                        rho = 0.0
                pair_values[k].append(float(rho))
    for values in pair_values:
        x = np.asarray(values, dtype=np.float64)
        parts.extend([float(x.mean()), float(x.std(ddof=0)),
                      *np.quantile(x, [.1, .5, .9])]) if len(x) else parts.extend([0.0] * 5)
    result = np.r_[parts, float(n_seizures >= 2), math.log1p(n_seizures)]
    if result.shape != (122,):
        raise RuntimeError("SR_PERSIST dimension changed")
    return result


def shuffled_arrays(arrays, subject_id):
    counts = [len(a) for a in arrays]
    pool = np.concatenate(arrays)
    seed = 42 + int.from_bytes(hashlib.sha256(subject_id.encode("utf-8")).digest()[:8], "little")
    perm = np.random.default_rng(seed).permutation(len(pool))
    return list(np.split(pool[perm], np.cumsum(counts)[:-1]))


def contexts_for_row(row, static_pca, seizure_pca, subject_id):
    arrays = seizure_arrays(row, seizure_pca)
    dist = dist_context(arrays)
    persist = persist_context(row, arrays)
    shuffle_dist = dist_context(shuffled_arrays(arrays, subject_id))
    return ({"STATIC_R4_MARG": static_context(row, static_pca), "SR_DIST": dist,
             "SR_PERSIST": persist, "SR_COMBINED": np.r_[dist, persist]},
            {"SR_DIST": shuffle_dist, "SR_COMBINED": np.r_[shuffle_dist, persist]})


def build_base(payload):
    fit, val = payload["fit"], payload["val"]
    fit_ids = sorted(fit)
    ids = fit_ids + sorted(val)
    fit_oracles = {sid: oracle(fit[sid]["R4"], fit[sid]["y"]) for sid in fit_ids}
    directions = np.stack([fit_oracles[sid]["w"] for sid in fit_ids])
    mean = unit(directions.mean(axis=0))
    centered = directions - directions.mean(axis=0)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    evr = np.cumsum(singular**2 / np.sum(singular**2))
    r80, r90 = int(np.searchsorted(evr, .80) + 1), int(np.searchsorted(evr, .90) + 1)
    basis = vt[:r80].T
    x = np.concatenate([fit[sid]["R4"] for sid in fit_ids])
    y = np.concatenate([fit[sid]["y"] for sid in fit_ids])
    weights = np.concatenate([np.where(fit[sid]["y"] == 1,
                                       .5 / np.sum(fit[sid]["y"] == 1),
                                       .5 / np.sum(fit[sid]["y"] == 0)) for sid in fit_ids])
    shared = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    shared.fit(x, y, sample_weight=weights)
    wshared = unit(shared.coef_[0])
    static_pca = PCA(n_components=8, svd_solver="full").fit(x)
    seizure_fit = np.concatenate([fit[sid]["E"][fit[sid]["present"]] for sid in fit_ids])
    seizure_pca = PCA(n_components=8, svd_solver="full").fit(seizure_fit)
    raw_context, shuffled = {}, {}
    for sid in ids:
        raw_context[sid], shuffled[sid] = contexts_for_row((fit if sid in fit else val)[sid], static_pca, seizure_pca, sid)
    models, associations, transformed, shuffled_transformed = {}, {}, {}, {}
    for context in CONTEXTS:
        fit_raw = np.stack([raw_context[sid][context] for sid in fit_ids])
        scaler = StandardScaler().fit(fit_raw)
        standardized = scaler.transform(fit_raw)
        ndim = min(16, fit_raw.shape[1], len(fit_ids) - 1)
        pca = PCA(n_components=ndim, svd_solver="full").fit(standardized)
        fit_z = pca.transform(standardized)
        transformed[context] = {sid: pca.transform(scaler.transform(raw_context[sid][context][None]))[0] for sid in ids}
        if context in ("SR_DIST", "SR_COMBINED"):
            shuffled_transformed[context] = {sid: pca.transform(scaler.transform(shuffled[sid][context][None]))[0] for sid in val}
        ridge_full = Ridge(alpha=10.0, fit_intercept=True).fit(fit_z, directions)
        coefficients = (directions - mean) @ basis
        ridge_r80 = Ridge(alpha=10.0, fit_intercept=True).fit(fit_z, coefficients)
        pair_i, pair_j = np.triu_indices(len(fit_ids), 1)
        distance = np.linalg.norm(fit_z[pair_i] - fit_z[pair_j], axis=1)
        direction_cos = np.sum(directions[pair_i] * directions[pair_j], axis=1)
        rho = float(spearmanr(distance, direction_cos).statistic)
        models[context] = {"full": ridge_full, "r80": ridge_r80, "ndim": ndim}
        associations[context] = {"rho": rho, "pairs": len(distance), "pair_i": pair_i,
                                 "pair_j": pair_j, "distance": distance.astype(np.float32),
                                 "direction_cos": direction_cos.astype(np.float32)}
    return {"lock_sha": LOCK_SHA, "fold": payload["fold"], "epoch": payload["epoch"],
            "fit_ids": fit_ids, "fit_oracles": fit_oracles, "mean": mean, "wshared": wshared,
            "basis": basis, "evr": evr, "r80": r80, "r90": r90,
            "models": models, "associations": associations,
            "context": transformed, "shuffled_context": shuffled_transformed,
            "raw_dimensions": {c: len(raw_context[fit_ids[0]][c]) for c in CONTEXTS}}


def scored(method, context, x, y, w, oracle_w, mean, shared):
    result = rank_metrics(x, y, w=w)
    c = cos(w, oracle_w)
    result.update({"predictor": method, "context": context, "cosine": c,
                   "angle_deg": math.degrees(math.acos(c)),
                   "delta_cos_D0": c - cos(mean, oracle_w), "delta_cos_D1": c - cos(shared, oracle_w)})
    return result


def analyze_target(payload, base, target_id):
    row = payload["val"][target_id]
    x = row["R4"]
    mean, shared, basis = base["mean"], base["wshared"], base["basis"]
    validation_order = sorted(payload["val"])
    donor = validation_order[(validation_order.index(target_id) + 1) % len(validation_order)]
    # All predictor directions and channel scores are frozen BEFORE target y is accessed.
    dirs = {("D0", ""): mean, ("D1", ""): shared}
    wrong_dirs, shuffle_dirs = {}, {}
    for context in CONTEXTS:
        z = base["context"][context][target_id][None]
        model = base["models"][context]
        dirs[("D2", context)] = unit(model["full"].predict(z)[0])
        dirs[("D3", context)] = unit(mean + basis @ model["r80"].predict(z)[0])
        wrong_z = base["context"][context][donor][None]
        wrong_dirs[("D2", context)] = unit(model["full"].predict(wrong_z)[0])
        wrong_dirs[("D3", context)] = unit(mean + basis @ model["r80"].predict(wrong_z)[0])
        if context in ("SR_DIST", "SR_COMBINED"):
            shuf_z = base["shuffled_context"][context][target_id][None]
            shuffle_dirs[("D2", context)] = unit(model["full"].predict(shuf_z)[0])
            shuffle_dirs[("D3", context)] = unit(mean + basis @ model["r80"].predict(shuf_z)[0])
    frozen_scores = {key: x @ w for key, w in dirs.items()}
    wrong_scores = {key: x @ w for key, w in wrong_dirs.items()}
    shuffle_scores = {key: x @ w for key, w in shuffle_dirs.items()}
    # Only now read validation labels to form the NONDEPLOYABLE target oracle and metrics.
    y = row["y"]
    target_oracle = oracle(x, y)
    ow = target_oracle["w"]
    r80_w = unit(mean + basis @ (basis.T @ (ow - mean)))
    predictions = [scored(key[0], key[1], x, y, w, ow, mean, shared) for key, w in dirs.items()]
    wrong = [{"predictor": key[0], "context": key[1], "correct_cosine": cos(dirs[key], ow),
              "wrong_cosine": cos(w, ow), "correct_ap": rank_metrics(x, y, score=frozen_scores[key])["ap"],
              "wrong_ap": rank_metrics(x, y, score=wrong_scores[key])["ap"]} for key, w in wrong_dirs.items()]
    shuffled = [{"predictor": key[0], "context": key[1], "true_cosine": cos(dirs[key], ow),
                 "shuffled_cosine": cos(w, ow), "true_ap": rank_metrics(x, y, score=frozen_scores[key])["ap"],
                 "shuffled_ap": rank_metrics(x, y, score=shuffle_scores[key])["ap"]} for key, w in shuffle_dirs.items()]
    return {"lock_sha": LOCK_SHA, "fold": payload["fold"], "epoch": payload["epoch"], "subject_id": target_id,
            "n_seizures": row["n_seizures"], "target_labels_accessed_only_after_all_predictor_scores_fixed": True,
            "oracle_stability": target_oracle["stability"], "oracle_inner_k": target_oracle["k"],
            "oracle": rank_metrics(x, y, w=ow), "r80_oracle": {**rank_metrics(x, y, w=r80_w), "cosine": cos(r80_w, ow)},
            "frozen_a1": rank_metrics(x, y, score=row["source_ez"]),
            "r80": base["r80"], "r90": base["r90"],
            "evr": {str(k): float(base["evr"][k-1]) for k in (1,2,4,8,12,16,24,32)},
            "association": {c: {"rho": base["associations"][c]["rho"], "pairs": base["associations"][c]["pairs"]} for c in CONTEXTS},
            "raw_dimensions": base["raw_dimensions"],
            "context_pca_dimensions": {c: base["models"][c]["ndim"] for c in CONTEXTS},
            "predictions": predictions, "wrong": wrong, "shuffled": shuffled}


def run_fold(fold):
    folder = RUNTIME / "private" / f"fold_{fold}"
    selected = read_csv(folder / "A1_VLOO_PRIVATE.csv")
    if len(selected) != 13:
        raise RuntimeError("Incomplete A1 VLOO")
    for chosen in selected:
        sid, epoch = chosen["subject_id"], int(chosen["selected_epoch"])
        target = folder / "cells" / f"cell_{sid.replace(':', '_')}.pkl"
        if target.exists():
            with target.open("rb") as f:
                cell = pickle.load(f)
            if cell["lock_sha"] != LOCK_SHA or cell["epoch"] != epoch or cell["subject_id"] != sid:
                raise RuntimeError("Cell cache provenance mismatch")
            continue
        with (folder / f"epoch_{epoch:02d}_representations.pkl").open("rb") as f:
            payload = pickle.load(f)
        if payload["fold"] != fold or payload["epoch"] != epoch or sid not in payload["val"]:
            raise RuntimeError("Target/source coordinate mismatch")
        base_path = folder / f"epoch_{epoch:02d}_base.pkl"
        if base_path.exists():
            with base_path.open("rb") as f:
                base = pickle.load(f)
            if base["lock_sha"] != LOCK_SHA or base["fold"] != fold or base["epoch"] != epoch:
                raise RuntimeError("FIT model cache provenance mismatch")
        else:
            base = build_base(payload)
            tmp = base_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(base, f, protocol=5)
            tmp.replace(base_path)
        cell = analyze_target(payload, base, sid)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(cell, f, protocol=5)
        tmp.replace(target)
        print(f"[CELL] fold={fold} complete={len(list(target.parent.glob('*.pkl')))}/13 epoch={epoch}", flush=True)
    write_json(folder / "analysis_status.json", {"fold": fold, "cells": 13, "lock_sha": LOCK_SHA,
               "outer_predictions_or_metrics": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1,6))
    opt = parser.parse_args()
    preflight()
    for fold in range(1,6):
        if opt.fold is None or opt.fold == fold:
            run_fold(fold)


if __name__ == "__main__":
    main()
