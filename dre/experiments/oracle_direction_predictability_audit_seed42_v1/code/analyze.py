"""Fixed, development-only oracle-direction predictability audit; private cell records."""
from __future__ import annotations

import argparse
import math
import pickle
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import LOCK_SHA, RUNTIME, preflight, read_csv, write_json

CONTEXTS = ("R3_MARG", "R3_COV", "R4_MARG", "R4_COV")


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Undefined direction norm")
    return x / norm


def cosine(a, b):
    return float(np.clip(np.dot(unit(a), unit(b)), -1, 1))


def oracle(x, y):
    y = np.asarray(y, dtype=np.int8)
    n_ez, n_nez = int(np.sum(y == 1)), int(np.sum(y == 0))
    k = min(5, n_ez, n_nez)
    if k < 2:
        raise RuntimeError("Oracle requires at least two channels in both classes")
    directions = []
    for train, _unused in StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x, y):
        fit = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        fit.fit(x[train], y[train])
        if list(fit.classes_) != [0, 1]:
            raise RuntimeError("Oracle positive class is not EZ=1")
        directions.append(unit(fit.coef_[0]))
    direction = unit(np.mean(directions, axis=0))
    stability = float(np.mean([cosine(directions[i], directions[j]) for i, j in combinations(range(k), 2)]))
    return {"w": direction, "stability": stability, "inner_k": k}


def rank_metrics(x, y, w=None, score=None):
    s = np.asarray(x @ w if score is None else score, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int8)
    if len(s) != len(y) or not np.isfinite(s).all() or set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Invalid ranking input")
    order = np.argsort(s)[::-1]
    rank_ez = np.flatnonzero(y[order] == 1)
    gain = y[order] / np.log2(np.arange(2, len(y) + 2))
    ideal = np.sort(y)[::-1] / np.log2(np.arange(2, len(y) + 2))
    return {"ap": float(average_precision_score(y, s)), "auc": float(roc_auc_score(y, s)),
            "mrr": float(1 / (rank_ez[0] + 1)), "top1": float(y[np.argmax(s)] == 1),
            "ndcg": float(gain.sum() / ideal.sum())}


def patient_context(x, kind):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    if kind == "MARG":
        return np.r_[mean, sd]
    cov = (x - mean).T @ (x - mean) / len(x)
    return np.r_[mean, sd, cov[np.triu_indices(x.shape[1])]]


def make_epoch_base(payload):
    fit = payload["fit"]
    val = payload["val"]
    fit_ids = sorted(fit)
    all_ids = fit_ids + sorted(val)
    fit_oracles = {sid: oracle(fit[sid]["R4"], fit[sid]["y"]) for sid in fit_ids}
    directions = np.stack([fit_oracles[sid]["w"] for sid in fit_ids])
    mu = directions.mean(axis=0)
    wbar = unit(mu)
    centered = directions - mu
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    evr = singular**2 / np.square(singular).sum()
    basis = vt[:4].T
    # The shared FIT direction gives each patient equal total mass and each class half its mass.
    x = np.concatenate([fit[sid]["R4"] for sid in fit_ids])
    y = np.concatenate([fit[sid]["y"] for sid in fit_ids])
    weight = np.concatenate([np.where(fit[sid]["y"] == 1,
                                       0.5 / np.sum(fit[sid]["y"] == 1),
                                       0.5 / np.sum(fit[sid]["y"] == 0))
                             for sid in fit_ids])
    shared = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    shared.fit(x, y, sample_weight=weight)
    wshared = unit(shared.coef_[0])
    contexts = {}
    models = {}
    associations = {}
    for source in ("R3", "R4"):
        all_fit_channels = np.concatenate([fit[sid][source] for sid in fit_ids])
        pca = PCA(n_components=8, svd_solver="full")
        pca.fit(all_fit_channels)
        for kind in ("MARG", "COV"):
            name = source + "_" + kind
            features = {sid: patient_context(pca.transform((fit if sid in fit else val)[sid][source]), kind)
                        for sid in all_ids}
            fit_features = np.stack([features[sid] for sid in fit_ids])
            scaler = StandardScaler().fit(fit_features)
            standardized = scaler.transform(fit_features)
            direct = Ridge(alpha=10.0, fit_intercept=True).fit(standardized, directions)
            coeff = (directions - mu) @ basis
            lowrank = Ridge(alpha=10.0, fit_intercept=True).fit(standardized, coeff)
            i, j = np.triu_indices(len(fit_ids), k=1)
            distance = np.linalg.norm(standardized[i] - standardized[j], axis=1)
            direction_cos = np.sum(directions[i] * directions[j], axis=1)
            rho = float(spearmanr(distance, direction_cos).statistic)
            contexts[name] = features
            models[name] = {"scaler": scaler, "fit_standardized": standardized,
                            "direct": direct, "lowrank": lowrank}
            associations[name] = {"rho": rho, "n_pairs": len(distance)}
    return {"lock_sha": LOCK_SHA, "fold": payload["fold"], "epoch": payload["epoch"],
            "fit_ids": fit_ids, "fit_oracles": fit_oracles, "mu": mu, "wbar": wbar,
            "basis": basis, "evr": evr, "wshared": wshared,
            "contexts": contexts, "models": models, "associations": associations}


def scored(name, context, x, y, w, target_oracle, wbar, wshared, basis):
    metrics = rank_metrics(x, y, w=w)
    c = cosine(w, target_oracle)
    metrics.update({"predictor": name, "context": context, "cosine": c,
                    "angle_deg": math.degrees(math.acos(c)),
                    "delta_cos_vs_population": c - cosine(wbar, target_oracle),
                    "delta_cos_vs_shared": c - cosine(wshared, target_oracle),
                    "projection_recovery_l2": float(np.linalg.norm(basis.T @ (w - target_oracle)))})
    return metrics


def analyze_target(payload, base, target_id):
    val = payload["val"]
    x = val[target_id]["R4"]
    y = val[target_id]["y"]
    true = oracle(x, y)
    wtrue = true["w"]
    mu, wbar, basis, wshared = base["mu"], base["wbar"], base["basis"], base["wshared"]
    projected = unit(mu + basis @ (basis.T @ (wtrue - mu)))
    preds = [scored("D0", "", x, y, wbar, wtrue, wbar, wshared, basis),
             scored("D1", "", x, y, wshared, wtrue, wbar, wshared, basis)]
    wrong_rows = []
    validation_order = sorted(val)
    donor = validation_order[(validation_order.index(target_id) + 1) % len(validation_order)]
    for context in CONTEXTS:
        features = base["contexts"][context]
        m = base["models"][context]
        c_target = m["scaler"].transform(features[target_id][None])[0]
        c_wrong = m["scaler"].transform(features[donor][None])[0]
        dist = np.linalg.norm(m["fit_standardized"] - c_target, axis=1)
        neighbors = np.argsort(dist, kind="stable")[:3]
        w2 = unit(np.mean([base["fit_oracles"][base["fit_ids"][i]]["w"] for i in neighbors], axis=0))
        w3 = unit(m["direct"].predict(c_target[None])[0])
        w4 = unit(mu + basis @ m["lowrank"].predict(c_target[None])[0])
        for method, w in (("D2", w2), ("D3", w3), ("D4", w4)):
            preds.append(scored(method, context, x, y, w, wtrue, wbar, wshared, basis))
        for method, estimator in (("D3", m["direct"]), ("D4", m["lowrank"])):
            wrong = unit(estimator.predict(c_wrong[None])[0] if method == "D3" else
                         mu + basis @ estimator.predict(c_wrong[None])[0])
            right_row = next(row for row in preds if row["predictor"] == method and row["context"] == context)
            wrong_metric = rank_metrics(x, y, w=wrong)
            wrong_rows.append({"predictor": method, "context": context,
                               "correct_cosine": right_row["cosine"], "wrong_cosine": cosine(wrong, wtrue),
                               "correct_ap": right_row["ap"], "wrong_ap": wrong_metric["ap"]})
    return {"lock_sha": LOCK_SHA, "fold": payload["fold"], "epoch": payload["epoch"],
            "subject_id": target_id, "oracle_stability": true["stability"], "oracle_inner_k": true["inner_k"],
            "oracle": rank_metrics(x, y, w=wtrue),
            "rank4": {**rank_metrics(x, y, w=projected), "cosine": cosine(projected, wtrue)},
            "source_a1": rank_metrics(x, y, score=val[target_id]["source_ez"]),
            "evr": {str(k): float(evr) for k, evr in ((k, base["evr"][:k].sum()) for k in (1, 2, 4, 8))},
            "association": base["associations"], "predictions": preds, "wrong": wrong_rows}


def run_fold(fold):
    folder = RUNTIME / "private" / f"fold_{fold}"
    rows = read_csv(folder / "A1_VLOO_PRIVATE.csv")
    if len(rows) != 13 or len(set(r["subject_id"] for r in rows)) != 13:
        raise RuntimeError("Expected 13 VLOO targets")
    for row in rows:
        target_id = row["subject_id"]
        epoch = int(row["selected_epoch"])
        cell_path = folder / "cells" / f"cell_{target_id.replace(':', '_')}.pkl"
        if cell_path.exists():
            with cell_path.open("rb") as f:
                cached = pickle.load(f)
            if cached["lock_sha"] != LOCK_SHA or cached["epoch"] != epoch or cached["subject_id"] != target_id:
                raise RuntimeError("Private cell cache provenance mismatch")
            continue
        with (folder / f"epoch_{epoch:02d}_representations.pkl").open("rb") as f:
            payload = pickle.load(f)
        if payload["epoch"] != epoch or payload["fold"] != fold or target_id not in payload["val"]:
            raise RuntimeError("Selected-epoch representation mismatch")
        base_path = folder / f"epoch_{epoch:02d}_base.pkl"
        if base_path.exists():
            with base_path.open("rb") as f:
                base = pickle.load(f)
            if base["lock_sha"] != LOCK_SHA or base["epoch"] != epoch or base["fold"] != fold:
                raise RuntimeError("Private FIT model cache provenance mismatch")
        else:
            base = make_epoch_base(payload)
            tmp = base_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(base, f, protocol=5)
            tmp.replace(base_path)
        cell = analyze_target(payload, base, target_id)
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cell_path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(cell, f, protocol=5)
        tmp.replace(cell_path)
        print(f"[CELL] fold={fold} complete={sum(1 for _ in (folder/'cells').glob('*.pkl'))}/13 epoch={epoch}", flush=True)
    write_json(folder / "analysis_status.json", {"fold": fold, "cells": 13, "lock_sha": LOCK_SHA,
               "outer_predictions_metrics": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    opt = parser.parse_args()
    preflight()
    for fold in range(1, 6):
        if opt.fold is None or opt.fold == fold:
            run_fold(fold)


if __name__ == "__main__":
    main()
