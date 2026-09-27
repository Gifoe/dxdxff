"""Frozen A1 few-shot audit helpers. Patient-level material stays in PRIVATE_RUNTIME."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PRIOR = PROJECT / "seizure_resolved_geometry_identifiability_seed42_v1"
PRIOR_RUNTIME = Path(os.environ.get("SRGI_RUNTIME", ""))
RUNTIME = Path(os.environ.get("AFPC_RUNTIME", ""))
SOURCE_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))
LOCK_SHA = "0a8ef24d36f6e942aa0587565270ddf72740f1bacbed687967395e06226e8821"
PRIOR_LOCK_SHA = "d9d29091e7d298b02e5031359f0ad913a6b911d0a0b49cf2ed7a55b83d79ce29"
SOURCE_LOCK_SHA = "6694da1b351d7130015661fcce1ec91bec643821392e409f1d5728f472b3aba0"
BUDGETS = (0, 1, 2, 4, 8, 16)
POLICIES = ("RANDOM", "UNCERTAINTY", "UNCERTAINTY_DIVERSITY", "ORACLE_BALANCED_RANDOM")
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    for key, path in (("AFPC_RUNTIME", RUNTIME), ("SRGI_RUNTIME", PRIOR_RUNTIME), ("A1_A2_RUNTIME", SOURCE_RUNTIME)):
        if not os.environ.get(key) or not path.is_absolute():
            raise RuntimeError(f"{key} must be a private absolute path")
    for path, expected in ((ROOT / "PROTOCOL_LOCK.json", LOCK_SHA),
                           (PRIOR / "PROTOCOL_LOCK.json", PRIOR_LOCK_SHA),
                           (PROJECT / "a1_a2_patient_equal_objective_seed42_v1" / "PROTOCOL_LOCK.json", SOURCE_LOCK_SHA)):
        if sha(path) != expected:
            raise RuntimeError(f"Frozen protocol provenance mismatch: {path}")
    if not (SOURCE_RUNTIME / "A1" / "fold_5" / "epoch_30.pt").is_file():
        raise RuntimeError("A1 checkpoint grid incomplete")


def stable_seed(*parts):
    data = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**32)


def split_indices(n, *seed_parts):
    if n < 4:
        raise RuntimeError("Too few valid channels for fixed calibration/query split")
    perm = np.random.default_rng(stable_seed(*seed_parts, "split")).permutation(n)
    return perm[:n // 2], perm[n // 2:]


def rank_metrics(y, score):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if len(y) != len(score) or len(y) == 0 or not np.isfinite(score).all():
        raise RuntimeError("Invalid fixed-query labels/scores")
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos == 0 or nneg == 0:
        return dict(ap=np.nan, auc=np.nan, mrr=np.nan, top1=np.nan, ndcg=np.nan)
    order = np.argsort(-score, kind="stable")
    first = np.flatnonzero(y[order] == 1)[0]
    discounts = np.log2(np.arange(2, len(y) + 2))
    return dict(ap=float(average_precision_score(y, score)), auc=float(roc_auc_score(y, score)),
                mrr=float(1.0 / (first + 1)), top1=float(y[order[0]]),
                ndcg=float(np.sum(y[order] / discounts) / np.sum(np.sort(y)[::-1] / discounts)))


def query_metrics(y, margin):
    out = rank_metrics(y, margin)
    y = np.asarray(y, dtype=np.int8)
    pred = (np.asarray(margin) > 0).astype(np.int8)
    out.update(macro_f1=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
               ez_f1=float(f1_score(y, pred, pos_label=1, zero_division=0)),
               nez_f1=float(f1_score(y, pred, pos_label=0, zero_division=0)),
               ba=float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
               predicted_ez_fraction=float(pred.mean()),
               query_n=len(y), query_ez=int(y.sum()), query_nez=int(len(y)-y.sum()))
    return out


def fit_residual(m0, z, y, lambda_w, bias_only=False):
    """Deterministic convex support fit, freshly zero initialized each call."""
    y = np.asarray(y, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if len(y) == 0:
        return 0.0, np.zeros(z.shape[1], dtype=np.float64)
    if len(y) != len(m0) or len(y) != len(z) or not np.isin(y, [0, 1]).all():
        raise RuntimeError("Invalid acquired support")
    t = 2*y-1
    n = len(y)
    d = 0 if bias_only else z.shape[1]

    def objective(theta):
        b = theta[0]
        w = theta[1:] if d else None
        margin = m0 + b + (z @ w if d else 0.0)
        loss = float(np.mean(np.logaddexp(0.0, -t*margin)) + b*b +
                     (float(lambda_w) * float(w @ w) if d else 0.0))
        deriv = -t*expit(-t*margin) / n
        grad_b = float(np.sum(deriv) + 2*b)
        grad = np.r_[grad_b, z.T @ deriv + 2*lambda_w*w] if d else np.array([grad_b])
        return loss, grad

    result = minimize(objective, np.zeros(d+1), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or np.linalg.norm(result.jac) > 1e-3:
        raise RuntimeError(f"Convex calibration optimizer failed: {result.message}")
    return float(result.x[0]), np.asarray(result.x[1:], dtype=np.float64) if d else np.zeros(z.shape[1])


def oracle_direction(x, y):
    y = np.asarray(y, dtype=np.int8)
    k = min(5, int(np.sum(y==1)), int(np.sum(y==0)))
    if k < 2:
        return None
    directions=[]
    for train, _ in StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x, y):
        model=LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        model.fit(x[train], y[train])
        w=model.coef_[0].astype(np.float64)
        directions.append(w / np.linalg.norm(w))
    w=np.mean(directions, axis=0)
    return w / np.linalg.norm(w)


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_representation(fold, epoch):
    path=PRIOR_RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}_representations.pkl"
    with path.open("rb") as f:
        payload=pickle.load(f)
    if payload["fold"] != fold or payload["epoch"] != epoch or payload["r4_dim"] != 64 or payload["r4_error"] > 1e-6:
        raise RuntimeError("Selected exact-R4 cache provenance failed")
    return payload
