"""Locked provenance, exact A1 contexts, deterministic FIT partitions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "active_fewshot_patient_calibration_seed42_v1" / "code"))
import common as afc  # noqa: E402

LOCK_SHA = "66f8468cf0987e0f53ee9cff204f1caefc12886a5cc3737d568c4a402920865b"
SOURCE_AP = .5767434626151353
B8_CURRENT = .5996322681940147
B8_BEST = .6052912572363056
LAMBDA_GRID = (0.0, .03, .1, .3)
R1 = "R1_DEEPSET_RESIDUAL"
R2 = "R2_PAIRWISE_RELRANK"
R3 = "R3_WEIGHTED_RELRANK"
ARCHS = (R1, R2, R3)
ARMS = ("BCE_ONLY", "RANK_SELECTED")
CONTEXTS = ("FULL_PATIENT_CONTEXT", "QUERY_ONLY_CONTEXT", "WRONG_PATIENT_CONTEXT", "SHUFFLED_RELATION")
RUNTIME = Path(os.environ.get("PSRR_RUNTIME", ""))
SRGI_RUNTIME = Path(os.environ.get("SRGI_RUNTIME", ""))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_pickle(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f: pickle.dump(row, f, protocol=5)
    tmp.replace(path)


def preflight():
    for name, path in (("PSRR_RUNTIME", RUNTIME), ("SRGI_RUNTIME", SRGI_RUNTIME)):
        if not os.environ.get(name) or not path.is_absolute():
            raise RuntimeError(f"{name} must be an absolute private data path")
    if sha(ROOT / "PROTOCOL_LOCK.json") != LOCK_SHA:
        raise RuntimeError("Patient-set relational protocol lock changed")
    source = json.loads((PROJECT / "zeroshot_geometry_canonicalization_metadg_seed42_v1" /
                         "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    b0 = json.loads((PROJECT / "zeroshot_geometry_canonicalization_metadg_seed42_v1" /
                     "B0_IDENTITY_AUDIT.json").read_text(encoding="utf-8"))
    if (source["checkpoints"] != 150 or source["max_grid_error"] > 1e-6 or
            source["R4_max_logit_replay_error"] > 1e-6 or
            abs(b0["source_ap"] - SOURCE_AP) > 1e-6 or not b0["pass"]):
        raise RuntimeError("Exact A1 source/B0 reproduction failed")
    afc.preflight()


def source_rows(fold):
    rows = afc.read_csv(SRGI_RUNTIME / "private" / f"fold_{fold}" / "A1_VLOO_PRIVATE.csv")
    if len(rows) != 13 or {int(r["fold"]) for r in rows} != {fold}:
        raise RuntimeError("Exact A1 13-cell fold selection changed")
    return rows


@lru_cache(maxsize=5)
def source_contexts(fold):
    contexts = {}
    for row in source_rows(fold):
        epoch = int(row["selected_epoch"])
        tau_text = row["selected_threshold"]
        tau = float(tau_text)
        if epoch not in range(1, 31) or not 0 < tau < 1:
            raise RuntimeError("Invalid A1 checkpoint or threshold")
        key = (epoch, tau_text)
        if key not in contexts:
            digest = hashlib.sha256(f"{fold}|{epoch}|{tau_text}".encode()).hexdigest()[:10]
            contexts[key] = dict(fold=fold, epoch=epoch, threshold=tau,
                                 threshold_text=tau_text,
                                 context_id=f"e{epoch:02d}_{digest}", target_ids=[])
        contexts[key]["target_ids"].append(row["subject_id"])
    return tuple(sorted(contexts.values(), key=lambda x: (x["epoch"], x["threshold"])))


def all_contexts():
    result = [ctx for fold in range(1, 6) for ctx in source_contexts(fold)]
    if len(result) != 17 or sum(len(x["target_ids"]) for x in result) != 65:
        raise RuntimeError("Expected 17 distinct selected A1 contexts covering 65 cells")
    return result


def context_by_id(fold, context_id):
    return next(c for c in source_contexts(fold) if c["context_id"] == context_id)


@lru_cache(maxsize=3)
def payload(fold, epoch):
    return afc.load_representation(fold, epoch)


def fit_partition(ids, fold):
    ordered = sorted(ids)
    rng = np.random.default_rng(afc.stable_seed(42, fold, "psrr_fit_patient_split"))
    perm = rng.permutation(len(ordered))
    nval = max(1, int(round(.2 * len(ordered))))
    train = [ordered[int(i)] for i in perm[nval:]]
    val = [ordered[int(i)] for i in perm[:nval]]
    if set(train) & set(val) or set(train + val) != set(ids):
        raise RuntimeError("FIT patient split is not disjoint/exhaustive")
    return train, val


def scaler_for(fit, train_ids):
    s = StandardScaler().fit(np.concatenate([np.asarray(fit[sid]["R4"], dtype=np.float64)
                                             for sid in sorted(train_ids)]))
    if s.mean_.shape != (64,) or s.scale_.shape != (64,) or not np.isfinite(s.scale_).all():
        raise RuntimeError("FIT-only R4 scaler invalid")
    return s


def patient_arrays(row, scaler, threshold):
    z = scaler.transform(np.asarray(row["R4"], dtype=np.float64)).astype(np.float32)
    m0 = (math.log(threshold / (1 - threshold)) + np.asarray(row["source_ez"], dtype=np.float64)).astype(np.float32)
    y = np.asarray(row["y"], dtype=np.float32)
    if z.shape != (len(m0), 64) or y.shape != m0.shape or not np.isfinite(z).all() or not np.isfinite(m0).all():
        raise RuntimeError("Malformed selected-epoch A1 R4/margin")
    return z, m0, y


def arm_name(arch, arm):
    return arch + "__" + arm
