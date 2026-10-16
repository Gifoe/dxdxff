"""Pinned sources and legal role loader, with no Outer result/label archive reader."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
PUB = ROOT / 'public'
BASE = Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
SOURCE = Path(r'E:\DRE-nips\new-pipeline\7-11')
PRIOR = Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
GRAPH = Path(r'C:\d2_graphrole_seed42_runtime')
sys.path.insert(0, str(SOURCE / 'raw_delta_encoder_seed42_v1/code'))
from io_utils import sha, read, save, write, load
from dependencies import context
from raw_store import Store

ORIGINAL7 = ['log_bp_beta', 'log_bp_low_gamma', 'partial_high_gamma_80_125', 'rms_legacy', 'variance', 'line_length_per_sec', 'spectral_entropy_10_125']
NEW11 = ['log_variance_stable', 'hjorth_mobility_per_second', 'hjorth_complexity_stable', 'permutation_entropy_m3_tau1', 'spectral_centroid_13_95', 'log_gamma_beta_ratio', 'partial_high_gamma_fraction_80_125', 'spectral_flatness_13_95', 'spectral_slope_proxy_20_95_notchgap', 'normalized_line_length', 'clipped_excess_kurtosis']
LOG3 = ['stable_log_beta_sameband', 'stable_log_low_gamma_sameband', 'stable_log_partial_high_gamma_sameband']
EXTRA4 = ['partial_alpha_10_13_legacy', 'log_total_power_10_125_legacy', 'hjorth_mobility_legacy', 'hjorth_complexity_legacy']
GRAPH7 = ['rebuilt_degree_norm', 'rebuilt_strength_norm', 'rebuilt_clustering_coeff', 'rebuilt_eigenvector_centrality', 'rebuilt_pagerank', 'rebuilt_kcore_norm', 'rebuilt_local_efficiency']
NAMES = ORIGINAL7 + NEW11 + LOG3 + EXTRA4 + GRAPH7
assert len(NAMES) == len(set(NAMES)) == 32
METRICS = ['accuracy', 'auroc', 'auprc', 'macro_f1', 'f1', 'nez_f1', 'mrr', 'hit_at_1', 'ez_fraction_bias']


def jr(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def legal():
    c, core, fac, trainer, records, rec, banks, hist, deps = context(SOURCE, BASE)
    for bank in banks:
        assert set(bank['patient']) == bank['roles']['fit'] | bank['roles']['validation']
        assert set(bank['patient']).isdisjoint(bank['roles']['test'])
    return c, core, fac, trainer, records, rec, banks, deps


def source_binding():
    return {p.name: sha(p) for p in HERE.iterdir() if p.suffix in ['.py', '.json']}


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert jr(path) == value, path.name
    else:
        write(path, value)


def norm(name):
    return ''.join(str(name).upper().split()).replace('-', '').replace('_', '')


def ev():
    return load('discovery_v2_original_metrics', Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))
