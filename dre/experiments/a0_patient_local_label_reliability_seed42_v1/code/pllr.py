"""FIT-only PCA8, leave-one-channel-out geometry and frozen class-mass weights."""
import hashlib
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(list(parts), ensure_ascii=True,
        separators=(',', ':')).encode()).digest()[:8], 'little')


def class_normalize(raw, y):
    w = np.asarray(raw, dtype=float).copy()
    for label in [0, 1]:
        take = y == label
        if take.any(): w[take] /= w[take].mean()
    assert np.isfinite(w).all() and (w > 0).all()
    return w


def leave_one_out(z, y):
    """The query is absent from both labeled prototypes, never its own anchor."""
    z = np.asarray(z, dtype=float); y = np.asarray(y, dtype=int)
    counts = [int((y == k).sum()) for k in [0, 1]]
    eligible = min(counts) >= 5
    if not eligible: return np.zeros(len(z)), False
    margin = np.empty(len(z))
    for label in [0, 1]:
        take = y == label
        same = (z[take].sum(0)[None, :] - z[take]) / (counts[label] - 1)
        other = z[~take].mean(0)
        margin[take] = ((z[take] - same)**2).mean(1) - ((z[take] - other)**2).mean(1)
    return margin, True


def raw_weights(margin, tau):
    if tau is None: return np.ones(len(margin))
    conflict = np.maximum(np.asarray(margin, dtype=float), 0.)
    return 1. - .5 * conflict / (conflict + tau)


def permutation(weights, y, channels, fold, patient):
    out = np.asarray(weights, dtype=float).copy(); changed = []
    for label in [0, 1]:
        ix = np.flatnonzero(y == label)
        ix = ix[np.argsort(np.asarray(channels)[ix].astype(str), kind='stable')]
        if len(ix) < 2: continue
        rng = np.random.default_rng(stable_seed(42, fold, str(patient), label))
        perm = rng.permutation(len(ix))
        nonconstant = np.ptp(weights[ix]) > 1e-12
        # Fixed before data/outcomes: avoid accidentally retaining the entire
        # feature correspondence in a nonconstant group. Duplicated weights are
        # legal, so report the realized unchanged-channel fraction as well.
        if nonconstant:
            for _ in range(100):
                if not np.array_equal(weights[ix][perm], weights[ix]): break
                perm = rng.permutation(len(ix))
            if np.array_equal(weights[ix][perm], weights[ix]): perm = np.roll(np.arange(len(ix)), 1)
        out[ix] = weights[ix][perm]
        assert np.array_equal(np.sort(out[ix]), np.sort(weights[ix]))
        if nonconstant: assert not np.array_equal(out[ix], weights[ix])
        changed.extend((out[ix] != weights[ix]).tolist())
    return out, changed


def rank_correlation(a, b):
    ca, cb = np.ptp(a) <= 1e-12, np.ptp(b) <= 1e-12
    if ca or cb: return 1. if ca and cb and np.allclose(a, b, atol=1e-12, rtol=0) else 0.
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float(np.dot(ra, rb) / np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))


def stability(z, y, weights, tau, fold, patient, draws=100):
    rng = np.random.default_rng(stable_seed(42, fold, str(patient), 'prototype_bootstrap'))
    margins = np.empty((draws, len(y)))
    for c in range(len(y)):
        same = np.flatnonzero(y == y[c]); same = same[same != c]
        other = np.flatnonzero(y != y[c])
        assert len(same) >= 4 and len(other) >= 5 and c not in same and c not in other
        ms = z[same[rng.integers(0, len(same), (draws, len(same)))]].mean(1)
        mo = z[other[rng.integers(0, len(other), (draws, len(other)))]].mean(1)
        margins[:, c] = ((z[c] - ms)**2).mean(1) - ((z[c] - mo)**2).mean(1)
    boot = np.asarray([class_normalize(raw_weights(m, tau), y) for m in margins])
    rho = np.asarray([rank_correlation(weights, row) for row in boot])
    original_margin, _ = leave_one_out(z, y)
    return {'mean_rho': float(rho.mean()), 'q10_rho': float(np.quantile(rho, .1)),
        'mean_abs_weight_change': float(np.abs(boot - weights).mean()),
        'conflict_sign_change_fraction': float(((margins > 0) != (original_margin > 0)).mean()),
        'constant_original': bool(np.ptp(weights) <= 1e-12)}


def build_pllr(x_fit, y_fit, patient, channel, center, fold, draws=100):
    """Interface deliberately takes FIT arrays only, not global labels/roles."""
    x = np.asarray(x_fit, dtype=float); y = np.asarray(y_fit, dtype=int)
    patient, channel, center = [np.asarray(a).astype(str) for a in [patient, channel, center]]
    assert x.ndim == 2 and x.shape[1] == 88 and len(x) >= 8
    assert np.isfinite(x).all() and set(y) <= {0, 1}
    assert len(set(zip(patient, channel))) == len(x)
    # Canonical order makes PCA floating-point sums and seeded permutations
    # independent of the caller's channel ordering.
    order = np.lexsort((channel, patient)); inverse = np.argsort(order)
    x, y, patient, channel, center = [v[order] for v in [x, y, patient, channel, center]]
    pca = PCA(n_components=8, svd_solver='full')
    z = pca.fit_transform(x)
    mean, std = z.mean(0), z.std(0)
    scale = np.where(std > 1e-12, std, 1.)
    z = (z - mean) / scale
    assert np.isfinite(z).all()
    margin = np.zeros(len(x)); eligible = np.zeros(len(x), bool)
    indices = [np.flatnonzero(patient == p) for p in np.unique(patient)]
    for ix in indices:
        margin[ix], good = leave_one_out(z[ix], y[ix]); eligible[ix] = good
    positive = margin[eligible & (margin > 0)]
    tau = float(np.median(positive)) if len(positive) else None
    raw = raw_weights(margin, tau); raw[~eligible] = 1.
    w = np.empty(len(x)); b1 = w.copy(); patients = []; classes = []; stability_rows = []
    for ix in indices:
        p = patient[ix[0]]; ce = center[ix[0]]
        assert len(set(center[ix])) == 1
        w[ix] = class_normalize(raw[ix], y[ix])
        b1[ix], changed = permutation(w[ix], y[ix], channel[ix], fold, p)
        good = bool(eligible[ix[0]])
        patients.append({'fold': fold, 'center': ce, 'channels': len(ix),
            'eligible': good, 'eligible_channels': len(ix) if good else 0,
            'B1_changed_fraction': float(np.mean(changed)) if changed else 0.,
            'nonconstant': bool(np.ptp(w[ix]) > 1e-12)})
        if good:
            stability_rows.append({'fold': fold, 'center': ce,
                **stability(z[ix], y[ix], w[ix], tau, fold, p, draws)})
        else: assert np.array_equal(w[ix], np.ones(len(ix)))
        for label in [0, 1]:
            ci = ix[y[ix] == label]
            if not len(ci): continue
            assert abs(w[ci].sum() - len(ci)) < 1e-9
            assert np.array_equal(np.sort(b1[ci]), np.sort(w[ci]))
            classes.append({'fold': fold, 'center': ce, 'observed_class': 'EZ' if label == 0 else 'NEZ',
                'channels': len(ci), 'raw_min': float(raw[ci].min()), 'raw_max': float(raw[ci].max()),
                'raw_mean': float(raw[ci].mean()), 'weight_min': float(w[ci].min()),
                'weight_max': float(w[ci].max()), 'weight_mean': float(w[ci].mean()),
                'weight_std': float(w[ci].std()), 'effective_sample_size': float(w[ci].sum()**2 / np.square(w[ci]).sum()),
                'raw_downweighted': int((raw[ci] < 1).sum()), 'final_downweighted': int((w[ci] < 1).sum()),
                'mass_error': float(abs(w[ci].sum() - len(ci)))})
    artifact = {'pca_components': pca.components_, 'pca_mean': pca.mean_,
        'explained_variance_ratio': np.nan_to_num(pca.explained_variance_ratio_),
        'projection_mean': mean, 'projection_scale': scale, 'tau': tau,
        'z': z[inverse], 'margin': margin[inverse], 'raw': raw[inverse],
        'eligible': eligible[inverse], 'B2': w[inverse], 'B1': b1[inverse]}
    return artifact, patients, classes, stability_rows
