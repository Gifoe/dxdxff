"""Label-free sampling, normalization and correspondence control."""
import hashlib
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly
from scipy.stats import rankdata


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def select_windows(times, valid, patient, run, fold_seed, epoch=None):
    """Aligned record schedule; each channel filters common priority by validity.

    Strata are chronological thirds of record candidate centers. Deficits are
    filled by the shared deterministic chronological-coverage order. No labels.
    """
    times = np.asarray(times)
    order = np.argsort(times, kind='stable')
    strata = np.array_split(order, 3)
    rng = np.random.default_rng(stable_seed(fold_seed, patient, run, epoch))
    priorities = []
    for group in strata:
        if epoch is not None:
            priorities.append(rng.permutation(group))
        else:
            # Four interleaved coverage passes, followed by remaining chronological centers.
            q = np.unique(np.rint(np.linspace(0, max(len(group)-1, 0), min(4, len(group)))).astype(int))
            priorities.append(np.r_[group[q], np.setdiff1d(group, group[q], assume_unique=True)])
    fill = np.array([x for group in priorities for x in group], dtype=int)
    out = np.full((len(valid), 12), -1, dtype=int)
    for c, mask in enumerate(valid):
        chosen = []
        for group in priorities:
            chosen.extend([int(j) for j in group if mask[j]][:4])
        for j in fill:
            if len(chosen) >= 12:
                break
            if mask[j] and j not in chosen:
                chosen.append(int(j))
        out[c, :len(chosen)] = sorted(chosen, key=lambda j: times[j])
    return out


def resample_window(wave, sfreq):
    ratio = Fraction(256 / sfreq).limit_denominator(100000)
    assert wave.shape[-1] == round(2 * sfreq)
    out = resample_poly(wave, ratio.numerator, ratio.denominator, axis=-1,
                        window=('kaiser', 5.0), padtype='constant')
    assert out.shape[-1] == 512, 'resampling must not silently truncate'
    return out.astype(np.float32)


def normalize_record(wave, start, length, starts, valid, sfreq):
    """Median/MAD from measured context only, raw-relative energy before scaling."""
    context = wave[:, start:start+length]
    median = np.nanmedian(context, axis=1)
    scale = 1.4826 * np.nanmedian(np.abs(context - median[:, None]), axis=1)
    # Fixed numerical floor; no population/label/validation fitting.
    scale = np.maximum(scale, 1e-8)
    result = np.zeros((*valid.shape, 512), np.float32)
    aux = np.zeros((*valid.shape, 2), np.float32)
    size = round(2 * sfreq)
    for j, a in enumerate(starts):
        ix = np.flatnonzero(valid[:, j])
        if not len(ix):
            continue
        raw = wave[ix, a:a+size].astype(np.float64)
        energy = np.log(np.maximum(np.mean(raw ** 2, axis=1), 1e-20))
        aux[ix, j, 0] = energy - np.median(energy)
        aux[ix, j, 1] = (rankdata(energy, method='average') - .5) / len(ix)
        result[ix, j] = resample_window((raw - median[ix, None]) / scale[ix, None], sfreq)
    assert np.isfinite(result).all() and np.isfinite(aux).all()
    return result, aux


def permutation(seizure_valid, patient, seed=42):
    """Within-patient derangement of equal seizure-availability signatures.

    This keeps run membership and availability unchanged, while using one fixed
    canonical-channel assignment across all corresponding seizures. Singleton
    signatures cannot be shuffled and are explicitly counted.
    """
    groups = {}
    for c, signature in enumerate(seizure_valid):
        if signature.any():
            groups.setdefault(tuple(signature), []).append(c)
    out = np.arange(len(seizure_valid))
    rng = np.random.default_rng(stable_seed(seed, patient, 'shuffle'))
    for members in groups.values():
        if len(members) > 1:
            order = rng.permutation(members)
            out[order] = np.roll(order, 1)
    return out
