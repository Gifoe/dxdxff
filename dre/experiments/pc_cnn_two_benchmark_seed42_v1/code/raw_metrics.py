"""Validation-only metric units for the matched RawCNN baseline."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)


def binary_metrics(y, score, threshold=0.5):
    y, score = np.asarray(y, int), np.asarray(score, float)
    if len(y) != len(score) or not len(y) or not np.isfinite(score).all():
        raise RuntimeError("Invalid frozen validation score arrays")
    pred = score >= threshold
    result = {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    if len(np.unique(y)) < 2:
        result.update({"auroc": None, "ap": None, "mrr": None, "top1": None})
    else:
        ordering = np.argsort(-score, kind="stable")
        positions = np.flatnonzero(y[ordering] == 1)
        result.update({"auroc": float(roc_auc_score(y, score)),
                       "ap": float(average_precision_score(y, score)),
                       "mrr": float(1 / (1 + positions[0])),
                       "top1": float(y[ordering[0]])})
    return result


def _mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def ictal_validation(patient_records):
    """Mean record probabilities to canonical patient-channel, then equal patients."""
    patients, private = [], {}
    for patient, records in patient_records.items():
        scores, label = defaultdict(list), {}
        for record in records:
            for channel, y, p in zip(record["channel"], record["label"], record["score"]):
                if y >= 0:
                    if channel in label and label[channel] != y:
                        raise RuntimeError("Conflicting ictal channel labels")
                    label[channel] = int(y)
                    scores[channel].append(float(p))
        ids = sorted(scores)
        y = np.asarray([label[k] for k in ids], int)
        p = np.asarray([np.mean(scores[k]) for k in ids], float)
        patients.append(binary_metrics(y, p))
        private[patient] = {"labels": y.tolist(), "scores": p.tolist()}
    if not patients:
        raise RuntimeError("Empty ictal validation cohort")
    return {"auroc": _mean(patients, "auroc"), "ap": _mean(patients, "ap"),
            "mrr": _mean(patients, "mrr"), "top1": _mean(patients, "top1"),
            "macro_f1_0_5": _mean(patients, "macro_f1"),
            "patients": len(patients)}, private


def omni_validation(patient_records):
    """Official pooled (EDF,channel) averages, plus patient-equal diagnostics."""
    pooled_y, pooled_score, patient_rows, private = [], [], [], {}
    for patient, records in patient_records.items():
        by_edf_channel, labels = defaultdict(list), {}
        for record in records:
            for channel, y, p in zip(record["channel"], record["label"], record["score"]):
                if y < 0:
                    continue
                key = (record["edf"], channel)
                if key in labels and labels[key] != y:
                    raise RuntimeError("Conflicting official EDF-channel labels")
                labels[key] = int(y)
                by_edf_channel[key].append(float(p))
        keys = sorted(by_edf_channel)
        y = np.asarray([labels[k] for k in keys], int)
        score = np.asarray([np.mean(by_edf_channel[k]) for k in keys], float)
        pooled_y.extend(y.tolist())
        pooled_score.extend(score.tolist())
        by_channel, channel_labels, conflicts = defaultdict(list), {}, set()
        for (_, channel), label, value in zip(keys, y, score):
            if channel in channel_labels and channel_labels[channel] != label:
                conflicts.add(channel)
            channel_labels[channel] = int(label)
            by_channel[channel].append(float(value))
        canonical = sorted(set(by_channel) - conflicts)
        if not canonical:
            raise RuntimeError("No consistent patient-channel labels for ranking")
        patient_rows.append(binary_metrics(
            [channel_labels[channel] for channel in canonical],
            [np.mean(by_channel[channel]) for channel in canonical]))
        private[patient] = {"labels": y.tolist(), "scores": score.tolist(),
                            "edf": [key[0] for key in keys],
                            "channel": [key[1] for key in keys]}
    pooled = binary_metrics(pooled_y, pooled_score)
    return {"auroc": pooled["auroc"], "ap": pooled["ap"],
            "macro_f1_0_5": pooled["macro_f1"],
            "patient_equal_auroc": _mean(patient_rows, "auroc"),
            "patient_ap": _mean(patient_rows, "ap"),
            "mrr": _mean(patient_rows, "mrr"),
            "top1": _mean(patient_rows, "top1"),
            "patients": len(patient_rows), "edf_channel_units": len(pooled_y)}, private
