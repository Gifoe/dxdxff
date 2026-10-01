"""One-time TRAIN aggregation parity probe; no model inference and no writes.

The fixed model emits a normal-class logit.  This diagnostic enumerates only
historically meaningful aggregation units; it does not select a threshold,
change a model, or inspect test data.
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


root = Path(sys.argv[1])
y, sigmoid_mean, mean_sigmoid, first, last = [], [], [], [], []
segment_y, segment_normal_score = [], []
patient_channel_scores = {}
patient_channel_labels = {}
for path in sorted(root.glob("*.npz")):
    with np.load(path, allow_pickle=False) as z:
        offsets, logits = z["segment_offsets"], z["segment_logits"]
        labels = z["pathological_labels"].astype(np.int64)
        patient = str(z["patient"].item())
        channels = z["channel_names"].astype(str)
        y.extend(labels.tolist())
        for index in range(len(z["channel_names"])):
            value = logits[offsets[index]:offsets[index + 1]]
            probability = float((1.0 / (1.0 + np.exp(-value))).mean())
            sigmoid_mean.append(probability)
            mean_sigmoid.append(float(1.0 / (1.0 + np.exp(-value.mean()))) )
            first.append(float(1.0 / (1.0 + np.exp(-value[0]))))
            last.append(float(1.0 / (1.0 + np.exp(-value[-1]))))
            if labels[index] >= 0:
                segment_y.extend([int(labels[index])] * len(value))
                segment_normal_score.extend((1.0 / (1.0 + np.exp(-value))).tolist())
            if labels[index] >= 0:
                key = (patient, str(channels[index]))
                patient_channel_scores.setdefault(key, []).append(probability)
                patient_channel_labels.setdefault(key, set()).add(int(labels[index]))
for name, score in [("sigmoid_then_mean", sigmoid_mean), ("mean_logit_then_sigmoid", mean_sigmoid),
                    ("first_segment", first), ("last_segment", last)]:
    y_array, score_array = np.asarray(y), np.asarray(score)
    keep = y_array >= 0
    print(name, f"{roc_auc_score(y_array[keep], score_array[keep]):.10f}")
print("all_pairs", len(y), "labeled_pairs", int(np.sum(np.asarray(y) >= 0)), "class_counts", np.bincount(np.asarray(y)[np.asarray(y) >= 0]).tolist())
print("segment_level", f"{roc_auc_score(np.asarray(segment_y), np.asarray(segment_normal_score)):.10f}",
      "segments", len(segment_y))

conflicting = {key: labels for key, labels in patient_channel_labels.items() if len(labels) != 1}
if conflicting:
    print("patient_channel_label_conflicts", len(conflicting))
else:
    grouped_y = np.asarray([next(iter(labels)) for labels in patient_channel_labels.values()])
    grouped_normal_score = np.asarray([
        float(np.mean(patient_channel_scores[key])) for key in patient_channel_labels
    ])
    print("patient_channel_mean_across_edf", f"{roc_auc_score(grouped_y, grouped_normal_score):.10f}")
    print("patient_channel_units", len(grouped_y), "class_counts", np.bincount(grouped_y).tolist())
