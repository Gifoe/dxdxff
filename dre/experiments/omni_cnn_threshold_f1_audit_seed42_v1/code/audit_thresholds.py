"""Post-hoc operating-point analysis of one SHA-verified frozen score table.

No model load, training, signal extraction, score adjustment, or threshold
re-optimization inside the primary bootstrap is permitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def metrics(tp, fp, tn, fn) -> dict[str, np.ndarray]:
    tp, fp, tn, fn = (np.asarray(x, dtype=float) for x in (tp, fp, tn, fn))
    pathological_precision = ratio(tp, tp + fp)
    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    pathological_f1 = ratio(2 * tp, 2 * tp + fp + fn)
    normal_f1 = ratio(2 * tn, 2 * tn + fp + fn)
    total = tp + fp + tn + fn
    return {
        "macro_f1": (pathological_f1 + normal_f1) / 2,
        "pathological_f1": pathological_f1,
        "normal_f1": normal_f1,
        "pathological_precision": pathological_precision,
        "pathological_recall_sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "accuracy": ratio(tp + tn, total),
        "predicted_pathological_fraction": ratio(tp + fp, total),
        "youden_j": sensitivity + specificity - 1,
    }


def make_sweep(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(np.concatenate([score, [0.0, 0.5, 1.0]]))
    if not np.isfinite(thresholds).all() or (score < 0).any() or (score > 1).any():
        raise RuntimeError("Invalid pathological probabilities")
    order = np.argsort(score, kind="stable")
    scores_sorted = score[order]
    positives = y[order].astype(np.int64)
    negs = 1 - positives
    prefix_pos = np.concatenate([[0], np.cumsum(positives)])
    prefix_neg = np.concatenate([[0], np.cumsum(negs)])
    at = np.searchsorted(scores_sorted, thresholds, side="left")
    fn = prefix_pos[at]
    tn = prefix_neg[at]
    tp = prefix_pos[-1] - fn
    fp = prefix_neg[-1] - tn
    frame = pd.DataFrame({"threshold": thresholds, "TP": tp, "FP": fp, "TN": tn, "FN": fn})
    for name, value in metrics(tp, fp, tn, fn).items():
        frame[name] = value
    return frame


def row_for(frame: pd.DataFrame, threshold: float, tol: float) -> pd.Series:
    candidates = frame.threshold.to_numpy()
    matches = np.flatnonzero(candidates == threshold)
    if len(matches) == 0:
        matches = np.flatnonzero(np.abs(candidates - threshold) <= tol)
    if len(matches) != 1:
        raise RuntimeError(f"Operating point absent/ambiguous in candidate sweep: {threshold}")
    return frame.iloc[int(matches[0])]


def plateau(frame: pd.DataFrame, fraction: float, optimum: float,
            youden: float) -> dict:
    mask = frame.macro_f1.to_numpy() >= fraction * optimum - 1e-15
    selected = frame.threshold.to_numpy()[mask]
    if not len(selected):
        raise RuntimeError("Empty F1 plateau")
    indices = np.flatnonzero(mask)
    discontinuities = np.flatnonzero(np.diff(indices) != 1) + 1
    runs = np.split(indices, discontinuities)
    return {
        "fraction_of_max": fraction,
        "candidate_count": len(selected),
        "candidate_thresholds": selected.tolist(),
        "lower_threshold": float(selected[0]),
        "upper_threshold": float(selected[-1]),
        "hull_width": float(selected[-1] - selected[0]),
        "disjoint_candidate_runs": [
            {"lower": float(frame.threshold.iloc[run[0]]),
             "upper": float(frame.threshold.iloc[run[-1]]),
             "candidate_count": len(run)} for run in runs],
        "youden_distance_to_nearest_candidate": float(np.min(np.abs(selected - youden))),
        "half_distance_to_nearest_candidate": float(np.min(np.abs(selected - 0.5))),
        "youden_in_hull": bool(selected[0] <= youden <= selected[-1]),
        "half_in_hull": bool(selected[0] <= 0.5 <= selected[-1]),
        "youden_candidate_qualifies": bool(np.any(np.abs(selected - youden) <= 1e-12)),
        "half_candidate_qualifies": bool(np.any(np.abs(selected - 0.5) <= 1e-12)),
    }


def bootstrap(frame: pd.DataFrame, youden: float, draws: int, seed: int) -> pd.DataFrame:
    y = frame.y_true.to_numpy(dtype=np.int8)
    scores = frame.pathological_score.to_numpy(dtype=float)
    patients = pd.Categorical(frame.patient)
    codes = patients.codes
    count = len(patients.categories)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, count, size=(draws, count), endpoint=False)
    result = {"draw": np.arange(draws, dtype=int)}
    for name, threshold in (("youden", youden), ("half", 0.5)):
        pred = scores >= threshold
        cluster = np.stack([
            np.bincount(codes, weights=((y == 1) & pred), minlength=count),
            np.bincount(codes, weights=((y == 0) & pred), minlength=count),
            np.bincount(codes, weights=((y == 0) & ~pred), minlength=count),
            np.bincount(codes, weights=((y == 1) & ~pred), minlength=count),
        ], axis=1).astype(np.int64)
        totals = cluster[sampled].sum(axis=1)
        m = metrics(totals[:, 0], totals[:, 1], totals[:, 2], totals[:, 3])
        for k in ("macro_f1", "pathological_f1", "pathological_recall_sensitivity", "specificity"):
            result[f"{k}_{name}"] = m[k]
        result[f"positive_class_absent_{name}"] = (totals[:, 0] + totals[:, 3]) == 0
        result[f"normal_class_absent_{name}"] = (totals[:, 1] + totals[:, 2]) == 0
    result["macro_f1_difference_half_minus_youden"] = result["macro_f1_half"] - result["macro_f1_youden"]
    return pd.DataFrame(result)


def plot_all(sweep: pd.DataFrame, frozen: pd.DataFrame, points: dict[str, float],
             output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2,
                         "pdf.fonttype": 42})
    colors = {"Youden": "#b04a46", "0.5": "#276895", "F1 oracle": "#6d5794"}
    def marks(ax):
        for label, x in points.items():
            ax.axvline(x, color=colors[label], linewidth=1.2, linestyle="--",
                       label=f"{label} ({x:.4g})")
    x = sweep.threshold.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, sweep.macro_f1, color="#2d3748", linewidth=1.3)
    marks(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Pathological-score threshold",
           ylabel="Macro-F1", title="Frozen Omni CNN: Macro-F1 vs threshold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(output / "macro_f1_vs_threshold.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, sweep.pathological_recall_sensitivity, label="Sensitivity", color="#b04a46")
    ax.plot(x, sweep.specificity, label="Specificity", color="#276895")
    marks(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Pathological-score threshold",
           ylabel="Rate", title="Sensitivity and specificity")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(output / "sensitivity_specificity_vs_threshold.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, sweep.youden_j, label="Youden J", color="#b04a46")
    ax.plot(x, sweep.macro_f1, label="Macro-F1", color="#276895")
    marks(ax)
    ax.set(xlim=(0, 1), ylim=(-0.1, 1), xlabel="Pathological-score threshold",
           ylabel="Metric value (same scale)", title="Youden J and Macro-F1 optima")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(output / "youden_vs_f1_vs_threshold.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, 1, 51)
    for label, color, name in ((0, "#276895", "Normal"), (1, "#b04a46", "Pathological")):
        vals = frozen.loc[frozen.y_true == label, "pathological_score"]
        ax.hist(vals, bins=bins, density=True, alpha=0.48, label=f"{name} (n={len(vals)})",
                color=color)
    marks(ax)
    ax.set(xlim=(0, 1), xlabel="Pathological score", ylabel="Within-class density",
           title="Frozen EDF-channel score distributions")
    ax.legend(loc="upper center", frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(output / "score_distribution_by_class.pdf"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--private-predictions", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    replay = json.loads(args.replay_audit.read_text(encoding="utf-8"))
    if replay["status"] != "PASS" or sha256(args.private_predictions) != replay["channel_predictions_sha256"]:
        raise RuntimeError("Private frozen score table missing or SHA mismatch")
    if replay["source_result_sha256"] != lock["source_result_sha256"] or replay["checkpoint_sha256"] != lock["frozen_checkpoint_sha256"]:
        raise RuntimeError("Replay provenance differs from locked source")
    frame = pd.read_csv(args.private_predictions)
    expected_columns = {"edf", "patient", "channel", "y_true", "pathological_score", "clips"}
    if not expected_columns.issubset(frame.columns) or len(frame) != lock["expected_labeled_pairs"]:
        raise RuntimeError("Frozen channel score schema/cohort mismatch")
    y = frame.y_true.to_numpy(dtype=int)
    score = frame.pathological_score.to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0, 1}) or (y == 0).sum() != lock["expected_normal_pairs"] or (y == 1).sum() != lock["expected_pathological_pairs"]:
        raise RuntimeError("Frozen labels differ from locked counts")
    if frame[["edf", "channel"]].duplicated().any() or frame.patient.isna().any():
        raise RuntimeError("Missing patient cluster or duplicate EDF-channel")
    tol = float(lock["comparison_absolute_tolerance"])
    if abs(roc_auc_score(y, score) - lock["expected_auroc"]) > tol:
        raise RuntimeError("Frozen score AUROC differs from original")

    sweep = make_sweep(y, score)
    youden = float(lock["expected_youden_threshold"])
    you = row_for(sweep, youden, tol)
    half = row_for(sweep, 0.5, tol)
    if abs(float(you.macro_f1) - lock["expected_youden_macro_f1"]) > tol or abs(float(half.macro_f1) - lock["expected_half_macro_f1"]) > tol:
        raise RuntimeError("Frozen operating points differ from original")
    oracle_idx = int(np.argmax(sweep.macro_f1.to_numpy()))
    oracle = sweep.iloc[oracle_idx]
    optimum = float(oracle.macro_f1)
    f1_ties = int(np.sum(np.abs(sweep.macro_f1.to_numpy() - optimum) <= 1e-15))
    robust = {
        "f1_max_posthoc_diagnostic_only": optimum,
        "f1_oracle_threshold_smallest_tie": float(oracle.threshold),
        "f1_maximizer_tie_count": f1_ties,
        "youden_threshold": youden,
        "fixed_threshold": 0.5,
        "youden_to_oracle_threshold_absolute_distance": abs(youden - float(oracle.threshold)),
        "levels": {"99_percent": plateau(sweep, 0.99, optimum, youden),
                   "95_percent": plateau(sweep, 0.95, optimum, youden)},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output / "THRESHOLD_SWEEP.csv", index=False, float_format="%.17g")
    points = pd.DataFrame([you, half, oracle]).reset_index(drop=True)
    points.insert(0, "operating_point", ["official_test_youden", "fixed_0.5", "posthoc_macro_f1_oracle_diagnostic_only"])
    points.to_csv(args.output / "KEY_OPERATING_POINTS.csv", index=False, float_format="%.17g")
    (args.output / "F1_ROBUSTNESS.json").write_text(json.dumps(robust, indent=2) + "\n", encoding="utf-8")
    boot = bootstrap(frame, youden, int(lock["bootstrap_draws"]), int(lock["bootstrap_seed"]))
    boot.to_csv(args.output / "PATIENT_BOOTSTRAP_FIXED_THRESHOLDS.csv", index=False, float_format="%.17g")
    plot_all(sweep, frame, {"Youden": youden, "0.5": 0.5, "F1 oracle": float(oracle.threshold)}, args.output)

    diff = boot.macro_f1_difference_half_minus_youden.to_numpy()
    ci = np.percentile(diff, [2.5, 97.5])
    def pp(value): return f"{float(value):.6f}"
    def op(label, r):
        return f"| {label} | {pp(r.threshold)} | {pp(r.macro_f1)} | {pp(r.pathological_f1)} | {int(r.TP)} | {int(r.FP)} | {int(r.TN)} | {int(r.FN)} |"
    p99, p95 = robust["levels"]["99_percent"], robust["levels"]["95_percent"]
    fp_change = int(you.FP - half.FP)
    fn_change = int(you.FN - half.FN)
    report = f"""# Frozen Omni CNN threshold and F1 audit (seed 42)

The frozen CNN reproduces channel AUROC **{pp(lock['expected_auroc'])}**,
close to the published 0.8061, independently of any threshold. Its
official-style test-derived Youden Macro-F1 is **{pp(you.macro_f1)}**,
below the published 0.6469, whereas fixed 0.5 yields **{pp(half.macro_f1)}**.
This is a **post-hoc, exploratory repeated-test audit**, not a new blind
benchmark or a reversal of `RAW_ENCODER_REPRODUCTION_FAILED`.

| Operating point | Threshold | Macro-F1 | Pathological F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
{op('Official test-Youden', you)}
{op('Fixed 0.5', half)}
{op('Macro-F1 oracle (diagnostic only)', oracle)}

The post-hoc maximum Macro-F1 is **{pp(optimum)}** at threshold
**{pp(oracle.threshold)}** (smallest threshold among {f1_ties} tied
maximizers). The Youden-to-oracle threshold separation is
**{pp(abs(youden - float(oracle.threshold)))}**. No oracle threshold was used
for formal performance or inside the primary bootstrap.

Youden maximizes sensitivity + specificity - 1, not Macro-F1 or precision.
At Youden versus fixed 0.5, FP changes by **{fp_change:+d}** and FN by
**{fn_change:+d}**. Thus the confusion-matrix tradeoff, in the presence of
{lock['expected_pathological_pairs']} pathological and
{lock['expected_normal_pairs']} normal pairs, explains the F1 difference.

The 99% F1-max candidate range is [{pp(p99['lower_threshold'])},
{pp(p99['upper_threshold'])}] (width {pp(p99['hull_width'])}); the 95%
range is [{pp(p95['lower_threshold'])}, {pp(p95['upper_threshold'])}]
(width {pp(p95['hull_width'])}). Fixed 0.5 qualifies for the 99% range:
**{p99['half_candidate_qualifies']}**; Youden qualifies: **{p99['youden_candidate_qualifies']}**.
See `F1_ROBUSTNESS.json` for every qualifying candidate and its nearest-score
distance; range hulls may include gaps.

Patient-cluster bootstrap used {lock['bootstrap_draws']:,} seed-{lock['bootstrap_seed']}
resamples of patients, including all their EDF-channel pairs, with both
thresholds held numerically fixed. The paired Macro-F1 difference
`F1(0.5) - F1(Youden)` is **{pp(float(half.macro_f1-you.macro_f1))}** on
the full cohort, with bootstrap percentile 95% CI
**[{pp(ci[0])}, {pp(ci[1])}]** and Pr(diff > 0) **{pp(np.mean(diff > 0))}**.
The full fixed-threshold distributions for Macro-F1, pathological F1,
sensitivity, and specificity are in
`PATIENT_BOOTSTRAP_FIXED_THRESHOLDS.csv`.

The optional time-resolved ictal audit was **not run**: no compatible frozen
time-bin scores were established for this CNN with the same channel labels,
and constructing them would be a new experiment. No per-time-bin threshold
was selected.

All 8,104 `(EDF, channel)` predictions and the 240,074 segment probabilities
remain **private on the original server**. Replayed scores matched the prior
aggregate AUROC, both Macro-F1s and Youden threshold to the locked tolerance
before the score table was persisted. No checkpoint, waveform cache, labels,
or predictions were modified. The frozen table SHA-256 is
`{replay['channel_predictions_sha256']}`.

The evidence supports a specific operating-point explanation for the F1
gap in this frozen model, **not** a claim that the published Youden Macro-F1
was reproduced or that A1-NET's hard gate passed. Published-code extraction
frequency ambiguity remains a separate provenance limitation. The user's
question 13 was truncated in the supplied message; no additional unprovided
claim is inferred here.

## Direct answers to the supplied questions

1. **AUROC independently reproduced?** Yes: {pp(lock['expected_auroc'])}
   versus published 0.8061; AUROC uses score ranking, not a fixed threshold.
2. **Reproduced Youden threshold:** {pp(youden)}.
3. **Macro-F1 at Youden:** {pp(you.macro_f1)}.
4. **Macro-F1 at 0.5:** {pp(half.macro_f1)}.
5. **Post-hoc maximum Macro-F1:** {pp(optimum)}; diagnostic only.
6. **Macro-F1-maximizing threshold:** {pp(oracle.threshold)} (smallest tied candidate).
7. **Threshold separation:** {pp(abs(youden - float(oracle.threshold)))}.
8. **Why Youden selects that point:** Its J is {pp(you.youden_j)}, versus
   {pp(half.youden_j)} at 0.5; the rule rewards sensitivity and specificity
   equally, irrespective of precision and class imbalance.
9. **Error terms behind its F1 loss versus 0.5:** FP {int(you.FP)} versus
   {int(half.FP)} ({fp_change:+d}); FN {int(you.FN)} versus {int(half.FN)}
   ({fn_change:+d}). The full TP/TN counts are in the table above.
10. **Is 0.5 inside the high-F1 region?** 99%: {p99['half_candidate_qualifies']};
    95%: {p95['half_candidate_qualifies']} (candidate-level membership).
11. **Is Youden outside it?** 99% outside: {not p99['youden_candidate_qualifies']};
    95% outside: {not p95['youden_candidate_qualifies']}.
12. **Is the 0.5-versus-Youden difference stable across patients?**
    Paired 95% interval [{pp(ci[0])}, {pp(ci[1])}]; strictly above zero:
    {bool(ci[0] > 0)}. This is a fixed-threshold resampling audit, not an
    independent test set.
13. The supplied question 13 ends after an opening backtick, so its proposed
    assertion cannot be evaluated without inventing text.
"""
    (args.output.parent / "FINAL_AUDIT.md").write_text(report, encoding="utf-8")
    public_audit = {
        "status": "PASS", "source_prediction_sha256": replay["channel_predictions_sha256"],
        "source_result_sha256": replay["source_result_sha256"],
        "n_patients": replay["patient_clusters"], "n_labeled_pairs": len(frame),
        "n_thresholds": len(sweep), "n_bootstrap_draws": len(boot),
        "youden_macro_f1": float(you.macro_f1), "half_macro_f1": float(half.macro_f1),
        "posthoc_oracle_macro_f1_diagnostic_only": optimum,
        "paired_difference_ci95": ci.tolist(),
        "prior_hard_gate_changed": False,
    }
    (args.output / "AUDIT_VALIDATION.json").write_text(json.dumps(public_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(public_audit), flush=True)


if __name__ == "__main__":
    main()
