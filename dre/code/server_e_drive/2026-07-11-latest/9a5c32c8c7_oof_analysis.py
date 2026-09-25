"""Strict patient-level analyses over completed Task 1 OOF predictions.

This module deliberately consumes only frozen test ledgers and their already
selected validation thresholds.  It has no training, calibration fitting, or
threshold-selection path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


MODELS = {"PRQ-Net": "p2_score_nez", "BCR-Net": "v3_score_nez", "CDEL": "fused_score_nez"}
KEYS = ["seed", "outer_fold", "subject_id", "center", "channel_key"]


def _require(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _patient_ranking(group: pd.DataFrame, score: str) -> dict[str, float]:
    label_ez = 1 - group.label_nez.to_numpy(int)
    probability_ez = 1.0 - group[score].to_numpy(float)
    order = np.argsort(-probability_ez, kind="mergesort")
    ranked = label_ez[order]
    positives = int(label_ez.sum())
    if positives < 1:
        raise ValueError(f"Patient {group.subject_id.iloc[0]!r} has no EZ channel")
    discount = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    ideal = float((np.sort(label_ez)[::-1] * discount).sum())
    top = ranked[:positives]
    return {
        "patient_ez_auprc": float(average_precision_score(label_ez, probability_ez)),
        "NDCG_EZ": float((ranked * discount).sum() / ideal),
        "MRR_EZ": float(1.0 / (np.flatnonzero(ranked == 1)[0] + 1)),
        "Recall_at_true_EZ_count": float(top.sum() / positives),
        "Precision_at_true_EZ_count": float(top.mean()),
        "Top1_is_EZ": float(ranked[0] == 1),
    }


def _ece(label_nez: np.ndarray, score_nez: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (score_nez >= low) & ((score_nez < high) if high < 1 else (score_nez <= high))
        if mask.any():
            value += float(mask.mean() * abs(score_nez[mask].mean() - label_nez[mask].mean()))
    return value


def _patient_rows(ledger: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    values: list[dict[str, object]] = []
    threshold_map = thresholds.set_index(["seed", "outer_fold", "model"])["threshold"]
    for model, score in MODELS.items():
        for key, patient in ledger.groupby(["seed", "outer_fold", "subject_id", "center"], sort=True):
            seed, fold, subject, center = key
            try:
                threshold = float(threshold_map.loc[(seed, fold, model)])
            except KeyError as exc:
                raise ValueError(f"No validation threshold for seed={seed}, fold={fold}, model={model}") from exc
            label = patient.label_nez.to_numpy(int)
            probability = patient[score].to_numpy(float)
            predicted = (probability >= threshold).astype(int)
            ranking = _patient_ranking(patient, score)
            values.append({
                "seed": int(seed), "outer_fold": int(fold), "subject_id": str(subject), "center": str(center),
                "model": model, "selected_threshold": threshold, "n_channels": int(len(patient)),
                "patient_macro_f1": float(f1_score(label, predicted, average="macro", labels=[0, 1], zero_division=0)),
                "patient_ez_f1": float(f1_score(label, predicted, pos_label=0, zero_division=0)),
                "patient_nez_f1": float(f1_score(label, predicted, pos_label=1, zero_division=0)),
                "patient_brier_nez": float(np.mean((probability - label) ** 2)),
                "patient_ece_nez": _ece(label, probability),
                **ranking,
            })
    return pd.DataFrame(values)


def _bootstrap(patient: pd.DataFrame, repeats: int, rng_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = ["patient_macro_f1", "patient_ez_f1", "patient_ez_auprc", "NDCG_EZ"]
    means = patient.groupby(["subject_id", "model"], as_index=False)[metric_columns].mean()
    samples: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    rng = np.random.default_rng(rng_seed)
    for reference in ("PRQ-Net", "BCR-Net"):
        for metric in metric_columns:
            pivot = means.pivot(index="subject_id", columns="model", values=metric)
            if not {"CDEL", reference}.issubset(pivot.columns) or pivot[["CDEL", reference]].isna().any().any():
                raise ValueError(f"Paired bootstrap alignment failed for {reference}/{metric}")
            delta = (pivot["CDEL"] - pivot[reference]).to_numpy(float)
            draws = delta[rng.integers(0, len(delta), size=(repeats, len(delta)))].mean(axis=1)
            comparison = f"CDEL-{reference}"
            samples.extend({"comparison": comparison, "metric": metric, "replicate": index, "mean_delta": float(value)} for index, value in enumerate(draws))
            summary.append({
                "comparison": comparison, "metric": metric, "metric_unit": "patient (three training seeds averaged first)",
                "n_patients": int(len(delta)), "observed_mean_delta": float(delta.mean()), "bootstrap_mean": float(draws.mean()),
                "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                "p_value_two_sided": float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
                "bootstrap_samples": int(repeats), "bootstrap_rng_seed": int(rng_seed),
            })
    return pd.DataFrame(samples), pd.DataFrame(summary)


def _boundary(ledger: pd.DataFrame, thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_map = thresholds.set_index(["seed", "outer_fold", "model"])["threshold"]
    pieces: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for (seed, fold, subject), group in ledger.groupby(["seed", "outer_fold", "subject_id"], sort=True):
        threshold = float(threshold_map.loc[(seed, fold, "PRQ-Net")])
        ordered = group.assign(prq_margin=(group.p2_score_nez - threshold).abs()).sort_values("prq_margin", kind="mergesort")
        for fraction in (.10, .20, .30):
            hard = ordered.head(max(1, int(np.ceil(len(ordered) * fraction)))).copy()
            hard["hard_fraction"] = fraction
            hard["prq_threshold"] = threshold
            pieces.append(hard)
            for model, score in MODELS.items():
                model_threshold = float(threshold_map.loc[(seed, fold, model)])
                predicted = (hard[score].to_numpy(float) >= model_threshold).astype(int)
                label = hard.label_nez.to_numpy(int)
                rows.append({"seed": seed, "outer_fold": fold, "subject_id": subject, "hard_fraction": fraction, "model": model,
                             "hard_channel_ez_f1": float(f1_score(label, predicted, pos_label=0, zero_division=0)),
                             "hard_channel_macro_f1": float(f1_score(label, predicted, labels=[0, 1], average="macro", zero_division=0)),
                             "n_hard_channels": len(hard), "selected_validation_threshold": model_threshold})
    channels = pd.concat(pieces, ignore_index=True)
    summary = pd.DataFrame(rows).groupby(["hard_fraction", "model"], as_index=False).mean(numeric_only=True)
    return channels, summary


def _validate(ledger: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    _require(ledger, ["seed", "outer_fold", "subject_id", "center", "channel_key", "label_nez", *MODELS.values()], "OOF ledger")
    _require(thresholds, ["seed", "outer_fold", "model", "threshold", "threshold_source"], "threshold ledger")
    if ledger.duplicated(KEYS).any():
        raise ValueError("OOF ledger contains duplicate seed/fold/patient/channel keys")
    if not ledger.label_nez.isin([0, 1]).all():
        raise ValueError("Label semantics must be EZ=0, NEZ=1")
    if not ledger.groupby(["seed", "subject_id"]).label_nez.apply(lambda x: (x == 0).any() and (x == 1).any()).all():
        raise ValueError("Each patient must have at least one EZ and one NEZ channel")
    if not thresholds.threshold_source.astype(str).eq("validation_only").all():
        raise ValueError("Every selected threshold must be sourced from validation_only")
    if thresholds.duplicated(["seed", "outer_fold", "model"]).any():
        raise ValueError("Threshold ledger contains duplicate seed/fold/model rows")
    values = ledger[list(MODELS.values())].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("All OOF probabilities must be finite values in [0, 1]")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without requiring pandas' tabulate extra."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_figures(bootstrap: pd.DataFrame, hard_summary: pd.DataFrame, figures: Path) -> None:
    """Create compact, deterministic paper figures from frozen-result summaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forest = bootstrap.copy()
    forest["label"] = forest["comparison"] + " / " + forest["metric"]
    forest = forest.iloc[::-1].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(8, max(3.0, .42 * len(forest))))
    axis.axvline(0.0, color="black", linewidth=.8)
    for index, row in forest.iterrows():
        axis.plot([row.ci_low, row.ci_high], [index, index], color="#3174a7", linewidth=1.5)
        axis.scatter(row.observed_mean_delta, index, color="#d95f02", zorder=3)
    axis.set_yticks(range(len(forest)), forest.label)
    axis.set_xlabel("CDEL − reference (patient-level metric)")
    fig.tight_layout(); fig.savefig(figures / "paired_bootstrap_forest.png", dpi=220); fig.savefig(figures / "paired_bootstrap_forest.pdf"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 3.8))
    for model, group in hard_summary.groupby("model", sort=False):
        axis.plot(group.hard_fraction * 100, group.hard_channel_ez_f1, marker="o", label=model)
    axis.set_xlabel("PRQ-defined hard channels (%)"); axis.set_ylabel("EZ F1"); axis.set_xticks([10, 20, 30]); axis.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures / "boundary_hard_case.png", dpi=220); fig.savefig(figures / "boundary_hard_case.pdf"); plt.close(fig)


def analyze_oof_ledgers(*, ledger: pd.DataFrame, thresholds: pd.DataFrame, output_dir: str | Path, bootstrap_samples: int = 10000, bootstrap_seed: int = 20260724) -> dict[str, object]:
    """Run all no-retraining AAAI OOF analyses and write new result artifacts."""
    _validate(ledger, thresholds)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    for name in ("audit", "oof_analysis", "tables", "reports", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    patients = _patient_rows(ledger, thresholds)
    ranking = patients.groupby("model", as_index=False)[["patient_ez_auprc", "NDCG_EZ", "MRR_EZ", "Recall_at_true_EZ_count", "Precision_at_true_EZ_count", "Top1_is_EZ"]].mean()
    center = patients.groupby(["center", "model"], as_index=False).mean(numeric_only=True)
    calibration = patients.groupby("model", as_index=False)[["patient_brier_nez", "patient_ece_nez"]].mean()
    samples, bootstrap = _bootstrap(patients, bootstrap_samples, bootstrap_seed)
    hard_channels, hard_summary = _boundary(ledger, thresholds)
    patients.to_csv(output / "oof_analysis" / "patient_metrics_long.csv", index=False)
    samples.to_csv(output / "oof_analysis" / "paired_bootstrap_samples.csv", index=False)
    hard_channels.to_csv(output / "oof_analysis" / "boundary_hard_channels.csv", index=False)
    ranking.to_csv(output / "tables" / "task1_ranking_metrics.csv", index=False)
    bootstrap.to_csv(output / "tables" / "paired_bootstrap_summary.csv", index=False)
    hard_summary.to_csv(output / "tables" / "boundary_hard_case_summary.csv", index=False)
    center.to_csv(output / "tables" / "centerwise_comparison.csv", index=False)
    calibration.to_csv(output / "tables" / "calibration_patient_equal.csv", index=False)
    (output / "tables" / "task1_ranking_metrics.tex").write_text(ranking.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    (output / "tables" / "paired_bootstrap_summary.tex").write_text(bootstrap.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    (output / "tables" / "boundary_hard_case_summary.tex").write_text(hard_summary.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    _write_figures(bootstrap, hard_summary, output / "figures")
    audit = {"status": "passed", "label_semantics": {"EZ": 0, "NEZ": 1}, "fusion": "0.80*PRQ + 0.20*BCR", "threshold_source": "validation_only", "n_seeds": int(ledger.seed.nunique()), "n_patients": int(ledger.subject_id.nunique()), "n_channels": int(len(ledger)), "bootstrap_unit": "subject_id", "bootstrap_samples": int(bootstrap_samples)}
    (output / "audit" / "oof_analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "reports" / "task1_oof_analysis.md").write_text("# Task 1 frozen-OOF analysis\n\nAll inference, thresholds, and model weights were fixed before this analysis. Bootstrap resampling unit: patient.\n\n" + _markdown_table(ranking) + "\n", encoding="utf-8")
    return audit
