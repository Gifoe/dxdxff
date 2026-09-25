from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline_common.reporting import frame_to_markdown
from .metrics import compute_task1_metrics


def _patient_macro_scores(table: pd.DataFrame) -> pd.Series:
    return table.groupby("subject_id").apply(
        lambda group: f1_score(
            group["label_nez"].to_numpy(dtype=int), group["predicted_nez"].to_numpy(dtype=int),
            average="macro", labels=[0, 1], zero_division=0,
        ),
        include_groups=False,
    )


def _normalize_v3_reference(reference: pd.DataFrame) -> pd.DataFrame | None:
    channel = next((name for name in ("channel_name", "channel_name_norm", "contact_name") if name in reference), None)
    label = next((name for name in ("label_nez", "true_nez") if name in reference), None)
    predicted = next((name for name in ("predicted_nez",) if name in reference), None)
    if label is None and "true_ez" in reference:
        reference = reference.assign(label_nez=1 - pd.to_numeric(reference["true_ez"], errors="raise").astype(int))
        label = "label_nez"
    if predicted is None and "predicted_ez" in reference:
        reference = reference.assign(predicted_nez=1 - pd.to_numeric(reference["predicted_ez"], errors="raise").astype(int))
        predicted = "predicted_nez"
    if channel is None or label is None or predicted is None or "subject_id" not in reference:
        return None
    output = reference[["subject_id", channel, label, predicted]].copy()
    output.columns = ["subject_id", "channel_name", "label_nez", "predicted_nez"]
    output["subject_id"] = output["subject_id"].astype(str)
    output["channel_name"] = output["channel_name"].astype(str).str.upper().str.replace(" ", "", regex=False)
    return output


def _task1_bootstrap(oof: pd.DataFrame, reference: pd.DataFrame | None, samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    normalized_reference = _normalize_v3_reference(reference) if reference is not None else None
    reference_scores = _patient_macro_scores(normalized_reference) if normalized_reference is not None else None
    rows = []
    for (model, seed), group in oof.groupby(["model", "seed"], sort=False):
        scores = _patient_macro_scores(group)
        draws = [float(scores.iloc[rng.integers(0, len(scores), len(scores))].mean()) for _ in range(int(samples))]
        lower, upper = np.percentile(draws, [2.5, 97.5])
        row = {
            "model": model, "seed": int(seed), "metric": "patient_macro_f1", "estimate": float(scores.mean()),
            "ci_lower": float(lower), "ci_upper": float(upper), "bootstrap_samples": int(samples),
            "paired_reference": "Frozen V3", "paired_status": "unavailable", "paired_delta": np.nan,
            "paired_ci_lower": np.nan, "paired_ci_upper": np.nan,
        }
        if reference_scores is not None and set(scores.index) == set(reference_scores.index):
            aligned = pd.concat([scores.rename("model"), reference_scores.rename("reference")], axis=1).dropna()
            differences = aligned["model"] - aligned["reference"]
            delta_draws = [float(differences.iloc[rng.integers(0, len(differences), len(differences))].mean()) for _ in range(int(samples))]
            paired_lower, paired_upper = np.percentile(delta_draws, [2.5, 97.5])
            row.update({
                "paired_status": "evaluated", "paired_delta": float(differences.mean()),
                "paired_ci_lower": float(paired_lower), "paired_ci_upper": float(paired_upper),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_task1(
    oof: pd.DataFrame,
    output_dir: str | Path,
    *,
    failures: pd.DataFrame | None = None,
    v3_reference: pd.DataFrame | None = None,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    reports_dir = output / "reports"
    comparison_dir = output / "comparison"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for (model, seed), group in oof.groupby(["model", "seed"], sort=False):
        rows.append({"model": model, "seed": int(seed), **compute_task1_metrics(group)})
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(metrics_dir / "task1_by_seed.csv", index=False)
    numeric = [column for column in by_seed if column not in {"model", "seed"}]
    mean_std = by_seed.groupby("model", as_index=False)[numeric].agg(["mean", "std"])
    mean_std.columns = ["model" if column[0] == "model" else f"{column[0]}_{column[1]}" for column in mean_std.columns]
    mean_std.to_csv(metrics_dir / "task1_mean_std.csv", index=False)
    fold_rows = [{"model": model, "seed": seed, "outer_fold": fold, **compute_task1_metrics(group)} for (model, seed, fold), group in oof.groupby(["model", "seed", "outer_fold"])]
    center_rows = [{"model": model, "seed": seed, "center": center, **compute_task1_metrics(group)} for (model, seed, center), group in oof.groupby(["model", "seed", "center"])]
    pd.DataFrame(fold_rows).to_csv(metrics_dir / "task1_by_fold.csv", index=False)
    pd.DataFrame(center_rows).to_csv(metrics_dir / "task1_by_center.csv", index=False)
    bootstrap = _task1_bootstrap(oof, v3_reference, bootstrap_samples)
    bootstrap.to_csv(comparison_dir / "task1_bootstrap_vs_v3.csv", index=False)
    failure_count = 0 if failures is None else len(failures)
    report = [
        "# Task 1 Baseline Report",
        "",
        "Task: channel-level EZ/NEZ localization on the frozen old-90 V3 patient folds.",
        "",
        "Label direction: NEZ=1, EZ=0. Primary metric: patient_macro_f1.",
        "",
        f"Completed model/seed rows: {len(by_seed)}. Recorded failures: {failure_count}.",
        "",
        f"Patient bootstrap samples: {int(bootstrap_samples)}.",
        "",
        frame_to_markdown(by_seed),
    ]
    (reports_dir / "TASK1_BASELINE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return by_seed


__all__ = ["frame_to_markdown", "summarize_task1"]
