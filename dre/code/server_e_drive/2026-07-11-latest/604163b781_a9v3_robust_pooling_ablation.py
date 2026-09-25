"""Diagnostic-only robust pooling feasibility and low-cost ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

A9V3 = {
    "patient_macro_f1": 0.642714,
    "patient_macro_ez_f1": 0.470591,
    "patient_macro_auprc_ez": 0.518635,
    "patient_macro_ez_mrr": 0.715136,
    "top1_is_ez_rate": 0.600000,
}
ARTIFACT_PATTERNS = [
    "*record*channel*prediction*.csv",
    "*window*channel*prediction*.csv",
    "*channel_predictions*.csv",
    "test_channel_predictions_neuroez_v2_fold_*.csv",
    "val_channel_predictions_neuroez_v2_fold_*.csv",
]


def passes_main_gate(row: pd.Series | dict[str, float]) -> bool:
    return (
        float(row.get("patient_macro_f1", 0.0)) > A9V3["patient_macro_f1"]
        and float(row.get("patient_macro_ez_f1", 0.0)) > A9V3["patient_macro_ez_f1"]
        and float(row.get("patient_macro_auprc_ez", 0.0)) > A9V3["patient_macro_auprc_ez"]
        and float(row.get("patient_macro_ez_mrr", 0.0)) >= A9V3["patient_macro_ez_mrr"]
        and float(row.get("top1_is_ez_rate", 0.0)) >= A9V3["top1_is_ez_rate"]
    )


def detect_artifact_granularity(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    artifacts = []
    seen = set()
    for pattern in ARTIFACT_PATTERNS:
        for path in sorted(root.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            lower = path.name.lower()
            if "window" in lower:
                gran = "window"
            elif "record" in lower:
                gran = "record"
            elif "channel_predictions" in lower or "channel_prediction" in lower:
                gran = "patient"
            else:
                gran = "unknown"
            artifacts.append({"path": str(path), "artifact_type": gran})
    types = {item["artifact_type"] for item in artifacts}
    if "window" in types:
        granularity = "window_level_available"
    elif "record" in types:
        granularity = "record_level_available"
    elif "patient" in types:
        granularity = "patient_level_only"
    else:
        granularity = "unknown"
    return {"granularity": granularity, "artifacts": artifacts}


def _score_column(df: pd.DataFrame) -> str:
    for col in ("score", "score_eval", "score_ez", "score_ez_probability"):
        if col in df.columns:
            return col
    raise RuntimeError("No usable score column found for pooling ablation.")


def _pool(values: Sequence[float], mode: str, *, p: float = 0.10, tau: float = 0.10) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if mode == "mean":
        return float(arr.mean())
    if mode == "median":
        return float(np.median(arr))
    if mode == "top_p_mean":
        k = max(1, int(np.ceil(arr.size * p)))
        return float(np.sort(arr)[-k:].mean())
    if mode == "logsumexp":
        scaled = arr / max(float(tau), 1e-6)
        return float(max(float(tau), 1e-6) * (np.log(np.exp(scaled - scaled.max()).mean()) + scaled.max()))
    if mode == "noisy_or":
        clipped = np.clip(arr, 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - clipped))
    if mode == "hybrid_top30_median":
        return float(0.7 * _pool(arr, "top_p_mean", p=0.30) + 0.3 * _pool(arr, "median"))
    raise ValueError(f"Unknown pooling mode: {mode}")


def _select_topk(scores: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    if scores.size == 0:
        return pred
    k = max(1, min(int(k), int(scores.size)))
    pred[np.argsort(scores)[::-1][:k]] = True
    return pred


def _mrr(y_ez: np.ndarray, scores: np.ndarray) -> float:
    if y_ez.size == 0 or int(y_ez.sum()) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    positive = np.where(y_ez[order] == 1)[0]
    return float(1.0 / float(positive[0] + 1)) if positive.size else 0.0


def _patient_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, group in rows.groupby(["temporal_pooling", "record_pooling", "fold_idx", "subject_id"], dropna=False, sort=False):
        temporal, record, fold_idx, subject_id = keys
        y_ez = group["true_ez"].astype(int).to_numpy()
        y_nez = 1 - y_ez
        scores = group["score"].astype(float).to_numpy()
        pred_ez = _select_topk(scores, int(y_ez.sum()))
        pred_nez = (~pred_ez).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
        out.append(
            {
                "temporal_pooling": temporal,
                "record_pooling": record,
                "fold_idx": int(fold_idx),
                "subject_id": subject_id,
                "center": str(group["center"].iloc[0]),
                "patient_macro_f1": float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)),
                "patient_macro_ez_f1": float(f1[1]),
                "patient_macro_auprc_ez": float(average_precision_score(y_ez, scores)) if np.unique(y_ez).size > 1 else 0.0,
                "patient_macro_ez_mrr": _mrr(y_ez, scores),
                "top1_is_ez": float(y_ez[int(np.argmax(scores))] == 1) if scores.size else 0.0,
            }
        )
    return pd.DataFrame(out)


def _summarize(patient_rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if patient_rows.empty:
        return pd.DataFrame()
    out = (
        patient_rows.groupby(group_cols, dropna=False, sort=True)[
            ["patient_macro_f1", "patient_macro_ez_f1", "patient_macro_auprc_ez", "patient_macro_ez_mrr", "top1_is_ez"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"top1_is_ez": "top1_is_ez_rate"})
    )
    out["n_patients"] = patient_rows.groupby(group_cols, dropna=False, sort=True)["subject_id"].nunique().to_numpy()
    out["delta_f1_vs_a9v3"] = out["patient_macro_f1"] - A9V3["patient_macro_f1"]
    out["delta_ez_f1_vs_a9v3"] = out["patient_macro_ez_f1"] - A9V3["patient_macro_ez_f1"]
    out["delta_auprc_vs_a9v3"] = out["patient_macro_auprc_ez"] - A9V3["patient_macro_auprc_ez"]
    out["delta_mrr_vs_a9v3"] = out["patient_macro_ez_mrr"] - A9V3["patient_macro_ez_mrr"]
    out["delta_top1_vs_a9v3"] = out["top1_is_ez_rate"] - A9V3["top1_is_ez_rate"]
    out["passes_main_gate"] = out.apply(passes_main_gate, axis=1)
    return out


def _load_best_granularity_frame(run_dir: Path, granularity: str) -> pd.DataFrame:
    if granularity == "window_level_available":
        patterns = ["*window*channel*prediction*.csv"]
    elif granularity == "record_level_available":
        patterns = ["*record*channel*prediction*.csv"]
    else:
        raise RuntimeError(f"Cannot run true pooling ablation for granularity={granularity}")
    frames = []
    seen = set()
    for pattern in patterns:
        for path in sorted(run_dir.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No {granularity} artifacts found under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _ablate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    score_col = _score_column(df)
    work = df.copy()
    work["score"] = work[score_col].astype(float)
    if "record_id" not in work.columns:
        work["record_id"] = work.get("run_id", "record0")
    if "window_id" not in work.columns:
        work["window_id"] = np.arange(len(work))
    temporal_modes = ["mean", "top_p_mean", "logsumexp", "noisy_or"] if granularity == "window_level_available" else ["mean"]
    record_modes = ["mean", "median", "top_p_mean", "hybrid_top30_median", "noisy_or"]
    patient_channel_rows = []
    for temporal_mode in temporal_modes:
        if granularity == "window_level_available":
            record_rows = []
            for keys, group in work.groupby(["fold_idx", "subject_id", "center", "record_id", "channel_name", "true_ez"], dropna=False, sort=False):
                fold_idx, subject_id, center, record_id, channel_name, true_ez = keys
                record_rows.append(
                    {
                        "fold_idx": fold_idx,
                        "subject_id": subject_id,
                        "center": center,
                        "record_id": record_id,
                        "channel_name": channel_name,
                        "true_ez": true_ez,
                        "score": _pool(group["score"], temporal_mode, p=0.10, tau=0.10),
                    }
                )
            record_df = pd.DataFrame(record_rows)
        else:
            record_df = work[["fold_idx", "subject_id", "center", "record_id", "channel_name", "true_ez", "score"]].copy()
        for record_mode in record_modes:
            for keys, group in record_df.groupby(["fold_idx", "subject_id", "center", "channel_name", "true_ez"], dropna=False, sort=False):
                fold_idx, subject_id, center, channel_name, true_ez = keys
                patient_channel_rows.append(
                    {
                        "temporal_pooling": temporal_mode,
                        "record_pooling": record_mode,
                        "fold_idx": fold_idx,
                        "subject_id": subject_id,
                        "center": center,
                        "channel_name": channel_name,
                        "true_ez": true_ez,
                        "score": _pool(group["score"], record_mode, p=0.30, tau=0.10),
                    }
                )
    return pd.DataFrame(patient_channel_rows)


def _write_feasibility_only(output_dir: Path, discovery: dict[str, object], window_cache_path: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(discovery["artifacts"]).to_csv(output_dir / "pooling_ablation_available_artifacts.csv", index=False)
    report = {
        "granularity": discovery["granularity"],
        "window_cache_path": str(window_cache_path),
        "window_cache_exists": Path(window_cache_path).exists(),
        "true_ablation_possible": False,
        "reason": "Already pooled patient-level scores cannot support a real robust pooling ablation.",
    }
    (output_dir / "pooling_ablation_feasibility_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    recommendation = (
        "A true robust pooling ablation cannot be performed from already pooled patient-level scores. "
        "To evaluate robust pooling, future runs must save record/window-level per-channel logits before patient pooling."
    )
    (output_dir / "pooling_ablation_recommendation.txt").write_text(recommendation, encoding="utf-8")
    return report


def run_pooling_ablation(
    run_dir: str | Path,
    *,
    window_cache_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    root = Path(run_dir)
    out = Path(output_dir)
    discovery = detect_artifact_granularity(root)
    granularity = str(discovery["granularity"])
    if granularity not in {"record_level_available", "window_level_available"}:
        report = _write_feasibility_only(out, discovery, Path(window_cache_path))
        return {"granularity": granularity, "report": report}

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(discovery["artifacts"]).to_csv(out / "pooling_ablation_available_artifacts.csv", index=False)
    df = _load_best_granularity_frame(root, granularity)
    pooled = _ablate(df, granularity)
    patient_rows = _patient_metrics(pooled)
    summary = _summarize(patient_rows, ["temporal_pooling", "record_pooling"])
    by_fold = _summarize(patient_rows, ["temporal_pooling", "record_pooling", "fold_idx"])
    by_center = _summarize(patient_rows, ["temporal_pooling", "record_pooling", "center"])
    summary.to_csv(out / "pooling_ablation_summary.csv", index=False)
    by_fold.to_csv(out / "pooling_ablation_by_fold.csv", index=False)
    by_center.to_csv(out / "pooling_ablation_by_center.csv", index=False)
    patient_rows.to_csv(out / "pooling_ablation_patient_rows.csv", index=False)
    report = {
        "granularity": granularity,
        "window_cache_path": str(window_cache_path),
        "window_cache_exists": Path(window_cache_path).exists(),
        "true_ablation_possible": True,
        "diagnostic_only": True,
    }
    (out / "pooling_ablation_feasibility_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "pooling_ablation_recommendation.txt").write_text(
        "This is a diagnostic ablation only. Do not select a production pooling method from test labels; "
        "use it to decide whether future runs should save and validate record/window-level logits.",
        encoding="utf-8",
    )
    return {"granularity": granularity, "summary": summary, "by_fold": by_fold, "by_center": by_center}


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust pooling feasibility/ablation diagnostic without training.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--window_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = run_pooling_ablation(args.run_dir, window_cache_path=args.window_cache_path, output_dir=args.output_dir)
    print(f"Granularity: {result['granularity']}")
    if result["granularity"] == "patient_level_only":
        print("True pooling ablation is not possible from patient-level scores.")
    else:
        print("Wrote diagnostic pooling ablation outputs. Treat them as ablation only, not method selection.")


if __name__ == "__main__":
    main()
