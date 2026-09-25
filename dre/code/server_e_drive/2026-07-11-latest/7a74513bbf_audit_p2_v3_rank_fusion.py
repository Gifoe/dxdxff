"""Audit legal OOF P2/V3 channel-score rank complementarity.

This module is deliberately diagnostic-only.  It never trains, searches a
continuous fusion weight, or uses a test threshold.  All reported
classification metrics use the per-patient true EZ count and are therefore
explicitly marked as non-deployable diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score


ANALYSIS_STATUS = "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
TRUE_COUNT_USED = True
FORMAL_PREDICTION = False
CANDIDATES = ("P2", "V3", "RankAvg_50_50", "RankAvg_P2_70", "RankAvg_V3_70", "RRF_k60")
FUSIONS = CANDIDATES[2:]


def normalize_channel_name(value: object) -> str:
    text = str(value).strip().upper().replace("’", "'").replace("‘", "'")
    text = text.replace("′", "'").replace("＇", "'")
    return re.sub(r"\s+", "", text)


def _first_column(df: pd.DataFrame, names: Iterable[str], *, required: bool = True) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"missing required column; tried {list(names)}")
    return None


def _fold_column(df: pd.DataFrame) -> str:
    return _first_column(df, ("outer_fold", "fold", "fold_idx"))  # type: ignore[return-value]


def _subject_column(df: pd.DataFrame) -> str:
    return _first_column(df, ("subject_id", "patient_id", "patient"))  # type: ignore[return-value]


def _channel_column(df: pd.DataFrame) -> str:
    return _first_column(df, ("channel_name", "channel", "channel_id"))  # type: ignore[return-value]


def _label_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    subject_col, channel_col, fold_col = _subject_column(df), _channel_column(df), _fold_column(df)
    ez_col = _first_column(df, ("label_ez", "ez_label", "true_ez"), required=False)
    nez_col = _first_column(df, ("label_nez", "nez_label"), required=False)
    if ez_col is None and nez_col is None:
        raise ValueError(f"{name} ledger must contain label_ez or label_nez")
    def normalize_fold(value: object) -> str:
        text = str(value).strip()
        try:
            number = float(text)
            return str(int(number)) if number.is_integer() else text
        except ValueError:
            return text

    out = pd.DataFrame({
        "subject_id": df[subject_col].astype(str).str.strip(),
        "channel_key": df[channel_col].map(normalize_channel_name),
        "outer_fold": df[fold_col].map(normalize_fold),
    })
    if (out["subject_id"] == "").any() or (out["channel_key"] == "").any():
        raise ValueError(f"{name} ledger contains empty subject_id or channel_name")
    if ez_col is not None:
        out["label_ez"] = pd.to_numeric(df[ez_col], errors="raise")
    else:
        out["label_ez"] = 1 - pd.to_numeric(df[nez_col], errors="raise")
    if nez_col is not None:
        out["label_nez"] = pd.to_numeric(df[nez_col], errors="raise")
    else:
        out["label_nez"] = 1 - out["label_ez"]
    if not out["label_ez"].isin([0, 1]).all() or not out["label_nez"].isin([0, 1]).all():
        raise ValueError(f"{name} labels must be binary")
    out["label_ez"] = out["label_ez"].astype(int)
    out["label_nez"] = out["label_nez"].astype(int)
    if not (out["label_ez"] + out["label_nez"] == 1).all():
        raise ValueError(f"{name} label_ez/label_nez are inconsistent")
    out["center"] = df[_first_column(df, ("center", "center_id"), required=False)].astype(str).str.lower() if _first_column(df, ("center", "center_id"), required=False) else out["subject_id"].str.split(":", n=1).str[0].str.lower()
    return out


def _validate_unique_keys(frame: pd.DataFrame, name: str) -> None:
    keys = ["subject_id", "channel_key"]
    duplicates = frame[frame.duplicated(keys, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"{name} ledger has duplicate subject/channel keys: {duplicates[keys].head(5).to_dict('records')}")


def _score_semantics(value: str) -> str:
    value = str(value).lower().strip()
    allowed = {"ez_score", "nez_score", "ez_probability", "nez_probability"}
    if value not in allowed:
        raise ValueError(f"unsupported V3 score semantics {value!r}; use one of {sorted(allowed)}")
    return value


def _to_ez_score(values: pd.Series, semantics: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("score column contains non-finite values")
    semantics = _score_semantics(semantics)
    if semantics in {"ez_score", "ez_probability"}:
        return numeric
    if semantics == "nez_probability":
        return 1.0 - numeric
    return -numeric


def _v3_score_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"V3 score column not found: {requested}")
        return requested
    return _first_column(df, ("score_ez", "ez_score", "score_nez", "nez_score", "final_nez_logit"))  # type: ignore[return-value]


def _p2_score(df: pd.DataFrame) -> pd.Series:
    column = _first_column(df, ("direct_nez_logit", "score_nez", "nez_score", "final_nez_logit"))
    values = pd.to_numeric(df[column], errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("P2 score column contains non-finite values")
    return -values


def load_and_align(p2_path: str | Path, v3_path: str | Path, *, v3_score_column: str | None, v3_score_semantics: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p2_raw, v3_raw = pd.read_csv(p2_path), pd.read_csv(v3_path)
    p2, v3 = _label_frame(p2_raw, "P2"), _label_frame(v3_raw, "V3")
    p2["p2_ez_score"] = _p2_score(p2_raw).to_numpy()
    v3["v3_ez_score"] = _to_ez_score(v3_raw[_v3_score_column(v3_raw, v3_score_column)], v3_score_semantics).to_numpy()
    _validate_unique_keys(p2, "P2")
    _validate_unique_keys(v3, "V3")
    merged = p2.merge(v3, on=["subject_id", "channel_key"], how="outer", suffixes=("_p2", "_v3"), indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    if matched.empty:
        raise ValueError("P2 and V3 have no matched subject/channel rows")
    if not (matched["outer_fold_p2"].astype(str) == matched["outer_fold_v3"].astype(str)).all():
        raise ValueError("outer_fold mismatch on matched channels")
    if not (matched["label_ez_p2"] == matched["label_ez_v3"]).all() or not (matched["label_nez_p2"] == matched["label_nez_v3"]).all():
        raise ValueError("label mismatch on matched channels")
    counts = matched.groupby("subject_id").size()
    if (counts < 2).any():
        raise ValueError("every matched patient must have at least two valid channels")
    if not matched[["p2_ez_score", "v3_ez_score"]].apply(np.isfinite).all().all():
        raise ValueError("matched scores contain non-finite values")
    return matched, merged


def _patient_percentile(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    if len(values) < 2:
        return pd.Series(np.ones(len(values)), index=series.index)
    return pd.Series(rankdata(values, method="average") / len(values), index=series.index)


def add_candidates(matched: pd.DataFrame) -> pd.DataFrame:
    out = matched.copy()
    out["p2_rank"] = out.groupby("subject_id")["p2_ez_score"].transform(_patient_percentile)
    out["v3_rank"] = out.groupby("subject_id")["v3_ez_score"].transform(_patient_percentile)
    out["p2_rank_position"] = out.groupby("subject_id")["p2_ez_score"].rank(ascending=False, method="average")
    out["v3_rank_position"] = out.groupby("subject_id")["v3_ez_score"].rank(ascending=False, method="average")
    out["P2"] = out["p2_ez_score"]
    out["V3"] = out["v3_ez_score"]
    out["RankAvg_50_50"] = 0.5 * out["p2_rank"] + 0.5 * out["v3_rank"]
    out["RankAvg_P2_70"] = 0.7 * out["p2_rank"] + 0.3 * out["v3_rank"]
    out["RankAvg_V3_70"] = 0.3 * out["p2_rank"] + 0.7 * out["v3_rank"]
    out["RRF_k60"] = 1.0 / (60.0 + out["p2_rank_position"]) - 1.0 / (60.0 + out["v3_rank_position"])
    return out


def _mrr(y_ez: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    hits = np.flatnonzero(y_ez[order] == 1)
    return float(1.0 / (hits[0] + 1)) if len(hits) else float("nan")


def patient_metrics(frame: pd.DataFrame, candidate: str) -> dict[str, float]:
    y_ez = frame["label_ez_p2"].to_numpy(int)
    score = frame[candidate].to_numpy(float)
    true_k = int(y_ez.sum())
    order = np.argsort(-score, kind="mergesort")
    pred = np.zeros(len(frame), dtype=int)
    pred[order[:true_k]] = 1
    y_nez = 1 - y_ez
    pred_nez = 1 - pred
    return {
        "patient_macro_f1": float(f1_score(y_ez, pred, labels=[0, 1], average="macro", zero_division=0)),
        "patient_ez_f1": float(f1_score(y_ez, pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0)),
        "patient_nez_f1": float(f1_score(y_nez, pred_nez, labels=[0, 1], average="binary", pos_label=1, zero_division=0)),
        "patient_balanced_accuracy": float(balanced_accuracy_score(y_ez, pred)),
        "patient_ez_auprc": float(average_precision_score(y_ez, score)) if len(np.unique(y_ez)) > 1 else float("nan"),
        "patient_ez_mrr": _mrr(y_ez, score),
        "top1_is_ez": float(y_ez[order[0]] == 1),
        "true_ez_count": true_k,
        "predicted_ez_count": int(pred.sum()),
        "predicted_ez_mask": pred,
    }


def evaluate(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict] = []
    for (subject_id, outer_fold, center), group in matched.groupby(["subject_id", "outer_fold_p2", "center_p2"], sort=True):
        row_base = {"subject_id": subject_id, "outer_fold": outer_fold, "center": center, "n_channels": len(group)}
        per_candidate = {}
        for candidate in CANDIDATES:
            metrics = patient_metrics(group, candidate)
            per_candidate[candidate] = metrics
            row_base.update({f"{candidate}_{key}": value for key, value in metrics.items() if key != "predicted_ez_mask"})
        rho = spearmanr(group["p2_ez_score"], group["v3_ez_score"]).statistic
        p2_set = set(group.iloc[np.argsort(-group["p2_ez_score"].to_numpy(), kind="mergesort")[:per_candidate["P2"]["true_ez_count"]]]["channel_key"])
        v3_set = set(group.iloc[np.argsort(-group["v3_ez_score"].to_numpy(), kind="mergesort")[:per_candidate["V3"]["true_ez_count"]]]["channel_key"])
        row_base["p2_v3_spearman_rank_correlation"] = float(rho) if np.isfinite(rho) else float("nan")
        row_base["p2_v3_true_k_jaccard"] = float(len(p2_set & v3_set) / len(p2_set | v3_set)) if p2_set | v3_set else 1.0
        row_base["patient_oracle_max_p2_v3_f1"] = max(row_base["P2_patient_macro_f1"], row_base["V3_patient_macro_f1"])
        patient_rows.append(row_base)
    patients = pd.DataFrame(patient_rows)
    long_rows = []
    for candidate in CANDIDATES:
        for scope_name, scope_values in [("overall", [None]), ("by_outer_fold", patients["outer_fold"].unique().tolist()), ("by_center", patients["center"].unique().tolist())]:
            for group_value in scope_values:
                subset = patients if scope_name == "overall" else patients[patients["outer_fold" if scope_name == "by_outer_fold" else "center"] == group_value]
                for metric in ("patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_balanced_accuracy", "patient_ez_auprc", "patient_ez_mrr", "top1_is_ez"):
                    values = pd.to_numeric(subset[f"{candidate}_{metric}"], errors="coerce")
                    row = {"scope": scope_name, "candidate": candidate, "metric": metric, "value": float(values.mean())}
                    if scope_name != "overall": row["group"] = str(group_value)
                    long_rows.append(row)
    return patients, pd.DataFrame(long_rows)


def paired_bootstrap(patients: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(patients)
    if n == 0: raise ValueError("cannot bootstrap empty patient table")
    rows = []
    p2 = patients["P2_patient_macro_f1"].to_numpy(float)
    for candidate in FUSIONS:
        delta = patients[f"{candidate}_patient_macro_f1"].to_numpy(float) - p2
        samples = np.empty(repeats, dtype=float)
        for index in range(repeats):
            samples[index] = float(delta[rng.integers(0, n, size=n)].mean())
        rows.append({"candidate": candidate, "mean_delta": float(delta.mean()), "ci_2_5": float(np.percentile(samples, 2.5)), "ci_97_5": float(np.percentile(samples, 97.5)), "probability_delta_gt_0": float(np.mean(samples > 0)), "bootstrap_repeats": repeats, "seed": seed})
    return pd.DataFrame(rows)


def _alignment_report(matched: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame([{
        "metric": "rows", "value": len(matched), "status": "passed" if len(matched) else "failed"
    }, {"metric": "p2_rows", "value": int((merged["_merge"] == "left_only").sum() + len(matched)), "status": "passed"}, {
        "metric": "v3_rows", "value": int((merged["_merge"] == "right_only").sum() + len(matched)), "status": "passed"
    }, {"metric": "matched_patients", "value": matched["subject_id"].nunique(), "status": "passed"}, {
        "metric": "unmatched_rows", "value": int((merged["_merge"] != "both").sum()), "status": "passed" if (merged["_merge"] != "both").sum() == 0 else "warning"
    }])
    report["analysis_status"] = ANALYSIS_STATUS
    report["true_count_used_for_prediction"] = TRUE_COUNT_USED
    report["formal_prediction"] = FORMAL_PREDICTION
    return report


def _write_unmatched(merged: pd.DataFrame, output: Path) -> None:
    rows = merged[merged["_merge"] != "both"].copy()
    rows["match_status"] = rows["_merge"].map({"left_only": "p2_only", "right_only": "v3_only"})
    rows["analysis_status"] = ANALYSIS_STATUS
    rows["true_count_used_for_prediction"] = TRUE_COUNT_USED
    rows["formal_prediction"] = FORMAL_PREDICTION
    rows.to_csv(output, index=False)


def _wide_scope_table(summary: pd.DataFrame, scope: str) -> pd.DataFrame:
    frame = summary[summary["scope"] == scope]
    index = ["candidate"] if scope == "overall" else ["group", "candidate"]
    table = frame.pivot(index=index, columns="metric", values="value").reset_index()
    if scope == "by_outer_fold":
        table = table.rename(columns={"group": "outer_fold"})
    elif scope == "by_center":
        table = table.rename(columns={"group": "center"})
    table["analysis_status"] = ANALYSIS_STATUS
    table["true_count_used_for_prediction"] = TRUE_COUNT_USED
    table["formal_prediction"] = FORMAL_PREDICTION
    table["decision_rule"] = "true_k_topk_diagnostic"
    return table


def _add_complementarity_summary(table: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    p2 = patients["P2_patient_macro_f1"].to_numpy(float)
    table = table.copy()
    oracle = np.maximum(patients["P2_patient_macro_f1"], patients["V3_patient_macro_f1"])
    table["patient_oracle_mean_f1"] = float(oracle.mean())
    table["p2_v3_mean_spearman_rank_correlation"] = float(patients["p2_v3_spearman_rank_correlation"].mean())
    table["p2_v3_mean_true_k_jaccard"] = float(patients["p2_v3_true_k_jaccard"].mean())
    for candidate in FUSIONS:
        delta = patients[f"{candidate}_patient_macro_f1"].to_numpy(float) - p2
        mask = table["candidate"].eq(candidate)
        table.loc[mask, "fusion_improved_patient_count"] = int((delta > 0).sum())
        table.loc[mask, "fusion_declined_patient_count"] = int((delta < 0).sum())
        table.loc[mask, "fusion_gain_gt_0_05_count"] = int((delta > 0.05).sum())
        table.loc[mask, "fusion_drop_gt_0_05_count"] = int((delta < -0.05).sum())
        table.loc[mask, "fusion_mean_delta_vs_p2"] = float(delta.mean())
    table["analysis_status"] = ANALYSIS_STATUS
    return table


def make_report(patients: pd.DataFrame, summary: pd.DataFrame, bootstrap: pd.DataFrame, *, repeats: int, seed: int, p2_path: str, v3_path: str) -> str:
    overall = summary[summary["scope"] == "overall"].pivot(index="candidate", columns="metric", values="value")
    p2_f1 = float(overall.loc["P2", "patient_macro_f1"])
    best = overall["patient_macro_f1"].astype(float).max()
    best_candidate = str(overall["patient_macro_f1"].astype(float).idxmax())
    fold = summary[(summary["scope"] == "by_outer_fold") & (summary["metric"] == "patient_macro_f1")]
    fold_pivot = fold.pivot(index="group", columns="candidate", values="value")
    if best >= .680 and best - p2_f1 >= .015 and sum(float(fold_pivot.loc[idx, best_candidate]) >= float(fold_pivot.loc[idx, "P2"]) for idx in fold_pivot.index) >= 4:
        verdict = "strong pass"
    elif .670 <= best < .680 and best - p2_f1 >= .008:
        verdict = "weak pass"
    else:
        verdict = "not passed"
    return f"""# P2/V3 Rank Fusion Audit\n\n- `analysis_status`: `{ANALYSIS_STATUS}`\n- `true_count_used_for_prediction`: `True`\n- `formal_prediction`: `False`\n- P2 ledger: `{p2_path}`\n- V3 ledger: `{v3_path}`\n- Patients evaluated: `{len(patients)}`\n- Bootstrap: `{repeats}` patient-level resamples, seed `{seed}`\n\n## Verdict\n\n**{verdict}**. Best candidate by true-K patient macro-F1: `{best_candidate}` (`{best:.6f}`); matched-cohort P2 baseline: `{p2_f1:.6f}`; gain: `{best - p2_f1:.6f}`.\n\nThis verdict is a ranking diagnostic only. It is not deployable performance because true EZ count is used to form every prediction mask. No threshold, test-label calibration, center-specific weight, or continuous fusion search was used.\n\n## Outputs\n\nThe CSV files contain overall, fold, center, patient, alignment, unmatched-channel, and paired-bootstrap diagnostics. All candidate methods are limited to: `{', '.join(CANDIDATES)}`.\n"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-ledger", required=True)
    parser.add_argument("--v3-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v3-score-column", default=None)
    parser.add_argument("--v3-score-semantics", required=True, choices=["ez_score", "nez_score", "ez_probability", "nez_probability"])
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repeats <= 0: raise ValueError("--bootstrap-repeats must be positive")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    matched, merged = load_and_align(args.p2_ledger, args.v3_ledger, v3_score_column=args.v3_score_column, v3_score_semantics=args.v3_score_semantics)
    matched = add_candidates(matched)
    patients, summary = evaluate(matched)
    bootstrap = paired_bootstrap(patients, args.bootstrap_repeats, args.seed)
    _alignment_report(matched, merged).to_csv(output / "channel_alignment_audit.csv", index=False)
    _write_unmatched(merged, output / "unmatched_channels.csv")
    patients["analysis_status"] = ANALYSIS_STATUS
    patients["true_count_used_for_prediction"] = TRUE_COUNT_USED
    patients["formal_prediction"] = FORMAL_PREDICTION
    patients.to_csv(output / "rank_fusion_by_patient.csv", index=False)
    _wide_scope_table(summary, "by_outer_fold").to_csv(output / "rank_fusion_by_fold.csv", index=False)
    _wide_scope_table(summary, "by_center").to_csv(output / "rank_fusion_by_center.csv", index=False)
    overall_table = _add_complementarity_summary(_wide_scope_table(summary, "overall"), patients)
    overall_table.to_csv(output / "rank_fusion_summary.csv", index=False)
    bootstrap["analysis_status"] = ANALYSIS_STATUS
    bootstrap["true_count_used_for_prediction"] = TRUE_COUNT_USED
    bootstrap["formal_prediction"] = FORMAL_PREDICTION
    bootstrap.to_csv(output / "rank_fusion_bootstrap.csv", index=False)
    (output / "P2_V3_RANK_FUSION_AUDIT.md").write_text(make_report(patients, summary, bootstrap, repeats=args.bootstrap_repeats, seed=args.seed, p2_path=args.p2_ledger, v3_path=args.v3_ledger), encoding="utf-8")


if __name__ == "__main__":
    main()
