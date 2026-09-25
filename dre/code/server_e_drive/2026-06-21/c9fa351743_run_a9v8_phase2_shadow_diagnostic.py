"""A9v8 Phase 2.5 shadow diagnostic: teacher/physiology score fusion evaluation.

Does NOT train — only evaluates label-free rank fusion of A9v3 OOF scores
with physiology core scores.

Diagnostic A: label-free rank fusion (all patients)
Diagnostic B: strong-center surrogate validation (HUP+multicenter only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _z_patient(df: pd.DataFrame, score_col: str) -> pd.Series:
    def transform(values: pd.Series) -> pd.Series:
        arr = values.astype(float).to_numpy()
        std = float(arr.std(ddof=0))
        if std <= 1e-8:
            return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        return pd.Series((arr - float(arr.mean())) / std, index=values.index)
    return df.groupby("patient_id")[score_col].transform(transform)


def _evaluate_ledger(df: pd.DataFrame) -> dict[str, float]:
    """Evaluate patient-level metrics from a ledger-format dataframe."""
    f1s, ez_f1s, auprcs, mrrs, top1s = [], [], [], [], []
    for _, group in df.groupby("patient_id", sort=False):
        y_ez = group["label_ez"].astype(float).to_numpy()
        y_nez = 1 - y_ez
        k = int(round(float(y_ez.sum())))
        if k <= 0:
            continue
        scores = group["score_eval"].astype(float).to_numpy()
        ranks = scores.argsort()[::-1].argsort() + 1
        pred_ez = (ranks <= k).astype(int)
        pred_nez = 1 - pred_ez

        f1s.append(float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)))
        ez_f1 = f1_score(y_ez, pred_ez, average="binary", zero_division=0)
        ez_f1s.append(float(ez_f1))
        auprcs.append(float(average_precision_score(y_ez, scores)) if np.unique(y_ez).size > 1 else 0.0)

        pos = np.where(y_ez > 0.5)[0]
        mrr = float(max(1.0 / ranks[int(idx)] for idx in pos)) if len(pos) > 0 else 0.0
        mrrs.append(mrr)
        top1s.append(float(y_ez[ranks.argmin()]))

    def _mean(arr):
        return float(np.mean(arr)) if arr else 0.0

    auprc_val = _mean(auprcs)
    return {
        "patient_macro_f1": _mean(f1s),
        "patient_macro_ez_f1": _mean(ez_f1s),
        "patient_macro_auprc_ez": auprc_val,
        "auprc_ez": auprc_val,
        "patient_macro_ez_mrr": _mean(mrrs),
        "top1_is_ez_rate": _mean(top1s),
    }


def _load_audit(audit_json: str | Path | None) -> dict:
    if not audit_json:
        return {}
    path = Path(audit_json)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _bool_from_series_or_default(df: pd.DataFrame, column: str, default: bool) -> bool:
    if column not in df.columns:
        return bool(default)
    vals = df[column].dropna().astype(str).str.lower()
    if vals.empty:
        return bool(default)
    return bool(vals.isin({"1", "true", "yes"}).any())


def run_shadow_diagnostic(
    ledger_or_latent: pd.DataFrame,
    latent: pd.DataFrame | None = None,
    gammas: list[float] | None = None,
    *,
    audit_json: str | Path | None = None,
    require_phys_core: bool = False,
    allow_teacher_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both diagnostic A and B.

    Returns:
        (summary_df, by_center_df, strong_surrogate_df)
    """
    gammas = gammas or [0.05, 0.10, 0.15]
    if latent is None:
        merged = ledger_or_latent.copy()
    else:
        ledger = ledger_or_latent
        merge_keys = ["patient_id", "channel_name"]
        if "record_id" in ledger.columns and "record_id" in latent.columns:
            merge_keys = ["patient_id", "record_id", "channel_name"]
        keep_cols = merge_keys + ["a9v3_oof_score", "phys_core_score", "pseudo_core_q", "core_score", "phys_core_score_available", "teacher_only_fallback_used"]
        keep_cols = [col for col in keep_cols if col in latent.columns]
        merged = ledger.merge(latent[keep_cols], on=merge_keys, how="inner")
        if "label_ez" not in merged.columns and "label_ez" in latent.columns:
            merged = merged.merge(latent[merge_keys + ["label_ez"]], on=merge_keys, how="left")
    if merged.empty:
        raise ValueError("Shadow diagnostic input has no rows.")
    if "patient_id" not in merged.columns and "subject_id" in merged.columns:
        merged["patient_id"] = merged["subject_id"].astype(str)
    if "center" not in merged.columns:
        merged["center"] = "unknown"
    if "a9v3_oof_score" not in merged.columns and "score_eval" in merged.columns:
        merged["a9v3_oof_score"] = merged["score_eval"]
    if "core_score" not in merged.columns:
        merged["core_score"] = merged["a9v3_oof_score"]

    merged["teacher_z"] = _z_patient(merged, "a9v3_oof_score")
    if "phys_core_score" not in merged.columns:
        merged["phys_core_score"] = 0.0
    merged["phys_z"] = _z_patient(merged, "phys_core_score")

    audit = _load_audit(audit_json)
    phys_available = bool(audit.get("phys_core_score_available", _bool_from_series_or_default(merged, "phys_core_score_available", False)))
    teacher_only_used = bool(audit.get("teacher_only_fallback_used", _bool_from_series_or_default(merged, "teacher_only_fallback_used", False)))
    if phys_available:
        phys_available = float(merged["phys_core_score"].astype(float).std(ddof=0)) > 1e-8
    if require_phys_core and not phys_available:
        raise ValueError("phys_core_score is not available; rerun with --allow_teacher_only only for explicit teacher-only diagnostic.")
    if not phys_available and not allow_teacher_only:
        raise ValueError("phys_core_score is not available and --allow_teacher_only was not set.")
    latent_core_mode = "teacher_plus_phys" if phys_available else "teacher_only"

    # ---- score types to evaluate ----
    score_specs: dict[str, str] = {
        "a9v3_oof_score": "a9v3_oof_score",
        "teacher_only_core_score": "core_score",
    }
    if phys_available:
        score_specs["phys_core_score"] = "phys_core_score"
        for gamma in gammas:
            col = f"teacher_plus_phys_gamma_{gamma:g}"
            merged[col] = merged["teacher_z"] + float(gamma) * merged["phys_z"]
            score_specs[f"teacher_plus_phys_gamma_{gamma:g}"] = col

    # Diagnostic A: all patients
    summary_rows, by_center_rows = [], []
    for label, col in score_specs.items():
        df = merged.copy()
        df["score_eval"] = df[col].astype(float)
        metrics = _evaluate_ledger(df)
        metrics["score_type"] = label
        metrics["latent_core_mode"] = latent_core_mode
        metrics["gamma"] = float(label.replace("teacher_plus_phys_gamma_", "")) if "teacher_plus_phys_gamma" in label else float("nan")
        metrics["phys_core_score_available"] = phys_available
        metrics["teacher_only_fallback_used"] = teacher_only_used
        summary_rows.append(metrics)

        # by-center
        for center, cdf in df.groupby("center", sort=True):
            cm = _evaluate_ledger(cdf)
            cm["score_type"] = label
            cm["center"] = str(center)
            cm["latent_core_mode"] = latent_core_mode
            cm["gamma"] = metrics["gamma"]
            cm["phys_core_score_available"] = phys_available
            cm["teacher_only_fallback_used"] = teacher_only_used
            by_center_rows.append(cm)

    summary_df = pd.DataFrame(summary_rows)
    by_center_df = pd.DataFrame(by_center_rows)

    # Diagnostic B: strong-center surrogate (HUP + multicenter)
    strong = merged[~merged["center"].astype(str).str.lower().isin({"lzu", "pediatric"})].copy()
    strong_rows = []
    for label, col in score_specs.items():
        df = strong.copy()
        df["score_eval"] = df[col].astype(float)
        metrics = _evaluate_ledger(df)
        metrics["score_type"] = label
        metrics["center_group"] = "hup_multicenter"
        metrics["latent_core_mode"] = latent_core_mode
        metrics["gamma"] = float(label.replace("teacher_plus_phys_gamma_", "")) if "teacher_plus_phys_gamma" in label else float("nan")
        metrics["phys_core_score_available"] = phys_available
        metrics["teacher_only_fallback_used"] = teacher_only_used
        strong_rows.append(metrics)

    strong_df = pd.DataFrame(strong_rows)

    # Ensure required columns exist
    for _df, _name in [(summary_df, "summary"), (by_center_df, "by_center"), (strong_df, "strong")]:
        for req_col in ("score_type", "patient_macro_f1", "patient_macro_ez_mrr", "top1_is_ez_rate"):
            if req_col not in _df.columns:
                _df[req_col] = 0.0
        if "gamma" not in _df.columns:
            _df["gamma"] = float("nan")
        if "center_group" not in _df.columns:
            _df["center_group"] = "hup_multicenter" if _name == "strong" else ""

    return summary_df, by_center_df, strong_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A9v8 phase-2.5 shadow diagnostic without changing training."
    )
    parser.add_argument("--ledger_csv", type=str, default=None)
    parser.add_argument("--latent_core_csv", type=str, required=True)
    parser.add_argument("--audit_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--require_phys_core", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow_teacher_only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent = pd.read_csv(args.latent_core_csv)
    if args.ledger_csv:
        ledger = pd.read_csv(args.ledger_csv)
        summary, by_center, strong = run_shadow_diagnostic(
            ledger,
            latent,
            list(args.gammas),
            audit_json=args.audit_json,
            require_phys_core=bool(args.require_phys_core),
            allow_teacher_only=bool(args.allow_teacher_only),
        )
    else:
        summary, by_center, strong = run_shadow_diagnostic(
            latent,
            gammas=list(args.gammas),
            audit_json=args.audit_json,
            require_phys_core=bool(args.require_phys_core),
            allow_teacher_only=bool(args.allow_teacher_only),
        )

    summary.to_csv(output_dir / "shadow_diagnostic_summary.csv", index=False)
    by_center.to_csv(output_dir / "shadow_diagnostic_by_center.csv", index=False)
    strong.to_csv(output_dir / "strong_center_surrogate_summary.csv", index=False)

    print(f"Wrote {output_dir / 'shadow_diagnostic_summary.csv'}")
    print(f"Wrote {output_dir / 'shadow_diagnostic_by_center.csv'}")
    print(f"Wrote {output_dir / 'strong_center_surrogate_summary.csv'}")


if __name__ == "__main__":
    main()
