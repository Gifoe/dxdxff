"""Ledger-only formal evaluation for P2, V3, and the locked probability fusion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from P2_V3_AAAI_ABLATIONS.metrics import evaluate, summaries
from P2_V3_AAAI_ABLATIONS.threshold_protocol import select_fold_threshold
from neuroez_c.p2_v3_fusion_protocol import align_fold_ledgers, canonicalize_p2_fusion_ledger, canonicalize_v3_fusion_ledger, discover_fold_ledger
from neuroez_c.p2_v3_conservative_fusion import (
    LOCKED_P2_WEIGHT, LOCKED_V3_WEIGHT, conservative_probability_fusion,
    require_locked_weights,
)
from .provenance import (
    P2_METHOD_ID, P2_PROFILE_ID, V3_METHOD_ID, V3_PROFILE_ID,
    validate_branch_provenance,
)


MODELS = {"PRQ-Net": "p2_score_nez", "BCR-Net": "v3_score_nez", "CDEL": "fused_score_nez"}


def _evaluate_true_k_diagnostic(
    frame: pd.DataFrame, *, score_column: str, experiment: str,
) -> pd.DataFrame:
    """Evaluate ranking with label-derived K; this is never a formal prediction."""
    rows = []
    for _, patient in frame.groupby("subject_id", sort=True):
        label_nez = patient["label_nez"].to_numpy(dtype=int)
        label_ez = 1 - label_nez
        score_ez = 1.0 - patient[score_column].to_numpy(dtype=float)
        order = np.argsort(-score_ez, kind="mergesort")
        true_k = int(label_ez.sum())
        predicted_nez = np.ones(len(patient), dtype=int)
        predicted_nez[order[:true_k]] = 0
        rows.append({
            "subject_id": str(patient["subject_id"].iloc[0]),
            "center": str(patient["center"].iloc[0]),
            "outer_fold": int(patient["outer_fold"].iloc[0]),
            "n_channels": int(len(patient)),
            "true_ez_count": true_k,
            "predicted_ez_count": true_k,
            "true_ez_fraction": float(label_ez.mean()),
            "predicted_ez_fraction": float(label_ez.mean()),
            "patient_macro_f1": float(f1_score(label_nez, predicted_nez, labels=[0, 1], average="macro", zero_division=0)),
            "patient_ez_f1": float(f1_score(label_nez, predicted_nez, labels=[0], average="macro", zero_division=0)),
            "patient_nez_f1": float(f1_score(label_nez, predicted_nez, labels=[1], average="macro", zero_division=0)),
            "patient_accuracy": float((label_nez == predicted_nez).mean()),
            "patient_balanced_accuracy": float(balanced_accuracy_score(label_nez, predicted_nez)),
            # Threshold-free ranking metrics are copied from the formal evaluator.
            "patient_ez_auprc": np.nan,
            "patient_ez_auroc": np.nan,
            "patient_ez_mrr": np.nan,
            "patient_ez_ndcg": np.nan,
            "top1_is_ez_rate": np.nan,
            "predicted_ez_count_mae": 0.0,
            "predicted_ez_fraction_mae": 0.0,
            "truek_patient_macro_f1": float(f1_score(label_nez, predicted_nez, labels=[0, 1], average="macro", zero_division=0)),
            "selected_validation_threshold": np.nan,
            "experiment": experiment,
            "analysis_status": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
            "true_count_used_for_prediction": True,
            "formal_prediction": False,
        })
    return pd.DataFrame(rows)


def evaluate_pair(
    *, p2_root: str | Path, v3_root: str | Path, folds: Iterable[int], output_dir: str | Path,
    analysis_status: str = "PRIMARY_CONFIRMATORY", p2_weight: float = LOCKED_P2_WEIGHT, v3_weight: float = LOCKED_V3_WEIGHT,
    require_provenance: bool = False, expected_seed: int | None = None,
    expected_held_out_center: str | None = None,
) -> dict:
    require_locked_weights(float(p2_weight), float(v3_weight))
    target = Path(output_dir); (target / "metrics").mkdir(parents=True, exist_ok=True); (target / "ledgers").mkdir(exist_ok=True); (target / "thresholds").mkdir(exist_ok=True); (target / "audit").mkdir(exist_ok=True)
    provenance_audit: dict[str, object] = {"required": bool(require_provenance)}
    if require_provenance:
        p2_provenance = validate_branch_provenance(
            p2_root, expected_method_id=P2_METHOD_ID, expected_profile_id=P2_PROFILE_ID,
            expected_seed=expected_seed, expected_held_out_center=expected_held_out_center,
        )
        v3_provenance = validate_branch_provenance(
            v3_root, expected_method_id=V3_METHOD_ID, expected_profile_id=V3_PROFILE_ID,
            expected_seed=expected_seed, expected_held_out_center=expected_held_out_center,
        )
        for key in ("cohort_sha256", "partition_sha256", "feature_cache_sha256"):
            if p2_provenance[key] != v3_provenance[key]:
                raise RuntimeError(f"P2/V3 provenance mismatch for {key}")
        provenance_audit.update({"status": "passed", "p2": p2_provenance, "v3": v3_provenance})
    patient_rows, truek_rows, ledgers, threshold_rows, alignment = [], [], [], [], []
    for fold in sorted(set(map(int, folds))):
        aligned_roles = {}
        for role in ("validation", "test"):
            p2 = canonicalize_p2_fusion_ledger(
                pd.read_csv(discover_fold_ledger(p2_root, fold, role)),
                # P2 identity is verified from immutable branch provenance above.
                # Keep the ledger adapter ABI compatible with the frozen server copy.
                split_role=role,
            )
            v3 = canonicalize_v3_fusion_ledger(pd.read_csv(discover_fold_ledger(v3_root, fold, role)), split_role=role)
            merged, audit, _ = align_fold_ledgers(p2, v3, outer_fold=fold, split_role=role)
            merged["fused_score_nez"], _ = conservative_probability_fusion(
                merged.p2_score_nez.to_numpy(), merged.v3_score_nez.to_numpy(),
                p2_weight=float(p2_weight), v3_weight=float(v3_weight),
            )
            merged["fused_score_ez"] = 1.0 - merged.fused_score_nez
            aligned_roles[role] = merged
            alignment.append(audit)
        for model, column in MODELS.items():
            threshold, search = select_fold_threshold(aligned_roles["validation"], score_column=column)
            search["outer_fold"] = fold; search["model"] = model; search["analysis_status"] = analysis_status
            search.to_csv(target / "thresholds" / f"threshold_search_{model.replace('-', '_')}_fold_{fold}.csv", index=False)
            threshold_rows.append({"outer_fold": fold, "model": model, "threshold": threshold, "threshold_source": "validation_only", "grid_step": .005})
            formal = evaluate(aligned_roles["test"], score_column=column, threshold=threshold, experiment=model, analysis_status=analysis_status)
            patient_rows.append(formal)
            diagnostic = _evaluate_true_k_diagnostic(
                aligned_roles["test"], score_column=column, experiment=model,
            )
            # Ranking metrics do not depend on the decision mask, so retain the
            # same score-only values in the explicitly non-deployable true-K view.
            diagnostic = diagnostic.merge(
                formal[["subject_id", "patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr", "patient_ez_ndcg", "top1_is_ez_rate"]],
                on="subject_id", how="left", suffixes=("", "_formal"), validate="one_to_one",
            )
            for metric in ("patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr", "patient_ez_ndcg", "top1_is_ez_rate"):
                diagnostic[metric] = diagnostic.pop(f"{metric}_formal")
            truek_rows.append(diagnostic)
        copy = aligned_roles["test"].copy()
        copy["p2_weight"] = float(p2_weight); copy["v3_weight"] = float(v3_weight); ledgers.append(copy)
    patients = pd.concat(patient_rows, ignore_index=True); truek = pd.concat(truek_rows, ignore_index=True); ledger = pd.concat(ledgers, ignore_index=True)
    overall, by_fold, by_center = summaries(patients)
    true_overall, _, _ = summaries(truek)
    patients.to_csv(target / "metrics" / "patient_level.csv", index=False)
    truek.to_csv(target / "metrics" / "patient_level_truek_diagnostic.csv", index=False)
    overall.to_csv(target / "metrics" / "overall.csv", index=False)
    by_fold.to_csv(target / "metrics" / "by_fold.csv", index=False)
    by_center.to_csv(target / "metrics" / "by_center.csv", index=False)
    true_overall.to_csv(target / "metrics" / "truek_overall_diagnostic.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(target / "thresholds" / "selected_thresholds.csv", index=False)
    ledger.to_csv(target / "ledgers" / "oof_channel_predictions.csv", index=False)
    (target / "audit" / "alignment_audit.json").write_text(json.dumps({"status": "passed", "rows": alignment}, indent=2), encoding="utf-8")
    (target / "audit" / "branch_provenance_audit.json").write_text(
        json.dumps(provenance_audit, indent=2), encoding="utf-8",
    )
    return {"status": "passed", "n_patients": int(patients.subject_id.nunique()), "n_channels": int(len(ledger)), "models": overall.to_dict("records")}
