"""Label-free fixed-lambda patient-attention interventions on all frozen A1 epochs."""

from __future__ import annotations

import csv
import json

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from common import (EXPERIMENT, RUNTIME, assert_no_outer_loader, build_fold, core,
                    ensure_source, make_experiment, mean, read_csv, source_checkpoint,
                    write_csv)
from development_metrics import patient_grid
from patient_channel_ranker import _patient_relative_zscore

LAMBDAS = (1.0, 0.5, 0.0)


def score_agreement(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    tri = np.triu_indices(len(reference), 1)
    a = np.sign(reference[:, None] - reference[None, :])[tri]
    b = np.sign(candidate[:, None] - candidate[None, :])[tri]
    pairwise = float(np.mean(a == b)) if len(a) else 1.0
    rho = float(spearmanr(reference, candidate).statistic) if len(reference) > 1 else 1.0
    if not np.isfinite(rho):
        rho = 1.0 if np.array_equal(reference, candidate) else 0.0
    return pairwise, rho


def patient_metrics(batch: dict, i: int, score_nez: np.ndarray, score_ez: np.ndarray) -> dict:
    mask = batch["channel_mask"][i].numpy().astype(bool)
    record = {"subject_id": str(batch["subject_id"][i]), "channel_mask": mask,
              "labels": batch["labels_nez"][i].numpy(), "labels_nez": batch["labels_nez"][i].numpy(),
              "labels_ez": batch["labels_ez"][i].numpy(),
              "score_nez": score_nez, "score_ez": score_ez}
    return patient_grid(record)["fixed"]


def main() -> None:
    ensure_source()
    if not json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    exp = make_experiment()
    private = []
    for split in exp.outer_splits:
        fold, _train, train_loader, val_loader, test_loader, _norm = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        selected = {row["subject_id"]: int(row["selected_epoch"]) for row in
                    read_csv(RUNTIME / "private" / f"fold_{fold}_A1_SF1_PATIENT.csv")}
        model = exp.runtime["model_cls"](exp.args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        for epoch in range(1, 31):
            checkpoint = torch.load(source_checkpoint(fold, epoch), map_location=exp.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    out = model(core._move_tensors_to_device(batch, exp.device))
                    channel_mask = batch["channel_mask"].to(exp.device)
                    z = _patient_relative_zscore(out["patient_channel_embedding"], channel_mask)
                    padding = ~channel_mask
                    context, _ = model.channel_classifier.channel_attn(z, z, z, key_padding_mask=padding)
                    logits = {}
                    for lam in LAMBDAS:
                        contextual = model.channel_classifier.attn_norm(z + lam * context)
                        logits[lam] = model.channel_classifier.classifier(contextual).squeeze(-1).masked_fill(~channel_mask, -1e9)
                    if not torch.allclose(logits[1.0][channel_mask], out["logits"][channel_mask], rtol=0, atol=1e-6):
                        raise RuntimeError("T1.0 did not exactly replay frozen A1")
                    for i, subject_id in enumerate(batch["subject_id"]):
                        valid = batch["channel_mask"][i].numpy().astype(bool)
                        ref = 1.0 - torch.sigmoid(logits[1.0][i]).cpu().numpy()[valid]
                        for lam in LAMBDAS:
                            p_nez = torch.sigmoid(logits[lam][i]).cpu().numpy()
                            p_ez = 1.0 - p_nez
                            fixed = patient_metrics(batch, i, p_nez, p_ez)
                            agreement, rho = score_agreement(ref, p_ez[valid])
                            private.append({"fold": fold, "epoch": epoch, "subject_id": str(subject_id),
                                            "lambda": lam, "selected_by_SF1": epoch == selected[str(subject_id)],
                                            "n_channels": int(valid.sum()), "pairwise_agreement_T1": agreement,
                                            "score_spearman_T1": rho, **fixed})
            print(f"[ATTENTION] fold={fold} epoch={epoch}/30", flush=True)
    if len(private) != 5 * 30 * 13 * 3:
        raise RuntimeError("Attention intervention grid incomplete")
    write_csv(RUNTIME / "private" / "FROZEN_ATTENTION_PATIENT_EPOCH.csv", private)
    rows = []
    for fold in range(1, 6):
        for lam in LAMBDAS:
            subset = [row for row in private if row["fold"] == fold and row["lambda"] == lam and row["selected_by_SF1"]]
            if len(subset) != 13:
                raise RuntimeError("Selected A1 attention patients incomplete")
            rows.append({"fold": fold, "lambda": lam, "n_patients": len(subset),
                         **{metric: mean(subset, metric) for metric in ("patient_ez_auprc", "patient_ez_mrr", "patient_ez_auroc", "top1_is_ez",
                                                                      "pairwise_agreement_T1", "score_spearman_T1")}})
    write_csv(EXPERIMENT / "context_attention" / "FROZEN_ATTENTION_INTERVENTION.csv", rows)
    selected = {(row["fold"], row["subject_id"], row["lambda"]): row for row in private if row["selected_by_SF1"]}
    pairs = []
    for key, reference in selected.items():
        fold, subject_id, lam = key
        if lam != 1.0:
            continue
        intervention = selected[(fold, subject_id, 0.0)]
        pairs.append({"n_channels": reference["n_channels"],
                      "delta_auprc_T0_minus_T1": intervention["patient_ez_auprc"] - reference["patient_ez_auprc"],
                      "delta_mrr_T0_minus_T1": intervention["patient_ez_mrr"] - reference["patient_ez_mrr"]})
    diagnostics = []
    for field in ("delta_auprc_T0_minus_T1", "delta_mrr_T0_minus_T1"):
        x = np.asarray([row["n_channels"] for row in pairs], dtype=float)
        y = np.asarray([row[field] for row in pairs], dtype=float)
        diagnostics.append({"metric": field, "n_cases": len(pairs), "mean_delta": float(y.mean()),
                            "pearson_r_n_channels": float(pearsonr(x, y).statistic) if np.std(y) else "",
                            "spearman_rho_n_channels": float(spearmanr(x, y).statistic) if np.std(y) else "",
                            "diagnostic_only": True})
    write_csv(EXPERIMENT / "diagnostics" / "CHANNEL_COUNT_CONTEXT_SENSITIVITY.csv", diagnostics)
    print(json.dumps({"attention_intervention_complete": True, "patient_epoch_cells": len(private),
                      "T0_mean_auprc_delta": diagnostics[0]["mean_delta"]}), flush=True)


if __name__ == "__main__":
    main()
