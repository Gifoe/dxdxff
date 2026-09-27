"""Deterministic label-free channel-set perturbations at S-F1 VLOO-selected checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score

from common import (A1_RUNTIME, EXPERIMENT, RUNTIME, assert_no_outer_loader, build_fold,
                    core, ensure_source, make_experiment, mean, read_csv, source_checkpoint,
                    write_csv)
from train_variants import build_model

VARIANTS = ("CTX0", "CTX1", "CTX2", "CTX3")
FRACTIONS = (0.10, 0.25, 0.50)


def mask_seed(fold: int, subject: str, fraction: float, repeat: int) -> int:
    # Identical label-free mask for every structural variant.
    label_free = f"42|{fold}|{subject}|{fraction:.2f}|{repeat}"
    return int.from_bytes(hashlib.sha256(label_free.encode()).digest()[:8], "little")


def forward(model, batch: dict, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(core._move_tensors_to_device(batch, device))
    return out["score_ez"][0].detach().cpu().numpy(), out["logits"][0].detach().cpu().numpy()


def perturb(batch: dict, retained: np.ndarray) -> dict:
    new = dict(batch)
    channel = batch["channel_mask"].clone()
    seizure = batch["seizure_channel_mask"].clone()
    dropped = np.flatnonzero(~retained)
    channel[0, dropped] = False
    seizure[0, :, dropped] = False
    new["channel_mask"], new["seizure_channel_mask"] = channel, seizure
    return new


def comparison(full: np.ndarray, changed: np.ndarray, logits_full: np.ndarray,
               logits_changed: np.ndarray, retained: np.ndarray, labels: np.ndarray) -> dict:
    idx = np.flatnonzero(retained)
    a, b = full[idx], changed[idx]
    rho = float(spearmanr(a, b).statistic) if len(idx) > 1 else 1.0
    if not np.isfinite(rho):
        rho = 1.0 if np.array_equal(a, b) else 0.0
    pct_a = rankdata(a, method="average") / len(a)
    pct_b = rankdata(b, method="average") / len(b)
    top_n = max(1, math.ceil(0.2 * len(idx)))
    top_a = set(np.argsort(a, kind="stable")[-top_n:].tolist())
    top_b = set(np.argsort(b, kind="stable")[-top_n:].tolist())
    y = labels[idx]
    auprc_delta = (float(average_precision_score(y, b) - average_precision_score(y, a))
                   if len(set(y.tolist())) == 2 else "")
    return {"score_spearman": rho, "percentile_rank_mae": float(np.mean(np.abs(pct_a - pct_b))),
            "top20_jaccard": len(top_a & top_b) / len(top_a | top_b),
            "mean_abs_logit_shift": float(np.mean(np.abs(logits_full[idx] - logits_changed[idx]))),
            "retained_ez_auprc_delta": auprc_delta}


def main() -> None:
    ensure_source()
    exp = make_experiment()
    view_partition = json.loads((EXPERIMENT / "view_interference" / "RANDOM_VIEW_PARTITION.json").read_text(encoding="utf-8"))
    private_rows = []
    channel_counts = []
    for split in exp.outer_splits:
        fold, _train, train_loader, val_loader, test_loader, _norm = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        single_loader = exp._make_loader(val_loader.dataset, shuffle=False, batch_size=1)
        batches = {str(batch["subject_id"][0]): batch for batch in single_loader}
        channel_counts.extend(int(batch["channel_mask"].sum()) for batch in batches.values())
        initial = torch.load(A1_RUNTIME / "initial" / f"fold_{fold}_initial.pt", map_location="cpu", weights_only=True)
        for variant in VARIANTS:
            selected = {row["subject_id"]: int(row["selected_epoch"]) for row in
                        read_csv(RUNTIME / "private" / f"{variant}_fold_{fold}_SF1_PATIENT.csv")}
            if set(selected) != set(batches):
                raise RuntimeError("Selected validation channel-set membership changed")
            model, _counts = build_model(exp, train_loader, variant, fold, initial, view_partition)
            by_epoch = {}
            for patient, epoch in selected.items():
                by_epoch.setdefault(epoch, []).append(patient)
            for epoch, patients in sorted(by_epoch.items()):
                checkpoint_path = (source_checkpoint(fold, epoch) if variant == "CTX0" else
                                   RUNTIME / "private" / "training" / variant / f"fold_{fold}" / f"epoch_{epoch:02d}.pt")
                state = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
                model.load_state_dict(state["model_state_dict"], strict=True)
                for patient in patients:
                    batch = batches[patient]
                    valid = batch["channel_mask"][0].numpy().astype(bool)
                    n = int(valid.sum())
                    if n < 4:
                        raise RuntimeError("Channel set too small for perturbation")
                    labels = batch["labels_ez"][0].numpy().astype(int)
                    full, full_logits = forward(model, batch, exp.device)
                    for fraction in FRACTIONS:
                        dropped_n = min(n - 2, max(1, round(fraction * n)))
                        for repeat in range(20):
                            rng = random.Random(mask_seed(fold, patient, fraction, repeat))
                            dropped = rng.sample(np.flatnonzero(valid).tolist(), dropped_n)
                            retained = valid.copy()
                            retained[dropped] = False
                            changed, changed_logits = forward(model, perturb(batch, retained), exp.device)
                            private_rows.append({"fold": fold, "variant": variant, "subject_id": patient,
                                                 "selected_epoch": epoch, "n_channels": n, "fraction": fraction,
                                                 "repeat": repeat, "n_retained": int(retained.sum()),
                                                 **comparison(full, changed, full_logits, changed_logits, retained, labels)})
            print(f"[CHANNEL-SET] {variant} fold={fold} complete", flush=True)
    if len(private_rows) != 5 * 13 * 4 * 3 * 20:
        raise RuntimeError(f"Channel-set audit incomplete: {len(private_rows)}")
    write_csv(RUNTIME / "private" / "CHANNEL_SET_PATIENT_REPEATS.csv", private_rows)
    quartiles = np.quantile(channel_counts, [0.25, 0.5, 0.75])
    for row in private_rows:
        row["n_channels_quartile"] = "Q" + str(int(np.searchsorted(quartiles, row["n_channels"], side="left")) + 1)
    public = []
    for variant in VARIANTS:
        for fraction in FRACTIONS:
            for quartile in ("ALL", "Q1", "Q2", "Q3", "Q4"):
                subset = [row for row in private_rows if row["variant"] == variant and row["fraction"] == fraction and
                          (quartile == "ALL" or row["n_channels_quartile"] == quartile)]
                ap_valid = [row for row in subset if row["retained_ez_auprc_delta"] != ""]
                public.append({"variant": variant, "fraction": fraction, "n_channels_quartile": quartile,
                               "n_patient_repeats": len(subset), "n_auprc_valid": len(ap_valid),
                               **{name: mean(subset, name) for name in ("score_spearman", "percentile_rank_mae",
                                                                    "top20_jaccard", "mean_abs_logit_shift")},
                               "retained_ez_auprc_delta": mean(ap_valid, "retained_ez_auprc_delta") if ap_valid else ""})
    write_csv(EXPERIMENT / "context_attention" / "CHANNEL_SET_SENSITIVITY.csv", public)
    print(json.dumps({"channel_set_complete": True, "patient_repeats": len(private_rows)}), flush=True)


if __name__ == "__main__":
    main()
