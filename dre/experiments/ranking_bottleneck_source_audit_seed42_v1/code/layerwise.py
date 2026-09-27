"""Fixed linear ranking accessibility probes at frozen A1 selected checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import (A1_RUNTIME, EXPERIMENT, RUNTIME, assert_no_outer_loader, build_fold,
                    core, ensure_source, make_experiment, mean, read_csv,
                    source_checkpoint, write_csv)
from development_metrics import patient_grid
from objectives import patient_equal_weighted_bce_loss
from patient_channel_ranker import _patient_relative_zscore

LAYERS = ("L0", "L1", "L2", "L3")


def embeddings(model, loader, device) -> dict[str, dict]:
    captured = {}

    def pre(_module, inputs):
        captured["L2"] = inputs[0].detach()

    handle = model.channel_classifier.classifier.register_forward_pre_hook(pre)
    rows = {}
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                out = model(core._move_tensors_to_device(batch, device))
                if "L2" not in captured:
                    raise RuntimeError("Missing preclassifier contextual embedding")
                mask = batch["channel_mask"].to(device)
                l0 = out["patient_channel_embedding"]
                l1 = _patient_relative_zscore(l0, mask)
                l2 = captured["L2"]
                replay = model.channel_classifier.classifier(l2).squeeze(-1)
                if not torch.allclose(replay[mask], out["logits"][mask], atol=1e-6, rtol=0):
                    raise RuntimeError("L2 frozen logit replay failed")
                for i, subject in enumerate(batch["subject_id"]):
                    valid = batch["channel_mask"][i].numpy().astype(bool)
                    selected = torch.as_tensor(np.flatnonzero(valid), device=device)
                    rows[str(subject)] = {"L0": l0[i].index_select(0, selected).cpu().numpy().astype(np.float32),
                                          "L1": l1[i].index_select(0, selected).cpu().numpy().astype(np.float32),
                                          "L2": l2[i].index_select(0, selected).cpu().numpy().astype(np.float32),
                                          "L3": out["logits"][i].index_select(0, selected).cpu().numpy().astype(np.float32),
                                          "y_nez": batch["labels_nez"][i].numpy()[valid].astype(np.float32),
                                          "y_ez": batch["labels_ez"][i].numpy()[valid].astype(np.float32)}
    finally:
        handle.remove()
    return rows


def padded_fit(rows: dict[str, dict], layer: str, device):
    patients = [rows[key] for key in sorted(rows)]
    n = len(patients)
    c = max(len(row["y_ez"]) for row in patients)
    dim = patients[0][layer].shape[-1]
    x = torch.zeros((n, c, dim), dtype=torch.float32, device=device)
    y_ez = torch.zeros((n, c), dtype=torch.float32, device=device)
    y_nez = torch.zeros_like(y_ez)
    mask = torch.zeros((n, c), dtype=torch.bool, device=device)
    for i, row in enumerate(patients):
        channels = len(row["y_ez"])
        x[i, :channels] = torch.from_numpy(row[layer]).to(device)
        y_ez[i, :channels] = torch.from_numpy(row["y_ez"]).to(device)
        y_nez[i, :channels] = torch.from_numpy(row["y_nez"]).to(device)
        mask[i, :channels] = True
    return x, y_nez, y_ez, mask


def fit_linear(rows: dict[str, dict], layer: str, fold: int, source_epoch: int, device) -> nn.Linear:
    x, y_nez, y_ez, mask = padded_fit(rows, layer, device)
    seed = 42 + fold * 1000 + source_epoch * 10 + LAYERS.index(layer)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = nn.Linear(x.shape[-1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    head.train()
    for _epoch in range(20):
        logits = head(x).squeeze(-1)
        loss = patient_equal_weighted_bce_loss(logits, y_nez, y_ez, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def ranking_metrics(subject: str, row: dict, logits: np.ndarray) -> dict:
    score_nez = torch.sigmoid(torch.from_numpy(logits.astype(np.float32))).numpy()
    record = {"subject_id": subject, "labels": row["y_nez"], "labels_nez": row["y_nez"],
              "labels_ez": row["y_ez"], "channel_mask": np.ones(len(logits), dtype=bool),
              "score_nez": score_nez, "score_ez": 1.0 - score_nez}
    return patient_grid(record)["fixed"]


def run_fold(exp, split) -> None:
    fold, train_set, train_loader, val_loader, test_loader, _norm = build_fold(exp, split, "validation")
    assert_no_outer_loader(test_loader)
    output = RUNTIME / "private" / "layerwise" / f"fold_{fold}_patients.csv"
    if output.exists():
        print(f"[LAYERWISE] fold={fold} cached", flush=True)
        return
    selected = {row["subject_id"]: int(row["selected_epoch"]) for row in
                read_csv(RUNTIME / "private" / f"fold_{fold}_A1_SF1_PATIENT.csv")}
    fit_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
    model = exp.runtime["model_cls"](exp.args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, train_loader)
    results = []
    for epoch in sorted(set(selected.values())):
        checkpoint = torch.load(source_checkpoint(fold, epoch), map_location=exp.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        fit, val = embeddings(model, fit_loader, exp.device), embeddings(model, val_loader, exp.device)
        if set(val) != set(selected):
            raise RuntimeError("LAYERWISE validation membership changed")
        heads = {layer: fit_linear(fit, layer, fold, epoch, exp.device) for layer in LAYERS[:3]}
        for subject, chosen in selected.items():
            if chosen != epoch:
                continue
            row = val[subject]
            for layer in LAYERS:
                if layer == "L3":
                    logits = row["L3"]
                else:
                    with torch.no_grad():
                        logits = heads[layer](torch.from_numpy(row[layer]).to(exp.device)).squeeze(-1).cpu().numpy()
                results.append({"fold": fold, "subject_id": subject, "selected_epoch": epoch,
                                "layer": layer, **ranking_metrics(subject, row, logits)})
        print(f"[LAYERWISE] fold={fold} source_epoch={epoch} complete", flush=True)
    if len(results) != 13 * 4:
        raise RuntimeError("Layerwise VLOO case count changed")
    write_csv(output, results)


def aggregate() -> None:
    all_rows = []
    for fold in range(1, 6):
        all_rows.extend(read_csv(RUNTIME / "private" / "layerwise" / f"fold_{fold}_patients.csv"))
    if len(all_rows) != 65 * 4:
        raise RuntimeError("Layerwise audit incomplete")
    public = []
    for fold in range(1, 6):
        for layer in LAYERS:
            subset = [row for row in all_rows if int(row["fold"]) == fold and row["layer"] == layer]
            public.append({"fold": fold, "layer": layer, "n_patients": len(subset),
                           **{metric: mean(subset, metric) for metric in ("patient_ez_auprc", "patient_ez_mrr",
                                                                        "patient_ez_auroc", "top1_is_ez")}})
    write_csv(EXPERIMENT / "diagnostics" / "LAYERWISE_RANK_SEPARABILITY.csv", public)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--aggregate", action="store_true")
    options = parser.parse_args()
    ensure_source()
    if options.aggregate:
        aggregate()
        return
    exp = make_experiment()
    for split in exp.outer_splits:
        if options.fold is None or int(split["fold_idx"]) == options.fold:
            run_fold(exp, split)
    if options.fold is None:
        aggregate()


if __name__ == "__main__":
    main()
