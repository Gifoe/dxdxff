"""Independent, matched thirty-epoch ranking-source interventions; FIT/validation only."""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import (A1_RUNTIME, EXPERIMENT, LOCK_SHA256, RUNTIME, assert_no_outer_loader,
                    build_fold, core, custom_compute_loss, ensure_source, epoch_grid,
                    make_experiment, sha256, write_json)
from objectives import patient_equal_weighted_bce_loss, per_patient_weighted_bce

VARIANTS = ("CTX1", "CTX2", "CTX3", "OBJ1", "CTR1", "VIEW1", "VIEW2")
SEMANTIC_A = (0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 28, 29, 30, 31, 32, 33, 34, 35)
SEMANTIC_B = tuple(range(9, 27))


class TwoBranchEncoder(nn.Module):
    def __init__(self, group_a: tuple[int, ...], group_b: tuple[int, ...], dropout: float):
        super().__init__()
        if len(group_a) != len(group_b) != 18 or sorted((*group_a, *group_b)) != list(range(36)):
            raise RuntimeError("View groups must partition exactly 36 features")
        self.register_buffer("group_a", torch.tensor(group_a, dtype=torch.long), persistent=True)
        self.register_buffer("group_b", torch.tensor(group_b, dtype=torch.long), persistent=True)

        def branch():
            return nn.Sequential(nn.Linear(18, 16), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(16, 16), nn.LayerNorm(16))

        self.branch_a = branch()
        self.branch_b = branch()
        self.merge = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.LayerNorm(32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 36:
            raise RuntimeError("Frozen feature view width changed")
        return self.merge(torch.cat((self.branch_a(x.index_select(-1, self.group_a)),
                                     self.branch_b(x.index_select(-1, self.group_b))), dim=-1))


def no_patient_attention(self, patient_channel_embedding: torch.Tensor, channel_mask: torch.Tensor) -> dict:
    from patient_channel_ranker import _patient_relative_zscore

    z = _patient_relative_zscore(patient_channel_embedding, channel_mask)
    h = self.attn_norm(z)
    logits = self.classifier(h).squeeze(-1).masked_fill(~channel_mask, -1e9)
    score_nez = torch.sigmoid(logits).masked_fill(~channel_mask, 0.0)
    score_ez = (1.0 - score_nez).masked_fill(~channel_mask, 0.0)
    return {"logits": logits, "scores": score_nez, "score_nez": score_nez, "score_ez": score_ez}


def ranking_loss(self, outputs, batch, ez_negative_weight, *, split_name="train"):
    del split_name
    if float(ez_negative_weight) != 2.0:
        raise RuntimeError("A1 class weight changed")
    logits = outputs["logits"]
    bce = patient_equal_weighted_bce_loss(logits, batch["labels"], batch["labels_ez"], batch["channel_mask"])
    losses = []
    for i in range(logits.shape[0]):
        valid = batch["channel_mask"][i]
        ez = logits[i][valid & (batch["labels_ez"][i] > 0.5)]
        nez = logits[i][valid & (batch["labels_ez"][i] < 0.5)]
        if not ez.numel() or not nez.numel():
            raise RuntimeError("Pairwise ranking loss requires both classes per FIT patient")
        losses.append(F.softplus(0.05 + ez[:, None] - nez[None, :]).mean())
    rank = torch.stack(losses).mean()
    loss = bce + 0.05 * rank
    return loss, {"bce": float(bce.detach()), "rank": float(rank.detach())}


def center_loss_factory(weights: dict[str, float]):
    def center_loss(self, outputs, batch, ez_negative_weight, *, split_name="train"):
        if float(ez_negative_weight) != 2.0:
            raise RuntimeError("A1 class weight changed")
        if split_name != "train":
            bce = patient_equal_weighted_bce_loss(outputs["logits"], batch["labels"],
                                                  batch["labels_ez"], batch["channel_mask"])
            return bce, {"bce": float(bce.detach())}
        losses, active = per_patient_weighted_bce(outputs["logits"], batch["labels"],
                                                   batch["labels_ez"], batch["channel_mask"])
        if not bool(active.all()):
            raise RuntimeError("Inactive FIT patient in center-equal batch")
        factors = torch.tensor([weights[str(subject)] for subject in batch["subject_id"]],
                               dtype=losses.dtype, device=losses.device)
        loss = (factors * losses).mean()
        return loss, {"bce": float(loss.detach()), "mean_center_weight": float(factors.mean().detach())}
    return center_loss


def center_weights(train_set) -> tuple[dict[str, float], dict]:
    centers = {str(item["subject_id"]): str(item["center"]) for item in train_set.patient_examples}
    counts = Counter(centers.values())
    n, k = len(centers), len(counts)
    if n != len(train_set) or k < 2:
        raise RuntimeError("FIT center metadata invalid")
    weights = {patient: n / (k * counts[center]) for patient, center in centers.items()}
    if abs(sum(weights.values()) / n - 1.0) > 1e-12:
        raise RuntimeError("FIT center weights not expectation-normalized")
    return weights, {"fit_patients": n, "fit_centers": dict(counts), "weight_formula": "N/(K*N_k)"}


def build_model(exp, train_loader, variant: str, fold: int, initial: dict, view_partition: dict) -> tuple[nn.Module, dict]:
    core._set_random_seed(42 + fold)
    model = exp.runtime["model_cls"](exp.args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, train_loader)
    original_count = sum(parameter.numel() for parameter in model.parameters())
    if variant in ("VIEW1", "VIEW2"):
        if variant == "VIEW1":
            group_a = tuple(view_partition["group_A"])
            group_b = tuple(view_partition["group_B"])
        else:
            group_a, group_b = SEMANTIC_A, SEMANTIC_B
        core._set_random_seed(42 + fold + 700)
        model.b0_encoder.feature_mlp = TwoBranchEncoder(group_a, group_b, float(exp.args.dropout)).to(exp.device)
        target = model.state_dict()
        common = {key: value for key, value in initial.items() if key in target and target[key].shape == value.shape}
        if any(key.startswith("b0_encoder.feature_mlp") for key in common):
            raise RuntimeError("New view encoder accidentally inherited shared source MLP")
        target.update(common)
        model.load_state_dict(target, strict=True)
        if any(not torch.equal(model.state_dict()[key].cpu(), value) for key, value in common.items()):
            raise RuntimeError("Common VIEW parameters not matched to A1 initialization")
        count = sum(parameter.numel() for parameter in model.parameters())
        if abs(count / original_count - 1.0) > 0.02:
            raise RuntimeError(f"VIEW branch parameter count differs >2%: original={original_count}, new={count}")
    else:
        model.load_state_dict(initial, strict=True)
        count = original_count
    if variant in ("CTX1", "CTX3"):
        model.channel_classifier.forward = types.MethodType(no_patient_attention, model.channel_classifier)
    if variant in ("CTX2", "CTX3"):
        model.b0_encoder.use_channel_attention = False
    unused = 0
    if variant in ("CTX1", "CTX3"):
        unused += sum(p.numel() for p in model.channel_classifier.channel_attn.parameters())
    if variant in ("CTX2", "CTX3"):
        unused += sum(p.numel() for p in model.b0_encoder.channel_attn.parameters()) + sum(p.numel() for p in model.b0_encoder.attn_norm.parameters())
    return model, {"total_allocated_parameters": count, "trainable_used_parameters": count - unused,
                   "original_A1_parameters": original_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--variant", choices=VARIANTS)
    options = parser.parse_args()
    ensure_source()
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not source.get("pass"):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    view_partition = json.loads((EXPERIMENT / "view_interference" / "RANDOM_VIEW_PARTITION.json").read_text(encoding="utf-8"))
    exp = make_experiment()
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    for split in exp.outer_splits:
        if options.fold and int(split["fold_idx"]) != options.fold:
            continue
        fold, train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        initial_path = A1_RUNTIME / "initial" / f"fold_{fold}_initial.pt"
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        initial_hash = sha256(initial_path)
        ctr_weights, ctr_meta = center_weights(train_set)
        for variant in ((options.variant,) if options.variant else VARIANTS):
            folder = RUNTIME / "private" / "training" / variant / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "complete.json").is_file():
                print(f"[KEEP] {variant} fold={fold} complete", flush=True)
                continue
            model, counts = build_model(exp, train_loader, variant, fold, initial, view_partition)
            if variant == "OBJ1":
                exp._compute_loss = types.MethodType(ranking_loss, exp)
            elif variant == "CTR1":
                exp._compute_loss = types.MethodType(center_loss_factory(ctr_weights), exp)
            else:
                exp._compute_loss = types.MethodType(custom_compute_loss("A1"), exp)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
            for epoch in range(1, 31):
                checkpoint_path = folder / f"epoch_{epoch:02d}.pt"
                grid_path = folder / f"epoch_{epoch:02d}_validation_grid.json"
                if checkpoint_path.is_file():
                    saved = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
                    if (saved["variant"], saved["fold"], saved["epoch"], saved["lock_sha256"], saved["initial_sha256"]) != (variant, fold, epoch, LOCK_SHA256, initial_hash):
                        raise RuntimeError("Training resume provenance changed")
                    if not np.array_equal(saved["normalizer_mean"], normalizer.mean) or not np.array_equal(saved["normalizer_std"], normalizer.std):
                        raise RuntimeError("Training resume normalizer changed")
                    model.load_state_dict(saved["model_state_dict"], strict=True)
                    optimizer.load_state_dict(saved["optimizer_state_dict"])
                    metrics = saved["train_metrics"]
                else:
                    exp.current_epoch = epoch
                    core._set_random_seed(42 * 100000 + fold * 1000 + epoch)
                    metrics = exp._train_one_epoch(model, train_loader, optimizer, ez_weight)
                    if not math.isfinite(float(metrics["loss"])):
                        raise RuntimeError("Nonfinite training loss")
                    temporary = checkpoint_path.with_suffix(".pt.tmp")
                    torch.save({"variant": variant, "fold": fold, "epoch": epoch, "lock_sha256": LOCK_SHA256,
                                "initial_sha256": initial_hash, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "train_metrics": metrics,
                                "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std}, temporary)
                    temporary.replace(checkpoint_path)
                if not grid_path.is_file():
                    _, _, records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
                    write_json(grid_path, epoch_grid(records, epoch))
                print(f"[{variant}] fold={fold} epoch={epoch}/30 loss={metrics['loss']:.6f}", flush=True)
            write_json(folder / "complete.json", {"variant": variant, "fold": fold, "epochs": 30,
                       "parameter_counts": counts, "fit_center_weights": ctr_meta if variant == "CTR1" else None,
                       "outer_test_accessed": False})
            print(f"[DONE] {variant} fold={fold} all 30 epochs", flush=True)


if __name__ == "__main__":
    main()
