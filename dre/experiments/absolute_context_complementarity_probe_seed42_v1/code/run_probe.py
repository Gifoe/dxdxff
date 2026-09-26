"""Extract frozen A1 fit/validation features and train matched residual probes.

All patient-level tensors and probe states are private. No outer loader is built.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import epoch_grid  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402

LOCK_SHA256 = "76a9a94cbf364878088ed89370e062e6fe92687adafcd4f95b923e67ce1a36a3"
RUNTIME = Path(os.environ.get("ABS_CONTEXT_RUNTIME", ""))
A1_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def compare_grid(current: dict, original: dict) -> float:
    if [r["subject_id"] for r in current["patients"]] != [r["subject_id"] for r in original["patients"]]:
        raise RuntimeError("Validation patient order changed")
    maximum = 0.0
    for new, old in zip(current["patients"], original["patients"], strict=True):
        if new["n_channels"] != old["n_channels"]:
            raise RuntimeError("Validation channel count changed")
        for category in ("grid", "fixed"):
            if set(new[category]) != set(old[category]):
                raise RuntimeError("Validation metric fields changed")
            for metric in new[category]:
                maximum = max(maximum, float(np.max(np.abs(np.asarray(new[category][metric]) -
                                                       np.asarray(old[category][metric])))))
    return maximum


class ResidualHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Dropout(0.10), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.tanh(self.net(x).squeeze(-1))


def extract_split(exp, model, loader) -> list[dict]:
    classifier = model.channel_classifier.classifier
    captured = []

    def capture_input(_module, inputs):
        captured.append(inputs[0].detach())

    hook = classifier.register_forward_pre_hook(capture_input)
    rows = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                outputs = model(core._move_tensors_to_device(batch, exp.device))
                if len(captured) != 1 or "patient_channel_embedding" not in outputs:
                    raise RuntimeError("Could not capture exact pre-classifier relative embedding")
                h = outputs["patient_channel_embedding"].detach().cpu().numpy()
                r = captured[0].cpu().numpy()
                logits = outputs["logits"].detach().cpu().numpy()
                p = outputs["score_nez"].detach().cpu().numpy()
                q = outputs["score_ez"].detach().cpu().numpy()
                mask = batch["channel_mask"].numpy()
                if h.shape != r.shape or h.shape[:2] != logits.shape:
                    raise RuntimeError("Frozen A1 representation shapes disagree")
                for i, subject_id in enumerate(batch["subject_id"]):
                    valid = mask[i].astype(bool)
                    y_nez = batch["labels_nez"][i].numpy()[valid].astype(np.float32)
                    y_ez = batch["labels_ez"][i].numpy()[valid].astype(np.float32)
                    rows.append({"subject_id": str(subject_id), "h": h[i][valid].astype(np.float32),
                                 "r": r[i][valid].astype(np.float32), "logits": logits[i][valid].astype(np.float32),
                                 "score_nez": p[i][valid].astype(np.float32),
                                 "score_ez": q[i][valid].astype(np.float32),
                                 "y_nez": y_nez, "y_ez": y_ez})
    finally:
        hook.remove()
    if len(rows) != len({row["subject_id"] for row in rows}):
        raise RuntimeError("Duplicate patient in extracted split")
    return sorted(rows, key=lambda row: row["subject_id"])


def context_for(row: dict) -> np.ndarray:
    h = row["h"].astype(np.float64)
    mu = h.mean(axis=0)
    sigma = np.sqrt(np.mean((h - mu) ** 2, axis=0) + 1e-5)
    return np.concatenate([mu, np.log(sigma + 1e-6)])


def normalize_context(fit: list[dict], validation: list[dict]) -> dict:
    contexts = np.stack([context_for(row) for row in fit])
    mean = contexts.mean(axis=0)
    std = contexts.std(axis=0)
    if not np.isfinite(contexts).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Nonfinite fit absolute context")
    for rows in (fit, validation):
        for row in rows:
            row["context_raw"] = context_for(row)
            row["context_norm"] = ((row["context_raw"] - mean) / (std + 1e-6)).astype(np.float32)
    return {"fit_context_dim": int(contexts.shape[1]), "fit_context_mean_norm": float(np.linalg.norm(mean)),
            "fit_context_std_min": float(std.min()), "fit_context_std_max": float(std.max()),
            "fit_patients": len(fit), "validation_patients": len(validation)}


def make_channel_tensors(rows: list[dict], variant: str, device) -> tuple[torch.Tensor, ...]:
    inputs, logits, y_nez, y_ez, patient_indices = [], [], [], [], []
    for i, row in enumerate(rows):
        context = row["context_norm"] if variant == "P2" else np.zeros_like(row["context_norm"])
        x = np.concatenate([row["r"], np.repeat(context[None, :], len(row["r"]), axis=0)], axis=1)
        inputs.append(x.astype(np.float32))
        logits.append(row["logits"])
        y_nez.append(row["y_nez"])
        y_ez.append(row["y_ez"])
        patient_indices.append(np.full(len(row["r"]), i, dtype=np.int64))
    return tuple(torch.as_tensor(np.concatenate(parts), device=device) for parts in
                 (inputs, logits, y_nez, y_ez, patient_indices))


def fit_probe(fit: list[dict], variant: str, initial_state: dict, pair_seed: int,
              device) -> tuple[ResidualHead, list[float]]:
    torch.manual_seed(pair_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pair_seed)
    input_dim = fit[0]["r"].shape[1] + fit[0]["context_norm"].shape[0]
    head = ResidualHead(input_dim).to(device)
    head.load_state_dict(initial_state, strict=True)
    x, base, y_nez, y_ez, patient_idx = make_channel_tensors(fit, variant, device)
    weights = torch.where(y_ez > 0.5, 2.0, 1.0)
    denominator = torch.zeros(len(fit), device=device).index_add(0, patient_idx, weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _epoch in range(15):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        final = base + head(x)
        channel_loss = F.binary_cross_entropy_with_logits(final, y_nez, reduction="none")
        numerator = torch.zeros(len(fit), device=device).index_add(0, patient_idx, channel_loss * weights)
        loss = (numerator / denominator).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite matched probe loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return head.eval(), losses


def validation_grid(rows: list[dict], head: ResidualHead, variant: str, epoch: int,
                    device) -> tuple[dict, dict]:
    records, deltas = [], []
    with torch.no_grad():
        for row in rows:
            context = row["context_norm"] if variant == "P2" else np.zeros_like(row["context_norm"])
            x = np.concatenate([row["r"], np.repeat(context[None, :], len(row["r"]), axis=0)], axis=1)
            delta = head(torch.as_tensor(x.astype(np.float32), device=device)).cpu().numpy()
            logits = row["logits"] + delta
            p = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)
            q = 1.0 - p
            records.append({"subject_id": row["subject_id"], "labels": row["y_nez"],
                            "labels_nez": row["y_nez"], "labels_ez": row["y_ez"],
                            "score_nez": p, "score_ez": q,
                            "channel_mask": np.ones(len(p), dtype=bool)})
            deltas.append(delta)
    return epoch_grid(records, epoch), {"mean_abs_delta": float(np.mean(np.abs(np.concatenate(deltas)))),
                                         "max_abs_delta": float(np.max(np.abs(np.concatenate(deltas))))}


def p0_grid(rows: list[dict], epoch: int) -> dict:
    return epoch_grid([{"subject_id": row["subject_id"], "labels": row["y_nez"],
                        "labels_nez": row["y_nez"], "labels_ez": row["y_ez"],
                        "score_nez": row["score_nez"], "score_ez": row["score_ez"],
                        "channel_mask": np.ones(len(row["score_nez"]), dtype=bool)} for row in rows], epoch)


def main() -> None:
    if not os.environ.get("ABS_CONTEXT_RUNTIME") or not RUNTIME.is_absolute() or \
            not os.environ.get("A1_A2_RUNTIME") or not A1_RUNTIME.is_absolute():
        raise RuntimeError("Set private runtime directories")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != LOCK_SHA256:
        raise RuntimeError("Probe protocol lock changed")
    assert_sources()
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if source.get("pass") is not True or abs(source["observed_mean"] - 0.6259962097139906) > 1e-6:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    exp = core.Exp_EZHybridLocalization(args)
    completed = 0
    for split in exp.outer_splits:
        fold, train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
        if test_loader is not None:
            raise RuntimeError("Outer-test loader must not be built")
        fit_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
        model = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("A1 backbone is not frozen")
        model.eval()
        for epoch in range(1, 31):
            cell = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}"
            if all((cell / f"{variant}_probe.pt").exists() and (cell / f"{variant}_validation_grid.json").exists()
                   for variant in ("P1", "P2")) and (cell / "representation.pt").exists():
                checkpoint_path = A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"
                source_hash = sha256(checkpoint_path)
                for variant in ("P1", "P2"):
                    saved = torch.load(cell / f"{variant}_probe.pt", map_location="cpu", weights_only=False)
                    if saved["lock_sha256"] != LOCK_SHA256 or saved["source_checkpoint_sha256"] != source_hash or \
                            saved["probe_epochs"] != 15 or saved["variant"] != variant:
                        raise RuntimeError("Saved probe resume provenance changed")
                completed += 2
                print(f"[RESUME] fold={fold} epoch={epoch} both probes complete", flush=True)
                continue
            checkpoint_path = A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
            if checkpoint["variant"] != "A1" or int(checkpoint["fold"]) != fold or int(checkpoint["epoch"]) != epoch:
                raise RuntimeError("A1 checkpoint identity changed")
            if not np.array_equal(normalizer.mean, checkpoint["normalizer_mean"]) or \
                    not np.array_equal(normalizer.std, checkpoint["normalizer_std"]):
                raise RuntimeError("A1 fit normalizer changed")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if (cell / "representation.pt").exists():
                payload = torch.load(cell / "representation.pt", map_location="cpu", weights_only=False)
                if payload["lock_sha256"] != LOCK_SHA256 or payload["source_checkpoint_sha256"] != sha256(checkpoint_path):
                    raise RuntimeError("Private representation cache provenance changed")
                fit, validation, context_diag = payload["fit"], payload["validation"], payload["context_diag"]
            else:
                fit = extract_split(exp, model, fit_loader)
                validation = extract_split(exp, model, val_loader)
                if len(validation) != 13 or len(fit) == 0:
                    raise RuntimeError("Frozen fit/validation roles changed")
                context_diag = normalize_context(fit, validation)
                if any(row["h"].shape[1] != row["r"].shape[1] for row in fit + validation):
                    raise RuntimeError("Pre-z and post-attention embedding dimensions disagree")
                with torch.no_grad():
                    for row in fit[:1] + validation[:1]:
                        r = torch.as_tensor(row["r"], device=exp.device)
                        raw = model.channel_classifier.classifier(r).squeeze(-1).cpu().numpy()
                        if float(np.max(np.abs(raw - row["logits"]))) > 1e-6:
                            raise RuntimeError("Captured relative embedding does not reconstruct A1 logits")
                original = json.loads((A1_RUNTIME / "A1" / f"fold_{fold}" /
                                       f"epoch_{epoch:02d}_validation_grid.json").read_text(encoding="utf-8"))
                if compare_grid(p0_grid(validation, epoch), original) > 1e-8:
                    raise RuntimeError("P0 validation grid changed during feature extraction")
                cell.mkdir(parents=True, exist_ok=True)
                torch.save({"lock_sha256": LOCK_SHA256, "source_checkpoint_sha256": sha256(checkpoint_path),
                            "fit": fit, "validation": validation, "context_diag": context_diag},
                           cell / "representation.pt")
            if not np.isfinite(np.concatenate([row["context_norm"] for row in fit + validation])).all():
                raise RuntimeError("Nonfinite normalized context")
            input_dim = fit[0]["r"].shape[1] + fit[0]["context_norm"].shape[0]
            pair_seed = 42 + 1000 * fold + epoch
            torch.manual_seed(pair_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(pair_seed)
            initial = ResidualHead(input_dim).to(exp.device)
            initial_state = copy.deepcopy(initial.state_dict())
            initial.eval()
            with torch.no_grad():
                for variant in ("P1", "P2"):
                    x, base, *_ = make_channel_tensors(validation, variant, exp.device)
                    if torch.count_nonzero(initial(x)).item() != 0 or not torch.equal(base + initial(x), base):
                        raise RuntimeError("Zero-initialized probe does not exactly reproduce A1")
            del initial
            for variant in ("P1", "P2"):
                output_path = cell / f"{variant}_probe.pt"
                grid_path = cell / f"{variant}_validation_grid.json"
                if output_path.exists() and grid_path.exists():
                    existing = torch.load(output_path, map_location="cpu", weights_only=False)
                    if existing["lock_sha256"] != LOCK_SHA256 or existing["source_checkpoint_sha256"] != sha256(checkpoint_path):
                        raise RuntimeError("Saved probe provenance changed")
                    completed += 1
                    continue
                head, losses = fit_probe(fit, variant, initial_state, pair_seed, exp.device)
                grid, delta_diag = validation_grid(validation, head, variant, epoch, exp.device)
                torch.save({"variant": variant, "fold": fold, "base_epoch": epoch, "probe_epochs": 15,
                            "lock_sha256": LOCK_SHA256, "source_checkpoint_sha256": sha256(checkpoint_path),
                            "model_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            "losses": losses, "context_diag": context_diag,
                            "delta_diag": delta_diag}, output_path)
                write_json(grid_path, grid)
                completed += 1
            print(f"[PROBE] fold={fold} base_epoch={epoch} completed={completed}/300", flush=True)
    write_json(RUNTIME / "private" / "PROBE_TRAINING_COMPLETE.json",
               {"completed_probe_cells": completed, "expected": 300, "source_reproduced": True,
                "outer_test_accessed": False, "lock_sha256": LOCK_SHA256})
    if completed != 300:
        raise RuntimeError("Probe cell count incomplete")


if __name__ == "__main__":
    main()
