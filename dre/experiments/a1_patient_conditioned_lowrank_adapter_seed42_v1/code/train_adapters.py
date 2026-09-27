"""Resume-safe 20-epoch P1/P2/P3 training on frozen, private A1 R4 arrays."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from adapter import FrozenR4Adapter, assert_identity
from common import (ROOT, RUNTIME, checkpoint, epoch_grid, finalize_fold, preflight,
                    write_csv, write_json, patient_equal_weighted_bce_loss)

VARIANTS = ("P1", "P2", "P3")


def stable_seed(*parts) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def load_source(fold, epoch, device):
    path = RUNTIME / "private" / f"fold_{fold}" / f"source_epoch_{epoch:02d}.pkl"
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload["fold"] != fold or payload["epoch"] != epoch or payload["error"] > 1e-6:
        raise RuntimeError("Source R4 cache identity differs")
    dim = int(payload["dim"])
    source = torch.load(checkpoint(fold, epoch), map_location="cpu", weights_only=False)["model_state_dict"]
    classifier = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.GELU(), torch.nn.Dropout(.4), torch.nn.Linear(dim, 1))
    prefix = "channel_classifier.classifier."
    classifier.load_state_dict({key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)}, strict=True)
    classifier.to(device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return payload, classifier


def teacher_scores(fold, epoch, fit_rows):
    target = RUNTIME / "private" / f"fold_{fold}" / f"teacher_epoch_{epoch:02d}.pkl"
    if target.exists():
        with target.open("rb") as stream:
            teachers = pickle.load(stream)
        if set(teachers) != set(fit_rows):
            raise RuntimeError("FIT teacher cache membership differs")
        return teachers
    teachers = {}
    for subject, row in fit_rows.items():
        y = np.asarray(row["y_ez"], dtype=np.int8)
        if set(np.unique(y)) != {0, 1}:
            raise RuntimeError("FIT teacher requires two classes")
        model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        model.fit(row["h"], y)
        score = model.decision_function(row["h"])
        teachers[subject] = ((score - score.mean()) / max(float(score.std()), 1e-6)).astype(np.float32)
    tmp = target.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(teachers, stream, protocol=5)
    tmp.replace(target)
    return teachers


def tensors(rows, device):
    return {subject: {"h": torch.tensor(row["h"], dtype=torch.float32, device=device),
                      "y_ez": torch.tensor(row["y_ez"], dtype=torch.float32, device=device)}
            for subject, row in rows.items()}


def make_batch(subjects, rows, teachers, fold, base_epoch, adapter_epoch, device):
    max_channels = max(rows[subject]["h"].shape[0] for subject in subjects)
    dim = rows[subjects[0]]["h"].shape[1]
    b = len(subjects)
    h = torch.zeros((b, max_channels, dim), dtype=torch.float32, device=device)
    y_ez = torch.zeros((b, max_channels), dtype=torch.float32, device=device)
    valid = torch.zeros((b, max_channels), dtype=torch.bool, device=device)
    context = valid.clone()
    query = valid.clone()
    t = torch.zeros((b, max_channels), dtype=torch.float32, device=device) if teachers is not None else None
    for i, subject in enumerate(subjects):
        n = rows[subject]["h"].shape[0]
        if n < 4:
            raise RuntimeError("Patient cannot supply at least two context and two query channels")
        k = int(np.clip(round(.7 * n), 2, n - 2))
        rng = np.random.default_rng(stable_seed(42, fold, base_epoch, adapter_epoch, subject))
        permutation = rng.permutation(n)
        h[i, :n] = rows[subject]["h"]
        y_ez[i, :n] = rows[subject]["y_ez"]
        valid[i, :n] = True
        context[i, permutation[:k]] = True
        query[i, permutation[k:]] = True
        if t is not None:
            t[i, :n] = torch.tensor(teachers[subject], dtype=torch.float32, device=device)
    if torch.any(context.sum(1) < 2) or torch.any(query.sum(1) < 2) or torch.any(context & query):
        raise RuntimeError("Label-blind context/query split invalid")
    return h, y_ez, valid, context, query, t


def batch_loss(adapter, classifier, batch, variant):
    h, y_ez, valid, context, query, teacher = batch
    logits, a, _ = adapter(h, h, context, classifier)
    bce = patient_equal_weighted_bce_loss(logits, 1.0 - y_ez, y_ez, query)
    identity = a.square().sum(dim=-1).mean()
    geometry = logits.new_zeros(())
    if variant == "P3":
        student = -logits
        valid_float = valid.to(student.dtype)
        count = valid_float.sum(dim=1).clamp_min(1)
        mu = (student * valid_float).sum(dim=1) / count
        variance = (((student - mu[:, None]) * valid_float) ** 2).sum(dim=1) / count
        standardized = (student - mu[:, None]) / torch.clamp(torch.sqrt(variance)[:, None], min=1e-6)
        point = torch.nn.functional.smooth_l1_loss(standardized, teacher, reduction="none")
        geometry = ((point * query).sum(dim=1) / query.sum(dim=1)).mean()
    loss = bce + (0.1 * geometry if variant == "P3" else 0.0) + 1e-3 * identity
    return loss, float(bce.detach()), float(geometry.detach()), float(identity.detach())


def predict(adapter, classifier, rows, device):
    adapter.eval()
    output = {}
    with torch.no_grad():
        for subject, row in rows.items():
            h = row["h"].unsqueeze(0)
            mask = torch.ones((1, h.shape[1]), dtype=torch.bool, device=device)
            logits, a, adapted = adapter(h, h, mask, classifier)
            source = classifier(h).squeeze(-1)
            output[subject] = {"nez_logit": logits[0].cpu().numpy().astype(np.float32),
                               "y_ez": row["y_ez"].cpu().numpy().astype(np.int8),
                               "a": a[0].cpu().numpy().astype(np.float32),
                               "relative_change": float(torch.linalg.vector_norm(adapted - h, dim=-1).div(torch.linalg.vector_norm(h, dim=-1).clamp_min(1e-12)).mean()),
                               "cosine": float(torch.nn.functional.cosine_similarity(adapted, h, dim=-1).mean()),
                               "source_nez_logit": source[0].cpu().numpy().astype(np.float32)}
    return output


def grid_from_predictions(rows, epoch):
    records = []
    for subject, row in rows.items():
        logit = torch.from_numpy(row["nez_logit"])
        score_nez = torch.sigmoid(logit).numpy()
        y_ez = row["y_ez"].astype(np.float32)
        records.append({"subject_id": subject, "channel_mask": np.ones(len(y_ez), dtype=bool),
                        "labels": 1 - y_ez, "labels_nez": 1 - y_ez, "labels_ez": y_ez,
                        "score_nez": score_nez, "score_ez": 1 - score_nez})
    return epoch_grid(records, epoch)


def atomic_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_one(fold, epoch, variant, payload, classifier, device):
    outdir = RUNTIME / "private" / f"fold_{fold}" / variant / f"source_epoch_{epoch:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    grid_path = outdir / "validation_grid.json"
    pred_path = outdir / "validation_predictions.pkl"
    state_path = outdir / "latest_adapter.pt"
    if grid_path.exists() and pred_path.exists() and state_path.exists():
        return json.loads(grid_path.read_text(encoding="utf-8"))
    torch.manual_seed(42 + fold * 1000 + epoch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + fold * 1000 + epoch)
    adapter = FrozenR4Adapter(payload["dim"], variant).to(device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    fit, val = tensors(payload["fit"], device), tensors(payload["val"], device)
    example = next(iter(fit.values()))["h"].unsqueeze(0)
    mask = torch.ones((1, example.shape[1]), dtype=torch.bool, device=device)
    identity_error = assert_identity(adapter, classifier, example, mask)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    completed = 0
    if state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if (saved["fold"], saved["source_epoch"], saved["variant"]) != (fold, epoch, variant):
            raise RuntimeError("Adapter resume identity mismatch")
        adapter.load_state_dict(saved["adapter"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed = int(saved["adapter_epoch"])
    teachers = teacher_scores(fold, epoch, payload["fit"]) if variant == "P3" else None
    subjects = sorted(fit)
    for adapter_epoch in range(completed + 1, 21):
        adapter.train()
        order = np.random.default_rng(stable_seed(42, fold, epoch, adapter_epoch, "patient_order")).permutation(subjects).tolist()
        loss_values = []
        for start in range(0, len(order), 8):
            group = order[start:start + 8]
            batch = make_batch(group, fit, teachers, fold, epoch, adapter_epoch, device)
            loss, bce, geo, identity = batch_loss(adapter, classifier, batch, variant)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite adapter loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            loss_values.append(float(loss.detach()))
        atomic_torch(state_path, {"fold": fold, "source_epoch": epoch, "variant": variant,
                                  "adapter_epoch": adapter_epoch, "adapter": adapter.state_dict(),
                                  "optimizer": optimizer.state_dict(), "initial_identity_error": identity_error,
                                  "last_loss": float(np.mean(loss_values))})
        if adapter_epoch in (1, 10, 20):
            print(f"[TRAIN] fold={fold} source={epoch} {variant} adapter={adapter_epoch}/20 loss={np.mean(loss_values):.5f}", flush=True)
    val_predictions = predict(adapter, classifier, val, device)
    grid = grid_from_predictions(val_predictions, epoch)
    tmp = pred_path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(val_predictions, stream, protocol=5)
    tmp.replace(pred_path)
    write_json(grid_path, grid)
    return grid


def run_fold(fold, variants, device, source_epoch=None):
    folder = RUNTIME / "private" / f"fold_{fold}"
    for epoch in ([source_epoch] if source_epoch is not None else range(1, 31)):
        payload, classifier = load_source(fold, epoch, device)
        for variant in variants:
            train_one(fold, epoch, variant, payload, classifier, device)
    if source_epoch is not None:
        return
    for variant in variants:
        grids = [json.loads((folder / variant / f"source_epoch_{epoch:02d}" / "validation_grid.json").read_text(encoding="utf-8")) for epoch in range(1, 31)]
        public, fullval = finalize_fold(grids, variant, fold, folder / f"{variant}_VLOO_PRIVATE.csv")
        write_json(folder / f"{variant}_fold_summary.json", {"vloo": public, "fullval": fullval})
        print(f"[VLOO] fold={fold} {variant} MacroF1={public['patient_macro_f1']:.6f} AP={public['patient_ez_auprc']:.6f}", flush=True)


def aggregate():
    for variant, subfolder in (("P1", "p1_global"), ("P2", "p2_conditioned"), ("P3", "p3_geometry_teacher")):
        rows = [json.loads((RUNTIME / "private" / f"fold_{fold}" / f"{variant}_fold_summary.json").read_text(encoding="utf-8"))["vloo"] for fold in range(1, 6)]
        write_csv(ROOT / subfolder / f"{variant}_VLOO_BY_FOLD.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--source-epoch", type=int, choices=range(1, 31))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--aggregate", action="store_true")
    opts = parser.parse_args()
    if opts.source_epoch is not None and opts.fold is None:
        parser.error("--source-epoch requires --fold")
    preflight()
    if opts.aggregate:
        aggregate()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(1, 6):
        if opts.fold is None or opts.fold == fold:
            run_fold(fold, opts.variants, device, opts.source_epoch)
    if opts.fold is None and opts.source_epoch is None and set(opts.variants) == set(VARIANTS):
        aggregate()


if __name__ == "__main__":
    main()
