"""Patient-level 5-way cross-fit of historical B0 entirely inside each outer-fit split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def member_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def refit_fixed_epochs(model_class, fit_x: np.ndarray, fit_y: np.ndarray, subjects: np.ndarray,
                       *, seed: int, epochs: int, device: torch.device) -> torch.nn.Module:
    """Original B0 patient-batch AdamW schedule, with nested-selected epoch fixed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model_class(fit_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = max(int(fit_y.sum()), 1)
    negative = max(int((1 - fit_y).sum()), 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    rows = [np.flatnonzero(subjects == patient) for patient in np.unique(subjects)]
    for epoch in range(1, epochs + 1):
        model.train()
        for position in np.random.default_rng(seed + epoch).permutation(len(rows)):
            indexes = rows[int(position)]
            x = torch.as_tensor(fit_x[indexes], dtype=torch.float32, device=device)
            y = torch.as_tensor(fit_y[indexes], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "feature-table", "split-manifest", "feature-manifest", "b0-embeddings", "protocol-lock", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    a = p.parse_args()
    lock = json.loads(a.protocol_lock.read_text(encoding="utf-8"))
    if lock["seed"] != 42 or lock["cross_fit"]["patient_folds"] != 5 or lock["outer_folds"] != [1, 2, 3, 4, 5]:
        raise RuntimeError("Scientific cross-fit protocol lock changed")
    lock_sha256 = digest(a.protocol_lock)
    sys.path.insert(0, str(a.source_root.resolve()))
    from task1_baselines.patient_controls import _ChannelMLP, _Preprocessor, _fit_torch, feature_columns
    torch.set_num_threads(2)
    device = torch.device("cpu")
    table = pd.read_pickle(a.feature_table)
    expected = json.loads(a.feature_manifest.read_text(encoding="utf-8"))
    columns = feature_columns(table)
    if columns != expected["feature_names"] or len(columns) != 88:
        raise RuntimeError("Feature order is not exact historical B0 order")
    split = pd.read_csv(a.split_manifest).rename(columns={"split_role": "partition"})
    outer = split[split.outer_fold == a.fold]
    fit_ids = sorted(outer.loc[outer.partition == "fit", "subject_id"].astype(str).unique())
    forbidden = set(outer.loc[outer.partition.isin(["validation", "test"]), "subject_id"].astype(str))
    if len(fit_ids) not in (50, 51, 52) or set(fit_ids) & forbidden:
        raise RuntimeError("Frozen outer membership invalid")
    fit_table = table[table.subject_id.astype(str).isin(fit_ids)].reset_index(drop=True)
    if set(fit_table.subject_id.astype(str)) != set(fit_ids):
        raise RuntimeError("Feature table does not cover outer-fit patients")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = a.output_dir / f"fold{a.fold}"
    fold_dir.mkdir(exist_ok=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42000 + a.fold)
    cells = []
    for inner, (train_positions, heldout_positions) in enumerate(kfold.split(fit_ids), start=1):
        pool_ids = [fit_ids[int(j)] for j in train_positions]
        heldout_ids = [fit_ids[int(j)] for j in heldout_positions]
        if set(pool_ids) & set(heldout_ids) or set(pool_ids + heldout_ids) != set(fit_ids):
            raise RuntimeError("Cross-fit patient disjointness failed")
        cell = fold_dir / f"inner{inner}"
        complete_file = cell / "COMPLETE.json"
        oof_file = cell / "OOF_PRIVATE.npz"
        if complete_file.exists():
            saved = json.loads(complete_file.read_text(encoding="utf-8"))
            if (saved["heldout_membership_sha256"] != member_hash(heldout_ids)
                    or digest(oof_file) != saved["oof_sha256"] or saved["protocol_lock_sha256"] != lock_sha256):
                raise RuntimeError(f"Resume integrity failed: fold{a.fold}/inner{inner}")
            cells.append((saved, oof_file))
            print(f"SKIP_COMPLETE fold={a.fold} inner={inner}", flush=True)
            continue
        if cell.exists() and any(cell.iterdir()):
            raise RuntimeError(f"Incomplete nonempty cell; preserve and inspect {cell}")
        cell.mkdir(exist_ok=True)
        nested_seed = 42 + 1009 * a.fold + 97 * inner
        shuffled = np.random.default_rng(420000 + 1009 * a.fold + inner).permutation(pool_ids)
        nested_count = max(6, math.ceil(0.2 * len(pool_ids)))
        nested_val_ids = set(shuffled[:nested_count])
        nested_train_ids = set(shuffled[nested_count:])
        if nested_val_ids & set(heldout_ids) or nested_train_ids & set(heldout_ids):
            raise RuntimeError("Heldout patient entered nested selection")
        nested_fit = fit_table[fit_table.subject_id.astype(str).isin(nested_train_ids)].reset_index(drop=True)
        nested_val = fit_table[fit_table.subject_id.astype(str).isin(nested_val_ids)].reset_index(drop=True)
        pre = _Preprocessor(SimpleImputer(), StandardScaler(), True)
        nx = pre.fit_transform(nested_fit, columns)
        vx = pre.transform(nested_val, columns)
        np.random.seed(nested_seed)
        torch.manual_seed(nested_seed)
        selected_model, selected_threshold, selected_epoch, _ = _fit_torch(
            _ChannelMLP(88), nx, nested_fit.label_nez.to_numpy(int), nested_fit.subject_id.to_numpy(str),
            nested_val, vx, device=device, seed=nested_seed, max_epochs=30, patience=6,
            checkpoint_path=None, rank_scores=False)
        del selected_model
        pool = fit_table[fit_table.subject_id.astype(str).isin(pool_ids)].reset_index(drop=True)
        heldout = fit_table[fit_table.subject_id.astype(str).isin(heldout_ids)].reset_index(drop=True)
        final_pre = _Preprocessor(SimpleImputer(), StandardScaler(), True)
        px = final_pre.fit_transform(pool, columns)
        hx = final_pre.transform(heldout, columns)
        final_model = refit_fixed_epochs(_ChannelMLP, px, pool.label_nez.to_numpy(int),
                                         pool.subject_id.to_numpy(str), seed=nested_seed,
                                         epochs=selected_epoch, device=device)
        with torch.no_grad():
            hidden = final_model.network[:-1](torch.from_numpy(hx))
            logit_nez = final_model.network[-1](hidden).squeeze(-1)
        arrays = {"subject_id": heldout.subject_id.astype(str).to_numpy(dtype=str),
                  "channel_name": heldout.channel_name.astype(str).to_numpy(dtype=str),
                  "label_nez": heldout.label_nez.to_numpy(np.int8),
                  "valid_seizure_count": heldout.valid_seizure_count.to_numpy(np.float32),
                  "logit_ez": (-logit_nez).numpy().astype(np.float32),
                  "hidden": hidden.numpy().astype(np.float32)}
        if len(np.unique(arrays["subject_id"])) != len(heldout_ids) or len(arrays["subject_id"]) != len(heldout):
            raise RuntimeError("Cross-fit heldout prediction count changed")
        np.savez_compressed(oof_file, **arrays)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
                    "selected_epoch": selected_epoch, "seed": nested_seed}, cell / "REFIT_B0_PRIVATE.pt")
        private = {"outer_fold": a.fold, "inner_fold": inner, "nested_train_patient_ids": sorted(nested_train_ids),
                   "nested_validation_patient_ids": sorted(nested_val_ids), "refit_train_patient_ids": sorted(pool_ids),
                   "heldout_patient_ids": sorted(heldout_ids)}
        (cell / "MEMBERSHIP_PRIVATE.json").write_text(json.dumps(private, indent=2), encoding="utf-8")
        saved = {"outer_fold": a.fold, "inner_fold": inner, "nested_train_patients": len(nested_train_ids),
                 "nested_validation_patients": len(nested_val_ids), "refit_train_patients": len(pool_ids),
                 "heldout_patients": len(heldout_ids), "heldout_channels": len(heldout),
                 "heldout_membership_sha256": member_hash(heldout_ids),
                 "refit_membership_sha256": member_hash(pool_ids),
                 "nested_validation_macro_f1": float(selected_threshold.patient_macro_f1),
                 "nested_selected_epoch": selected_epoch, "oof_sha256": digest(oof_file),
                 "model_excluded_heldout_patients": True, "protocol_lock_sha256": lock_sha256}
        complete_file.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
        cells.append((saved, oof_file))
        print("CELL_COMPLETE " + json.dumps(saved), flush=True)
    if len(cells) != 5:
        raise RuntimeError("Expected five cross-fit cells")
    outputs = {key: [] for key in ("subject_id", "channel_name", "label_nez", "valid_seizure_count", "logit_ez", "hidden")}
    for saved, path in cells:
        with np.load(path, allow_pickle=False) as archive:
            for key in outputs:
                outputs[key].append(archive[key])
    combined = {key: np.concatenate(parts, axis=0) for key, parts in outputs.items()}
    covered = Counter(combined["subject_id"].astype(str))
    expected_counts = fit_table.groupby("subject_id").size().to_dict()
    if covered != expected_counts or len(covered) != len(fit_ids):
        raise RuntimeError("Each outer-fit patient must have exactly one OOF channel set")
    combined_path = fold_dir / "FOLD_OOF_PRIVATE.npz"
    if combined_path.exists():
        with np.load(combined_path, allow_pickle=False) as prior:
            if not np.array_equal(prior["subject_id"], combined["subject_id"]) or not np.allclose(prior["logit_ez"], combined["logit_ez"]):
                raise RuntimeError("Prior combined OOF differs")
    else:
        np.savez_compressed(combined_path, **combined)
    with np.load(a.b0_embeddings / f"fold{a.fold}.npz", allow_pickle=True) as previous:
        insample = previous["fit_logit_ez"].astype(float)
        if set(previous["fit_subject_id"].astype(str)) != set(fit_ids):
            raise RuntimeError("Historical in-sample fit membership differs")
    oof = combined["logit_ez"].astype(float)
    fold_audit = {"outer_fold": a.fold, "fit_patients": len(fit_ids), "fit_channels": len(oof),
                  "protocol_lock_sha256": lock_sha256,
                  "crossfit_cells": [saved for saved, _ in cells], "fit_membership_sha256": member_hash(fit_ids),
                  "combined_oof_sha256": digest(combined_path),
                  "oof_logit_mean": float(oof.mean()), "oof_logit_std": float(oof.std()),
                  "oof_logit_q10": float(np.quantile(oof, 0.1)), "oof_logit_q90": float(np.quantile(oof, 0.9)),
                  "insample_logit_mean": float(insample.mean()), "insample_logit_std": float(insample.std()),
                  "insample_logit_q10": float(np.quantile(insample, 0.1)),
                  "insample_logit_q90": float(np.quantile(insample, 0.9)),
                  "heldout_from_every_predictor": True,
                  "preprocessing": "patient z-score per patient; imputer and StandardScaler fitted on nested-fit for selection, all inner-train for refit"}
    (fold_dir / "FOLD_AUDIT.json").write_text(json.dumps(fold_audit, indent=2, sort_keys=True), encoding="utf-8")
    print("FOLD_CROSSFIT_PASS " + json.dumps({k: v for k, v in fold_audit.items() if k != "crossfit_cells"}), flush=True)


if __name__ == "__main__":
    main()
