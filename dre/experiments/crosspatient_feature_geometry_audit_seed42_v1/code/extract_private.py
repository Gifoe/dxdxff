"""Extract frozen FIT/validation representations; writes only to private runtime."""
from __future__ import annotations

import argparse
import hashlib
import pickle

import numpy as np
import torch

from common import (RUNTIME, assert_development_split, build_fold, core, ensure_source,
                    make_experiment, selected_epochs, source_checkpoint, write_json)
from patient_channel_ranker import _patient_relative_zscore


def raw_channel_aggregate(example):
    """Source 36D view: window mean within seizure; mean/std across available seizures."""
    mask = np.asarray(example["channel_mask"], dtype=bool)
    features = [[] for _ in range(len(mask))]
    for seizure, present, windows in zip(example["b0_features"], example["seizure_channel_mask"], example["window_mask"], strict=True):
        value = np.asarray(seizure, dtype=np.float32)[np.asarray(windows, dtype=bool)]
        for channel in np.flatnonzero(mask & np.asarray(present, dtype=bool)):
            features[channel].append(value[:, channel].mean(axis=0))
    mean, meanstd = [], []
    for channel in np.flatnonzero(mask):
        if not features[channel]:
            raise RuntimeError("Valid channel has no source seizure features")
        per_seizure = np.stack(features[channel])
        mean.append(per_seizure.mean(axis=0))
        meanstd.append(np.r_[per_seizure.mean(axis=0), per_seizure.std(axis=0)])
    return np.asarray(mean, dtype=np.float32), np.asarray(meanstd, dtype=np.float32)


def frozen_embeddings(model, loader, device):
    captured = {}
    def pre(_module, inputs):
        captured["R4"] = inputs[0].detach()
    hook = model.channel_classifier.classifier.register_forward_pre_hook(pre)
    result = {}
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                out = model(core._move_tensors_to_device(batch, device))
                if "R4" not in captured:
                    raise RuntimeError("Missing A1 contextual preclassifier hook")
                mask = batch["channel_mask"].to(device)
                r2 = out["patient_channel_embedding"]
                r3 = _patient_relative_zscore(r2, mask)
                r4 = captured["R4"]
                replay = model.channel_classifier.classifier(r4).squeeze(-1)
                if not torch.allclose(replay[mask], out["logits"][mask], atol=1e-6, rtol=0):
                    raise RuntimeError("A1 frozen logit replay failed")
                for i, subject in enumerate(batch["subject_id"]):
                    index = torch.nonzero(mask[i]).squeeze(1)
                    result[str(subject)] = {
                        "R2": r2[i].index_select(0, index).cpu().numpy().astype(np.float32),
                        "R3": r3[i].index_select(0, index).cpu().numpy().astype(np.float32),
                        "R4": r4[i].index_select(0, index).cpu().numpy().astype(np.float32),
                        "R5": (-out["logits"][i].index_select(0, index)).cpu().numpy().astype(np.float32),
                        "y": batch["labels_ez"][i][batch["channel_mask"][i]].numpy().astype(np.int8),
                    }
    finally:
        hook.remove()
    return result


def same_bytes(a, b):
    for subject in a:
        for key in ("R2", "R3", "R4", "R5", "y"):
            if not np.array_equal(a[subject][key], b[subject][key]):
                return False
    return True


def run_fold(exp, split):
    fold, train_set, train_loader, val_loader, test_loader, _norm = build_fold(exp, split, "validation")
    assert_development_split(test_loader)
    selected = selected_epochs(fold)
    if set(selected) != set(split["validation_subjects"]):
        raise RuntimeError("A1 selection and validation split mismatch")
    if set(split["fit_subjects"]) & set(selected):
        raise RuntimeError("FIT/validation patient overlap")
    folder = RUNTIME / "private" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "raw.pkl"
    if not raw_path.exists():
        from exp_ez_hybrid import flatten_window_samples
        fit_samples = flatten_window_samples(exp.run_records, subject_ids=split["fit_subjects"])
        val_samples = flatten_window_samples(exp.run_records, subject_ids=split["validation_subjects"])
        make = exp.runtime["build_patient_examples"]
        fit_examples = make(fit_samples, exp.patient_index, normalizer=None, subject_ids=split["fit_subjects"], args=exp.args)
        val_examples = make(val_samples, exp.patient_index, normalizer=None, subject_ids=split["validation_subjects"], args=exp.args)
        raw = {}
        for role, examples in (("fit", fit_examples), ("val", val_examples)):
            raw[role] = {}
            for example in examples:
                r0, r1 = raw_channel_aggregate(example)
                labels = np.asarray(example["labels_ez"])[np.asarray(example["channel_mask"], dtype=bool)].astype(np.int8)
                raw[role][example["subject_id"]] = {"R0": r0, "R1": r1, "y": labels}
        with raw_path.open("wb") as stream:
            pickle.dump(raw, stream, protocol=5)
    fit_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
    model = exp.runtime["model_cls"](exp.args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, train_loader)
    hashes = []
    for epoch in sorted(set(selected.values())):
        path = folder / f"epoch_{epoch:02d}.pkl"
        if path.exists():
            print(f"[EXTRACT] fold={fold} epoch={epoch} cached", flush=True)
            continue
        checkpoint = torch.load(source_checkpoint(fold, epoch), map_location=exp.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        fit = frozen_embeddings(model, fit_loader, exp.device)
        val = frozen_embeddings(model, val_loader, exp.device)
        own = next(subject for subject in val if selected[subject] == epoch)
        second = frozen_embeddings(model, val_loader, exp.device)
        if not same_bytes({own: val[own]}, {own: second[own]}):
            raise RuntimeError("Repeated representation extraction differed")
        digest = hashlib.sha256(val[own]["R4"].tobytes()).hexdigest()
        with path.open("wb") as stream:
            pickle.dump({"fit": fit, "val": val, "fold": fold, "epoch": epoch, "hash": digest}, stream, protocol=5)
        hashes.append({"epoch": epoch, "repeat_identical": True})
        print(f"[EXTRACT] fold={fold} epoch={epoch} complete", flush=True)
    write_json(folder / "extraction_status.json", {"fold": fold, "selected_epochs": sorted(set(selected.values())), "new_hash_checks": hashes, "outer_loader": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    options = parser.parse_args()
    ensure_source()
    exp = make_experiment()
    for split in exp.outer_splits:
        if options.fold is None or int(split["fold_idx"]) == options.fold:
            run_fold(exp, split)


if __name__ == "__main__":
    main()
