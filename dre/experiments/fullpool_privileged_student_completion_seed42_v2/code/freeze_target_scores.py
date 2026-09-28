"""Score all target patients B=0, then hash-freeze before outcome evaluation.

The legacy loader materializes target labels as part of examples; this script
does not index them. No target metric, checkpoint choice, or update occurs.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from student_core import LOCK_SHA, VARIANTS, atomic_json, imports, sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fit_audit = json.loads((a.output / "FIT_SELECTION_FREEZE_AUDIT.json").read_text(encoding="utf-8"))
    if fit_audit["lock_sha"] != LOCK_SHA or not fit_audit["all_models_fit_selected_before_target_outcomes"]:
        raise RuntimeError("All FIT selections must be frozen before target scoring")
    variants = list(VARIANTS[:5]) + ([VARIANTS[5]] if fit_audit["D4_FIT_ELIGIBLE"] else [])
    if fit_audit["n_fold_models"] != 5 * len(variants):
        raise RuntimeError("FIT model count mismatch")
    core, _, _, make_args, _, move = imports()
    manifest = {}
    for fold in range(1, 6):
        args = make_args("R0", a.runtime / "target_scratch" / f"fold_{fold}")
        exp = core.Exp_EZHybridLocalization(args)
        split = next(x for x in exp.outer_splits if int(x["fold_idx"]) == fold)
        fit, val = list(split["fit_subjects"]), list(split["validation_subjects"])
        if len(val) != 13 or set(fit) & set(val):
            raise RuntimeError("Target fold membership changed")
        train_set, val_set, _, normalizer = exp._build_datasets(fit, val, [])
        train_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
        val_loader = exp._make_loader(val_set, shuffle=False, batch_size=2)
        for variant in variants:
            selected = json.loads((a.runtime / "private" / variant / f"fold_{fold}" /
                                   "FIT_SELECTION.json").read_text(encoding="utf-8"))
            if selected["lock_sha"] != LOCK_SHA or selected["fold"] != fold:
                raise RuntimeError("FIT-selected Student identity changed")
            chosen = selected["selected"]
            checkpoint = (a.runtime / "private" / variant / f"fold_{fold}" /
                          chosen["candidate_id"] / f"epoch_{chosen['best_epoch']:02d}.pt")
            state = torch.load(checkpoint, map_location=exp.device, weights_only=False)
            if (state["lock_sha"] != LOCK_SHA or state["fold"] != fold or
                    state["variant"] != variant or state["epoch"] != chosen["best_epoch"]):
                raise RuntimeError("FIT-selected checkpoint changed")
            core._set_random_seed(42 + fold)
            model = exp.runtime["model_cls"](args).to(exp.device)
            exp._dry_initialize_lazy_layers(model, train_loader)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            scores = {}
            with torch.no_grad():
                for raw in val_loader:
                    batch = move(raw, exp.device)
                    output = model(batch)
                    mask = batch["channel_mask"].detach().cpu().numpy().astype(bool)
                    logits = output["logits"].detach().cpu().numpy()
                    for i, sid in enumerate(batch["subject_id"]):
                        if sid in scores:
                            raise RuntimeError("Duplicate target patient")
                        valid = mask[i]
                        scores[sid] = {"logit_nez": logits[i][valid].astype(np.float32),
                                       "n_channels": int(valid.sum())}
            if len(scores) != 13 or set(scores) != set(val):
                raise RuntimeError("Target score-only membership mismatch")
            path = a.runtime / "private" / "target_scores" / variant / f"fold_{fold}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with path.open("rb") as stream:
                    previous = pickle.load(stream)
                if set(previous) != set(scores) or any(
                        not np.array_equal(previous[sid]["logit_nez"], scores[sid]["logit_nez"])
                        for sid in scores):
                    raise RuntimeError("Existing target score file differs; refusing overwrite")
            else:
                temp = path.with_suffix(".pkl.tmp")
                with temp.open("wb") as stream:
                    pickle.dump(scores, stream, protocol=5)
                temp.replace(path)
            manifest[str(checkpoint.relative_to(a.runtime)).replace("\\", "/")] = sha(checkpoint)
            manifest[str(path.relative_to(a.runtime)).replace("\\", "/")] = sha(path)
            print(f"[TARGET_SCORE_FROZEN] {variant} fold={fold} cells=13", flush=True)
    if len(manifest) != 10 * len(variants):
        raise RuntimeError("Target score/checkpoint manifest incomplete")
    private = a.runtime / "private" / "TARGET_SCORE_AND_CHECKPOINT_MANIFEST.json"
    atomic_json(private, {"lock_sha": LOCK_SHA, "files": dict(sorted(manifest.items())),
                          "target_labels_indexed_for_scoring": False})
    atomic_json(a.output / "TARGET_SCORE_FREEZE_AUDIT.json",
                {"pass": True, "lock_sha": LOCK_SHA, "n_variants": len(variants),
                 "n_fold_models": 5 * len(variants), "n_target_score_files": 5 * len(variants),
                 "target_labels_indexed_for_scoring": False,
                 "legacy_loader_materialized_target_labels": True,
                 "private_manifest_sha256": sha(private), "outer_test_accessed": False})
    print(f"TARGET_SCORE_FREEZE_PASS models={5 * len(variants)}", flush=True)


if __name__ == "__main__":
    main()
