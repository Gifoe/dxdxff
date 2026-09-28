"""FIT-only, resumable exact-A1 continued-training/KD Student grid.

No validation target dataset or target outcome is instantiated here. Every
candidate starts from the same fold-specific A1 epoch30 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

from student_core import (LOCK_SHA, VARIANTS, atomic_json, kd_loss, make_fold,
                          meta_metrics, new_model, optimizer)

TAUS = (.5, 1.0, 2.0)
LAMBDAS = (.03, .1, .3, 1.0)


def key(row: dict, lam: float) -> tuple:
    m = row["meta"]
    return (round(m["ap"], 12), round(m["mrr"], 12), round(m["top1"], 12), -lam)


def candidate_id(tau: float, lam: float) -> str:
    return f"tau_{tau:g}_lambda_{lam:g}".replace(".", "p")


def list_epochs(folder: Path) -> list[int]:
    rows = sorted(int(p.stem.split("_")[1]) for p in folder.glob("epoch_*.json"))
    if rows != list(range(len(rows))):
        raise RuntimeError("Noncontiguous Student epoch resume rows")
    return rows


def run_candidate(data, variant: str, tau: float, lam: float,
                  runtime: Path, target_epoch: int) -> dict:
    folder = runtime / "private" / variant / f"fold_{data.fold}" / candidate_id(tau, lam)
    folder.mkdir(parents=True, exist_ok=True)
    model = new_model(data)
    opt = optimizer(model)
    existing = list_epochs(folder)
    if existing:
        last = existing[-1]
        state = torch.load(folder / f"epoch_{last:02d}.pt", map_location=data.exp.device, weights_only=False)
        if (state["lock_sha"] != LOCK_SHA or state["fold"] != data.fold or
                state["variant"] != variant or state["tau"] != tau or state["lambda"] != lam or
                state["epoch"] != last):
            raise RuntimeError("Student resume identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
    else:
        score0 = meta_metrics(data, model)
        row = {"epoch": 0, "meta": score0, "train_loss": None, "hard_loss": None,
               "kd_loss": None, "seconds": 0.0}
        torch.save({"lock_sha": LOCK_SHA, "fold": data.fold, "variant": variant,
                    "tau": tau, "lambda": lam, "epoch": 0,
                    "model": model.state_dict(), "optimizer": opt.state_dict()}, folder / "epoch_00.pt")
        atomic_json(folder / "epoch_00.json", row)
        existing = [0]
    for epoch in range(existing[-1] + 1, target_epoch + 1):
        data.core._set_random_seed(42 * 100000 + data.fold * 1000 + epoch)
        data.exp.current_epoch = epoch
        data.exp._v2_step = 0

        def compute(self, outputs, batch, ez_weight, *, split_name="train"):
            if float(ez_weight) != 2.0:
                raise RuntimeError("A1 class weight changed")
            hard = data.hard_loss(outputs["logits"], batch["labels"],
                                  batch["labels_ez"], batch["channel_mask"])
            teacher = (outputs["logits"].sum() * 0.0 if variant == VARIANTS[0] else
                       kd_loss(data, variant, tau, batch, outputs["logits"], epoch, self._v2_step))
            self._v2_step += 1
            total = hard + lam * teacher
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite Student loss")
            return total, {"hard": float(hard.detach()), "kd": float(teacher.detach())}

        data.exp._compute_loss = types.MethodType(compute, data.exp)
        start = time.perf_counter()
        train = data.exp._train_one_epoch(model, data.train_loader, opt,
                                          torch.tensor(2.0, device=data.exp.device))
        meta = meta_metrics(data, model)
        if not all(math.isfinite(value) for value in meta.values()):
            raise RuntimeError("Nonfinite FIT-meta metric")
        row = {"epoch": epoch, "meta": meta, "train_loss": float(train["loss"]),
               "hard_loss": float(train["hard"]), "kd_loss": float(train["kd"]),
               "seconds": time.perf_counter() - start}
        temp = folder / f"epoch_{epoch:02d}.pt.tmp"
        torch.save({"lock_sha": LOCK_SHA, "fold": data.fold, "variant": variant,
                    "tau": tau, "lambda": lam, "epoch": epoch,
                    "model": model.state_dict(), "optimizer": opt.state_dict()}, temp)
        temp.replace(folder / f"epoch_{epoch:02d}.pt")
        atomic_json(folder / f"epoch_{epoch:02d}.json", row)
        print(f"[FIT] {variant} fold={data.fold} {candidate_id(tau,lam)} "
              f"epoch={epoch}/{target_epoch} meta_AP={meta['ap']:.6f} sec={row['seconds']:.1f}", flush=True)
        # Same FIT-meta early-stopping rule for D0+ and KD. The rule is active
        # only in the final schedule segment (after epoch 8) so halving stages
        # have their prespecified checkpoint budgets.
        if target_epoch == 30 and epoch >= 11:
            history = [json.loads((folder / f"epoch_{i:02d}.json").read_text(encoding="utf-8"))
                       for i in range(epoch + 1)]
            best_epoch = max(history, key=lambda r: key(r, lam))["epoch"]
            if epoch - best_epoch >= 6:
                print(f"[EARLY_STOP] {variant} fold={data.fold} epoch={epoch} best={best_epoch}", flush=True)
                break
    rows = [json.loads((folder / f"epoch_{i:02d}.json").read_text(encoding="utf-8"))
            for i in list_epochs(folder)]
    best = max(rows, key=lambda r: key(r, lam))
    return {"tau": tau, "lambda": lam, "candidate_id": candidate_id(tau, lam),
            "last_epoch": rows[-1]["epoch"], "best_epoch": best["epoch"],
            "best_meta": best["meta"], "best_key": list(key(best, lam))}


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    folder = a.runtime / "private" / a.variant / f"fold_{a.fold}"
    summary_path = folder / "FIT_SELECTION.json"
    if summary_path.exists():
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        if value["lock_sha"] != LOCK_SHA or value["variant"] != a.variant or value["fold"] != a.fold:
            raise RuntimeError("Completed Student fold identity mismatch")
        print(f"[SKIP] {a.variant} fold={a.fold} FIT selection complete", flush=True)
        return
    data = make_fold(a.fold, a.runtime)
    print(f"[FIT_SPLIT] {a.variant} fold={a.fold} Student-train={len(data.train_ids)} "
          f"FIT-meta={len(data.meta_ids)} Teacher-views={len(data.v1.contexts(a.fold))}", flush=True)
    if a.variant == VARIANTS[0]:
        selected = run_candidate(data, a.variant, 1.0, 0.0, a.runtime, 30)
        candidates = [selected]
    else:
        stage1 = [run_candidate(data, a.variant, tau, lam, a.runtime, 2)
                  for tau in TAUS for lam in LAMBDAS]
        top3 = sorted(stage1, key=lambda x: tuple(x["best_key"]), reverse=True)[:3]
        stage2 = [run_candidate(data, a.variant, x["tau"], x["lambda"], a.runtime, 8)
                  for x in top3]
        top1 = max(stage2, key=lambda x: tuple(x["best_key"]))
        selected = run_candidate(data, a.variant, top1["tau"], top1["lambda"], a.runtime, 30)
        candidates = stage1 + stage2 + [selected]
    output = {"lock_sha": LOCK_SHA, "variant": a.variant, "fold": a.fold,
              "student_train_count": len(data.train_ids), "fit_meta_count": len(data.meta_ids),
              "selection_used_validation_target_labels": False,
              "selected": selected, "candidate_history": candidates}
    atomic_json(summary_path, output)
    print(f"[FIT_SELECTED] {a.variant} fold={a.fold} {selected['candidate_id']} "
          f"epoch={selected['best_epoch']} meta_AP={selected['best_meta']['ap']:.6f}", flush=True)


if __name__ == "__main__":
    run()
