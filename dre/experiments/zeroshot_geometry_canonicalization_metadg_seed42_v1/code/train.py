"""FIT-only selection and full-FIT source retraining for zero-shot variants.

All checkpoints, patient records and optimizer states are private. This module
never requests an outer-test loader and never uses validation target outcomes
to choose geometry/MLDG hyperparameters.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(PROJECT / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import epoch_grid, finalize_fold  # noqa: E402
from objectives import patient_equal_weighted_bce_loss  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402
from exp_ez_hybrid import _move_tensors_to_device  # noqa: E402
from geometry_losses import geometry_loss  # noqa: E402

LOCK_SHA = "80e028f57a6484f55273555fbdb76135551bc509ccece930006db85611958e5e"
RUNTIME = Path(os.environ.get("ZSG_RUNTIME", ""))
A1_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))
VARIANTS = (
    "Z1_DIRECTION_CANONICALIZATION", "Z2_CLASS_CONDITIONAL_ALIGNMENT",
    "Z2B_CROSSPATIENT_SUPCON", "Z3_PATIENT_HELDOUT_MLDG",
    "Z4_GEOMETRY_PLUS_MLDG", "C1_PATIENT_ADV",
)
DIR = (0.01, 0.03, 0.1, 0.3, 1.0)
CCA = (0.001, 0.003, 0.01, 0.03, 0.1)
SUP = tuple(dict(weight=w, tau=t) for w in (0.01, 0.03, 0.1) for t in (0.07, 0.1, 0.2))
META = tuple(dict(beta=b, alpha=a) for b in (0.1, 0.3, 1.0) for a in (1e-5, 3e-5, 1e-4))
ADV = (0.001, 0.01, 0.1)


def configs(variant, fold=None):
    if variant == VARIANTS[0]: return [dict(weight=x) for x in DIR]
    if variant == VARIANTS[1]: return [dict(weight=x) for x in CCA]
    if variant == VARIANTS[2]: return list(SUP)
    if variant == VARIANTS[3]: return list(META)
    if variant == VARIANTS[5]: return [dict(weight=x) for x in ADV]
    if variant == VARIANTS[4]:
        if fold is None: raise ValueError("Z4 requires fold-specific FIT-only selections")
        z1 = selected_config(fold, VARIANTS[0]); z2 = selected_config(fold, VARIANTS[1])
        z3 = selected_config(fold, VARIANTS[3])
        winner = z1 if z1["fit_ap"] >= z2["fit_ap"] else z2
        return [dict(geometry=winner["variant"], weight=winner["config"]["weight"],
                     beta=z3["config"]["beta"], alpha=z3["config"]["alpha"])]
    raise ValueError(variant)


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def preflight():
    if not os.environ.get("ZSG_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("ZSG_RUNTIME must be an absolute private directory")
    if not os.environ.get("A1_A2_RUNTIME") or not A1_RUNTIME.is_absolute():
        raise RuntimeError("A1_A2_RUNTIME must be an absolute private directory")
    if sha256(ROOT / "PROTOCOL_LOCK.json") != LOCK_SHA:
        raise RuntimeError("Zero-shot protocol lock changed")
    assert_sources()
    source = json.loads((PROJECT / "b8_teacher_ceiling_meta_readout_seed42_v1" / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if source["checkpoints"] != 150 or source["max_grid_error"] > 1e-6 or source["R4_max_logit_replay_error"] > 1e-6:
        raise RuntimeError("Exact A1 source reproduction failed")
    for fold in range(1, 6):
        if not (A1_RUNTIME / "A1" / f"fold_{fold}" / "epoch_30.pt").is_file():
            raise RuntimeError(f"A1 30-epoch grid incomplete, fold {fold}")


def fit_partition(ids, fold):
    ordered = sorted(ids)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"ZSG|42|{fold}|fit-split".encode()).digest()[:8], "little"))
    perm = rng.permutation(len(ordered)); n_val = max(1, int(round(.2 * len(ordered))))
    meta_val = [ordered[i] for i in perm[:n_val]]
    meta_train = [ordered[i] for i in perm[n_val:]]
    if set(meta_train) & set(meta_val) or set(meta_train + meta_val) != set(ids):
        raise RuntimeError("FIT partition invalid")
    return meta_train, meta_val


class CaptureR4:
    def __init__(self, model):
        self.h = None
        self.handle = model.channel_classifier.classifier.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        self.h = inputs[0]
        if self.h.shape[-1] != 64:
            raise RuntimeError(f"Expected exact R4 dimension 64, got {self.h.shape[-1]}")

    def close(self): self.handle.remove()


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def source_loss(outputs, batch):
    return patient_equal_weighted_bce_loss(outputs["logits"], batch["labels"], batch["labels_ez"], batch["channel_mask"])


def regularized_loss(model, capture, batch, variant, config, adv_head=None, patient_index=None):
    outputs = model(batch)
    loss = source_loss(outputs, batch)
    if variant in VARIANTS[:3] or variant == VARIANTS[4]:
        name = config["geometry"] if variant == VARIANTS[4] else variant
        loss = loss + float(config["weight"]) * geometry_loss(name, capture.h, batch, config)
    if variant == VARIANTS[5]:
        mask = batch["channel_mask"].unsqueeze(-1)
        pooled = (capture.h * mask).sum(1) / mask.sum(1).clamp_min(1)
        labels = torch.tensor([patient_index[sid] for sid in batch["subject_id"]], device=pooled.device)
        loss = loss + F.cross_entropy(adv_head(GradReverse.apply(pooled, float(config["weight"]))), labels)
    return loss


def slice_batch(batch, sl):
    n = len(batch["subject_id"])
    result = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.ndim and v.shape[0] == n: result[k] = v[sl]
        elif isinstance(v, (list, tuple)) and len(v) == n: result[k] = v[sl]
        else: result[k] = v
    return result


def train_epoch(exp, model, capture, loader, opt, variant, config, adv_head, patient_index):
    model.train()
    if adv_head is not None: adv_head.train()
    losses = []
    meta = variant in (VARIANTS[3], VARIANTS[4])
    for raw in loader:
        batch = _move_tensors_to_device(raw, exp.device)
        if meta:
            if len(batch["subject_id"]) < 4: continue
            tr = slice_batch(batch, slice(0, 2)); te = slice_batch(batch, slice(2, 4))
            if set(tr["subject_id"]) & set(te["subject_id"]):
                raise RuntimeError("MLDG meta-train/meta-test patient overlap")
            opt.zero_grad(set_to_none=True)
            trloss = regularized_loss(model, capture, tr, variant, config)
            trloss.backward()
            grad = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
            # Captured R4 carries a non-leaf autograd graph. It must not be
            # deep-copied along with the registered capture hook.
            capture.h = None
            virtual = copy.deepcopy(model)
            vcap = CaptureR4(virtual)
            with torch.no_grad():
                for p, g in zip(virtual.parameters(), grad):
                    if g is not None: p.add_(g, alpha=-float(config["alpha"]))
            virtual.train()
            vloss = source_loss(virtual(te), te)
            vloss.backward()
            # First-order: propagate meta-test gradient through the identity
            # Jacobian of the one-step virtual update; omit Hessian terms.
            for p, vp, g in zip(model.parameters(), virtual.parameters(), grad):
                p.grad = (torch.zeros_like(p) if g is None else g) + float(config["beta"]) * (torch.zeros_like(p) if vp.grad is None else vp.grad.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(float((trloss + float(config["beta"]) * vloss).detach().cpu()))
            vcap.close(); del virtual
        else:
            opt.zero_grad(set_to_none=True)
            loss = regularized_loss(model, capture, batch, variant, config, adv_head, patient_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(adv_head.parameters()) if adv_head is not None else []), 1.0)
            opt.step(); losses.append(float(loss.detach().cpu()))
    if not losses: raise RuntimeError("No training episodes completed")
    return float(np.mean(losses))


def evaluate_fit(exp, model, loader):
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    _, _, records = exp._evaluate(model, loader, ez_weight, split_name="val")
    from sklearn.metrics import average_precision_score, roc_auc_score
    aps, mrrs, top1 = [], [], []
    for row in records:
        valid = np.asarray(row["channel_mask"], dtype=bool)
        y = np.asarray(row["labels_ez"])[valid]
        s = np.asarray(row["score_ez"])[valid]
        if set(np.unique(y)) != {0., 1.}: continue
        aps.append(float(average_precision_score(y, s)))
        order = np.argsort(-s, kind="stable")
        mrrs.append(1.0 / (int(np.flatnonzero(y[order] == 1)[0]) + 1))
        top1.append(float(y[order[0]]))
    if not aps: raise RuntimeError("No estimable FIT meta-validation patients")
    return dict(ap=float(np.mean(aps)), mrr=float(np.mean(mrrs)), top1=float(np.mean(top1)), n=len(aps))


def score_only_snapshot(exp, model, capture, loader):
    """Persist all validation scores/R4 before the label-using VLOO pass."""
    model.eval()
    patients = {}
    with torch.no_grad():
        for raw in loader:
            # The legacy collator has already materialized labels; neither this
            # function nor model.forward indexes them. The limitation is audited.
            batch = _move_tensors_to_device(raw, exp.device)
            outputs = model(batch)
            h = capture.h.detach().cpu().numpy()
            score = outputs["score_ez"].detach().cpu().numpy()
            nez_logits = outputs["logits"].detach().cpu().numpy()
            mask = batch["channel_mask"].detach().cpu().numpy().astype(bool)
            for i, sid in enumerate(batch["subject_id"]):
                if sid in patients: raise RuntimeError("Duplicate validation patient")
                patients[sid] = dict(score_ez=score[i][mask[i]].astype(np.float32),
                                     logit_nez=nez_logits[i][mask[i]].astype(np.float32),
                                     R4=h[i][mask[i]].astype(np.float32),
                                     n_channels=int(mask[i].sum()))
    if len(patients) != 13: raise RuntimeError("Expected exact 13 validation patient snapshots")
    return patients


def selected_config(fold, variant):
    path = RUNTIME / "selection" / f"fold_{fold}" / f"{variant}.json"
    if not path.is_file(): raise RuntimeError(f"FIT-only selection missing: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["lock_sha"] != LOCK_SHA or row["fold"] != fold or row["variant"] != variant:
        raise RuntimeError("FIT-only selection provenance mismatch")
    return row


def select(fold, variant):
    variant_configs = configs(variant, fold)
    rows = []
    for i, cfg in enumerate(variant_configs):
        p = RUNTIME / "fit" / f"fold_{fold}" / variant / f"config_{i:02d}" / "summary.json"
        if not p.is_file(): raise RuntimeError(f"Incomplete FIT grid: {p}")
        row = json.loads(p.read_text(encoding="utf-8"))
        if row["config"] != cfg or row["lock_sha"] != LOCK_SHA: raise RuntimeError("FIT config mismatch")
        rows.append(row)
    winner = max(rows, key=lambda r: (round(r["best_ap"], 12), round(r["best_mrr"], 12),
                                      round(r["best_top1"], 12), -float(r["config"].get("weight", 0)),
                                      -int(r["best_epoch"]), -int(r["config_index"])))
    result = dict(lock_sha=LOCK_SHA, fold=fold, variant=variant, config=winner["config"],
                  config_index=winner["config_index"], fit_ap=winner["best_ap"],
                  fit_mrr=winner["best_mrr"], fit_top1=winner["best_top1"],
                  fit_epoch=winner["best_epoch"], candidates=len(rows))
    write_json(RUNTIME / "selection" / f"fold_{fold}" / f"{variant}.json", result)
    print(f"[SELECT] fold={fold} {variant} config={result['config_index']} FIT_AP={result['fit_ap']:.6f}", flush=True)


def run_train(fold, variant, index, stage):
    cfgs = configs(variant, fold)
    if not 0 <= index < len(cfgs): raise RuntimeError("Config index out of range")
    cfg = cfgs[index]
    if stage == "full":
        selection = selected_config(fold, variant)
        if selection["config_index"] != index or selection["config"] != cfg:
            raise RuntimeError("Full-FIT training must use the FIT-only selected config")
    split = next(s for s in EXP.outer_splits if int(s["fold_idx"]) == fold)
    fit_ids, val_ids = list(split["fit_subjects"]), list(split["validation_subjects"])
    if stage == "fit":
        fit_ids, val_ids = fit_partition(fit_ids, fold)
    if set(fit_ids) & set(val_ids) or set(fit_ids) & set(split["test_subjects"]):
        raise RuntimeError("Data role overlap")
    train_set, val_set, _, _ = EXP._build_datasets(fit_ids, val_ids, [])
    loader = EXP._make_loader(train_set, shuffle=True, batch_size=4 if variant in (VARIANTS[3], VARIANTS[4]) else 2)
    validation = EXP._make_loader(val_set, shuffle=False, batch_size=2)
    folder = RUNTIME / stage / f"fold_{fold}" / variant / f"config_{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / "summary.json"
    if summary_path.is_file():
        print(f"[SKIP] {stage} fold={fold} {variant} config={index}", flush=True); return
    core._set_random_seed(42 + fold)
    model = EXP.runtime["model_cls"](EXP.args).to(EXP.device)
    EXP._dry_initialize_lazy_layers(model, loader)
    init_path = A1_RUNTIME / "initial" / f"fold_{fold}_initial.pt"
    reference = torch.load(init_path, map_location="cpu", weights_only=True)
    model.load_state_dict(reference, strict=True)
    capture = CaptureR4(model)
    pindex = {sid: i for i, sid in enumerate(sorted(fit_ids))}
    adv_head = torch.nn.Linear(64, len(pindex)).to(EXP.device) if variant == VARIANTS[5] else None
    parameters = list(model.parameters()) + (list(adv_head.parameters()) if adv_head is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-3)
    metrics = []
    for epoch in range(1, 31):
        state_path = folder / f"epoch_{epoch:02d}.pt"
        record_path = folder / f"epoch_{epoch:02d}.json"
        if state_path.is_file() and record_path.is_file():
            state = torch.load(state_path, map_location=EXP.device, weights_only=False)
            if state["epoch"] != epoch or state["config"] != cfg or state["variant"] != variant:
                raise RuntimeError("Resume state mismatch")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            if adv_head is not None: adv_head.load_state_dict(state["adversary"], strict=True)
            row = json.loads(record_path.read_text(encoding="utf-8"))
            if stage == "full":
                score_path = folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
                if not score_path.is_file() or sha256(score_path) != row["score_sha256"]:
                    raise RuntimeError("Frozen score snapshot missing or changed")
        else:
            core._set_random_seed(42 * 100000 + fold * 1000 + epoch)
            train_loss = train_epoch(EXP, model, capture, loader, optimizer, variant, cfg, adv_head, pindex)
            if stage == "fit":
                row = dict(epoch=epoch, train_loss=train_loss, **evaluate_fit(EXP, model, validation))
            else:
                import pickle
                snapshot = score_only_snapshot(EXP, model, capture, validation)
                score_path = folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
                score_tmp = score_path.with_suffix(".tmp")
                with score_tmp.open("wb") as f: pickle.dump(snapshot, f, protocol=5)
                score_tmp.replace(score_path)
                row = dict(epoch=epoch, train_loss=train_loss,
                           score_sha256=sha256(score_path), n_val_patients=len(snapshot))
            temp = state_path.with_suffix(".pt.tmp")
            torch.save(dict(lock_sha=LOCK_SHA, epoch=epoch, variant=variant, config=cfg,
                            model=model.state_dict(), optimizer=optimizer.state_dict(),
                            adversary=adv_head.state_dict() if adv_head is not None else None), temp)
            temp.replace(state_path)
            write_json(record_path, row)
        metrics.append(row)
        print(f"[{stage}] fold={fold} {variant} config={index} epoch={epoch}/30 loss={row['train_loss']:.5f}", flush=True)
    if stage == "fit":
        winner = max(metrics, key=lambda r: (round(r["ap"], 12), round(r["mrr"], 12),
                                             round(r["top1"], 12), -r["epoch"]))
        summary = dict(lock_sha=LOCK_SHA, fold=fold, variant=variant, config_index=index,
                       config=cfg, best_epoch=winner["epoch"], best_ap=winner["ap"],
                       best_mrr=winner["mrr"], best_top1=winner["top1"],
                       fit_patients=len(fit_ids), meta_validation_patients=len(val_ids))
    else:
        summary = dict(lock_sha=LOCK_SHA, fold=fold, variant=variant, config_index=index,
                       config=cfg, checkpoints=30, validation_scores_frozen=True,
                       score_snapshot_hashes=[r["score_sha256"] for r in metrics])
    write_json(summary_path, summary)
    capture.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fit", "select", "full", "audit"), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--config-index", type=int, default=0)
    args = parser.parse_args()
    preflight()
    if args.stage == "audit":
        print("SOURCE_A1_REPRODUCED 150/150; no target evaluation", flush=True); return
    if args.fold is None or args.variant is None: parser.error("--fold and --variant required")
    if args.stage == "select": select(args.fold, args.variant); return
    install_interleaved_hlv_view()
    global EXP
    model_args = make_args("R0", RUNTIME / "scratch")
    EXP = core.Exp_EZHybridLocalization(model_args)
    if len(EXP.patient_index) != 80 or len(EXP.outer_splits) != 5:
        raise RuntimeError("Frozen A1 cohort changed")
    run_train(args.fold, args.variant, args.config_index, args.stage)


if __name__ == "__main__": main()
