"""Locked FIT-only data, Teacher-view, and exact-A1 Student utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA = "acfcbd600fbbd666ce5ccc8d480b50b6e131a7f7358da697860bb9d6db81fcb2"
VARIANTS = (
    "D0_CONTINUED_HARDLABEL", "D1_PAIRWISE_KD", "D2_LISTWISE_KD",
    "D3_CORRECTION_FOCUSED_KD_ALL", "D3_CORRECTION_FOCUSED_KD_TEACHER_WEIGHTED",
    "D4_PAIR_LIST_KD",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def imports():
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST",
                "R1_HLV_RUNTIME", "A1_A2_RUNTIME", "FPPD_RUNTIME", "AFPC_RUNTIME", "SRGI_RUNTIME"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing source environment variable: {key}")
    if sha(ROOT / "PROTOCOL_LOCK.json") != LOCK_SHA:
        raise RuntimeError("v2 Student protocol lock changed")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
    sys.path.insert(0, str(source / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
    sys.path.insert(0, str(source / "fullpool_privileged_ranking_distillation_seed42_v1" / "code"))
    import exp_ez_hybrid as core
    import protocol as v1
    import teacher as v1_teacher
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args
    from objectives import patient_equal_weighted_bce_loss
    from exp_ez_hybrid import _move_tensors_to_device
    v1.preflight()
    assert_sources()
    install_interleaved_hlv_view()
    provenance = json.loads((ROOT / "V1_TEACHER_PROVENANCE.json").read_text(encoding="utf-8"))
    if (not provenance["TEACHER_AP_SIGNAL_ELIGIBLE_FOR_STUDENT"] or
            provenance["private_file_manifest_sha256"] !=
            "53bffce7c7d191faa98fa8e27e46b64e504b3d71c1cc35a2585bc69e6773c458"):
        raise RuntimeError("Replayed v1 Teacher not eligible / manifest changed")
    manifest = Path(os.environ["FPPD_RUNTIME"]) / "v2_teacher_input_hash_manifest_PRIVATE.json"
    if sha(manifest) != provenance["private_file_manifest_sha256"]:
        raise RuntimeError("Teacher input hash manifest changed")
    return core, v1, v1_teacher, make_args, patient_equal_weighted_bce_loss, _move_tensors_to_device


def meta_split(fold: int, fit: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(fit, key=lambda sid: hashlib.sha256(
        f"42|{fold}|{sid}|student_meta".encode()).hexdigest())
    n_meta = math.ceil(.2 * len(ordered))
    meta, train = ordered[:n_meta], ordered[n_meta:]
    if set(meta) & set(train) or set(meta) | set(train) != set(fit):
        raise RuntimeError("Student FIT/meta membership mismatch")
    return train, meta


@dataclass
class FoldData:
    fold: int
    exp: object
    args: object
    train_ids: list[str]
    meta_ids: list[str]
    train_loader: object
    meta_loader: object
    source_state: dict
    normalizer: object
    teacher: dict
    patient_weight: dict
    core: object
    v1: object
    hard_loss: object
    move: object


def load_teacher_views(fold: int, fit: list[str], train_ids: list[str], v1, v1_teacher) -> tuple[dict, dict]:
    manifest_path = Path(os.environ["FPPD_RUNTIME"]) / "v2_teacher_input_hash_manifest_PRIVATE.json"
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    result = {sid: [] for sid in fit}
    gains = {sid: 0.0 for sid in fit}
    for ctx in v1.contexts(fold):
        weight = len(ctx["target_ids"]) / 13.0
        for sid in fit:
            path = v1_teacher.oof_path(ctx, sid)
            key = str(path.relative_to(v1.RUNTIME)).replace("\\", "/")
            if sha(path) != files[key]:
                raise RuntimeError("Teacher OOF target changed after v2 lock")
            with path.open("rb") as stream:
                row = pickle.load(stream)
            if row["sid"] != sid or row["context_id"] != ctx["context_id"]:
                raise RuntimeError("Teacher patient/context mismatch")
            t = np.asarray(row["teacher_score"], dtype=np.float32)
            a = np.asarray(row["a1_score"], dtype=np.float32)
            y = np.asarray(row["y"], dtype=np.int8)
            if len(t) != len(a) or len(t) != len(y) or not np.isfinite(t).all():
                raise RuntimeError("Malformed Teacher target")
            ap_t = v1.afc.rank_metrics(y, t)["ap"]
            ap_a = v1.afc.rank_metrics(y, a)["ap"]
            if np.isfinite(ap_t) and np.isfinite(ap_a):
                gains[sid] += weight * (ap_t - ap_a)
            result[sid].append({"t": (t - t.mean()) / max(float(t.std()), 1e-6),
                                "a": (a - a.mean()) / max(float(a.std()), 1e-6),
                                "y": y, "weight": weight})
    if any(abs(sum(view["weight"] for view in result[sid]) - 1.0) > 1e-6 for sid in fit):
        raise RuntimeError("Teacher context multiplicity did not sum to 13")
    clipped = {sid: min(max(gains[sid], 0.0), .2) for sid in train_ids}
    positive = [value for value in clipped.values() if value > 0]
    normalizer = float(np.mean(positive)) if positive else 1.0
    weighted = {sid: clipped[sid] / normalizer for sid in train_ids}
    return result, weighted


def make_fold(fold: int, runtime: Path) -> FoldData:
    core, v1, v1_teacher, make_args, hard_loss, move = imports()
    args = make_args("R0", runtime / "scratch" / f"fold_{fold}")
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("A1 cohort changed")
    split = next(x for x in exp.outer_splits if int(x["fold_idx"]) == fold)
    fit = list(split["fit_subjects"])
    val = list(split["validation_subjects"])
    test = list(split["test_subjects"])
    if (len(val) != 13 or len(fit) + len(val) + len(test) != 80 or
            set(fit) & set(val) or set(fit) & set(test) or set(val) & set(test)):
        raise RuntimeError("Frozen A1 roles changed")
    train_ids, meta_ids = meta_split(fold, fit)
    # A1 epoch30 was trained with the normalizer estimated on all fold-FIT
    # patients. Keep it exact; exclude FIT-meta IDs from v2 gradient updates.
    full_train, meta_set, _, normalizer = exp._build_datasets(fit, meta_ids, [])
    train_index = [i for i, patient in enumerate(full_train.patient_examples)
                   if patient["subject_id"] in set(train_ids)]
    if len(train_index) != len(train_ids) or len(meta_set) != len(meta_ids):
        raise RuntimeError("Student-train/meta dataset membership mismatch")
    train_loader = exp._make_loader(Subset(full_train, train_index), shuffle=True, batch_size=2)
    meta_loader = exp._make_loader(meta_set, shuffle=False, batch_size=2)
    source_path = Path(os.environ["A1_A2_RUNTIME"]) / "A1" / f"fold_{fold}" / "epoch_30.pt"
    source_state = torch.load(source_path, map_location="cpu", weights_only=False)
    if source_state["variant"] != "A1" or source_state["fold"] != fold or source_state["epoch"] != 30:
        raise RuntimeError("A1 source checkpoint identity mismatch")
    if (not np.allclose(source_state["normalizer_mean"], normalizer.mean, rtol=0, atol=1e-6) or
            not np.allclose(source_state["normalizer_std"], normalizer.std, rtol=0, atol=1e-6)):
        raise RuntimeError("A1 full-FIT normalizer replay mismatch")
    teacher, patient_weight = load_teacher_views(fold, fit, train_ids, v1, v1_teacher)
    return FoldData(fold, exp, args, train_ids, meta_ids, train_loader, meta_loader,
                    source_state, normalizer, teacher, patient_weight, core, v1, hard_loss, move)


def new_model(data: FoldData):
    data.core._set_random_seed(42 + data.fold)
    model = data.exp.runtime["model_cls"](data.args).to(data.exp.device)
    data.exp._dry_initialize_lazy_layers(model, data.train_loader)
    model.load_state_dict(data.source_state["model_state_dict"], strict=True)
    return model


def optimizer(model, ratio: float = .1):
    upper_names = ("channel_classifier.", "seizure_aggregator.", "final_norm.")
    upper, lower = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (upper if name.startswith(upper_names) else lower).append(parameter)
    if not upper or not lower:
        raise RuntimeError("Discriminative optimizer group empty")
    return torch.optim.AdamW([{"params": upper, "lr": 1e-4}, {"params": lower, "lr": ratio * 1e-4}],
                             weight_decay=1e-3)


def _standardize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-6)


def _pairs(n: int, fold: int, epoch: int, step: int, sid: str) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.triu_indices(n, k=1)
    if len(a) > 128:
        seed = int.from_bytes(hashlib.sha256(
            f"42|{fold}|{epoch}|{step}|{sid}|pair".encode()).digest()[:8], "little")
        choose = np.random.default_rng(seed).choice(len(a), size=128, replace=False)
        a, b = a[choose], b[choose]
    return a, b


def kd_loss(data: FoldData, variant: str, tau: float, batch: dict,
            logits: torch.Tensor, epoch: int, step: int) -> torch.Tensor:
    per_patient = []
    for i, sid in enumerate(batch["subject_id"]):
        valid = batch["channel_mask"][i].bool()
        s = _standardize(-logits[i][valid])
        y = batch["labels_ez"][i][valid].detach().cpu().numpy().astype(np.int8)
        context_losses = []
        for view in data.teacher[sid]:
            if len(s) != len(view["t"]) or not np.array_equal(y, view["y"]):
                raise RuntimeError("Teacher target/A1 training channel order mismatch")
            t = torch.as_tensor(view["t"], device=s.device, dtype=s.dtype)
            a1 = torch.as_tensor(view["a"], device=s.device, dtype=s.dtype)
            pair = None
            listwise = None
            if variant != "D2_LISTWISE_KD":
                ia, ib = _pairs(len(s), data.fold, epoch, step, sid)
                ia = torch.as_tensor(ia, device=s.device)
                ib = torch.as_tensor(ib, device=s.device)
                pt = torch.sigmoid((t[ia] - t[ib]) / tau).clamp(1e-6, 1 - 1e-6)
                ps_logit = (s[ia] - s[ib]) / tau
                entropy = -(pt * torch.log(pt) + (1 - pt) * torch.log1p(-pt))
                kl = F.binary_cross_entropy_with_logits(ps_logit, pt, reduction="none") - entropy
                if variant.startswith("D3_"):
                    pa = torch.sigmoid((a1[ia] - a1[ib]) / tau)
                    weights = (pt - pa).abs()
                    pair = (weights * kl).sum() / weights.sum().clamp_min(1e-6)
                else:
                    pair = kl.mean()
            if variant in ("D2_LISTWISE_KD", "D4_PAIR_LIST_KD"):
                listwise = F.kl_div(F.log_softmax(s / tau, dim=0), F.softmax(t / tau, dim=0),
                                    reduction="sum")
            loss = (listwise if pair is None else pair if listwise is None else .5 * (pair + listwise))
            context_losses.append(float(view["weight"]) * loss)
        patient_kd = sum(context_losses)
        if variant == "D3_CORRECTION_FOCUSED_KD_TEACHER_WEIGHTED":
            patient_kd = patient_kd * float(data.patient_weight[sid])
        per_patient.append(patient_kd)
    return torch.stack(per_patient).mean() if per_patient else logits.sum() * 0.0


def meta_metrics(data: FoldData, model) -> dict:
    model.eval()
    scores = []
    with torch.no_grad():
        for raw in data.meta_loader:
            batch = data.move(raw, data.exp.device)
            out = model(batch)
            for i, sid in enumerate(batch["subject_id"]):
                if sid not in data.meta_ids:
                    raise RuntimeError("Non-FIT-meta patient in selection loader")
                valid = batch["channel_mask"][i].bool()
                y = batch["labels_ez"][i][valid].detach().cpu().numpy().astype(np.int8)
                margin = -out["logits"][i][valid].detach().cpu().numpy()
                scores.append(data.v1.afc.query_metrics(y, margin))
    if len(scores) != len(data.meta_ids):
        raise RuntimeError("Incomplete FIT-meta scoring")
    return {name: float(np.mean([row[name] for row in scores]))
            for name in ("ap", "mrr", "top1", "auc", "macro_f1", "ez_f1", "ba")}
