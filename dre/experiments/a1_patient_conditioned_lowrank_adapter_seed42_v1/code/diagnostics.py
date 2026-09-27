"""VLOO-selected wrong-context, subset robustness, and adapter mechanism controls."""
from __future__ import annotations

import csv
import itertools
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from adapter import FrozenR4Adapter
from common import ROOT, RUNTIME, patient_grid, preflight, read_csv, write_csv, write_json
from train_adapters import load_source, stable_seed


def metrics(subject, y, logits):
    logits = np.asarray(logits, dtype=np.float32)
    snez = torch.sigmoid(torch.from_numpy(logits)).numpy()
    y = np.asarray(y, dtype=np.float32)
    record = {"subject_id": subject, "channel_mask": np.ones(len(y), dtype=bool),
              "labels": 1 - y, "labels_ez": y, "labels_nez": 1 - y,
              "score_nez": snez, "score_ez": 1 - snez}
    fixed = patient_grid(record)["fixed"]
    order = np.argsort(-np.asarray(record["score_ez"]), kind="stable")
    discount = 1.0 / np.log2(np.arange(2, len(y) + 2))
    ndcg = float((y[order] * discount).sum() / discount[:int(y.sum())].sum())
    return {"ap": fixed["patient_ez_auprc"], "mrr": fixed["patient_ez_mrr"],
            "top1": fixed["top1_is_ez"], "auroc": fixed["patient_ez_auroc"], "ndcg": ndcg}


def adapter_from_state(fold, epoch, variant, device, dim):
    path = RUNTIME / "private" / f"fold_{fold}" / variant / f"source_epoch_{epoch:02d}" / "latest_adapter.pt"
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["adapter_epoch"] != 20 or saved["variant"] != variant:
        raise RuntimeError("Diagnostic requires exact adapter epoch 20")
    adapter = FrozenR4Adapter(dim, variant).to(device)
    adapter.load_state_dict(saved["adapter"], strict=True)
    return adapter.eval()


def score_with_context(adapter, classifier, target_h, donor_h, indices=None):
    h = torch.tensor(target_h, dtype=torch.float32, device=next(adapter.parameters()).device).unsqueeze(0)
    context = donor_h if indices is None else donor_h[indices]
    c = torch.tensor(context, dtype=torch.float32, device=h.device).unsqueeze(0)
    mask = torch.ones(c.shape[:2], dtype=torch.bool, device=h.device)
    with torch.no_grad():
        logits, a, adapted = adapter(h, c, mask, classifier)
    return logits[0].cpu().numpy(), a[0].cpu().numpy(), adapted[0].cpu().numpy()


def run_variant(variant, device):
    shuffled, robustness, diagnostics = [], [], []
    for fold in range(1, 6):
        fold_dir = RUNTIME / "private" / f"fold_{fold}"
        selected = read_csv(fold_dir / f"{variant}_VLOO_PRIVATE.csv")
        if len(selected) != 13:
            raise RuntimeError("VLOO diagnostic expects 13 patients")
        ordered = sorted(row["subject_id"] for row in selected)
        by_epoch = defaultdict(list)
        for row in selected:
            by_epoch[int(row["selected_epoch"])].append(row)
        coefficients = []
        same_epoch_coeff = defaultdict(list)
        for epoch, choices in by_epoch.items():
            payload, classifier = load_source(fold, epoch, device)
            adapter = adapter_from_state(fold, epoch, variant, device, payload["dim"])
            for choice in choices:
                subject = choice["subject_id"]
                target = payload["val"][subject]
                donor = ordered[(ordered.index(subject) + 1) % 13]
                target_h, donor_h = target["h"], payload["val"][donor]["h"]
                correct_logit, a, adapted = score_with_context(adapter, classifier, target_h, target_h)
                wrong_logit, _, _ = score_with_context(adapter, classifier, target_h, donor_h)
                original = classifier(torch.tensor(target_h, dtype=torch.float32, device=device)).squeeze(-1).detach().cpu().numpy()
                norm_ratio = float(np.mean(np.linalg.norm(adapted - target_h, axis=1) / np.maximum(np.linalg.norm(target_h, axis=1), 1e-12)))
                cosine = float(np.mean(np.sum(adapted * target_h, axis=1) /
                                       (np.maximum(np.linalg.norm(adapted, axis=1), 1e-12) * np.maximum(np.linalg.norm(target_h, axis=1), 1e-12))))
                correct, wrong = metrics(subject, target["y_ez"], correct_logit), metrics(subject, target["y_ez"], wrong_logit)
                if abs(correct["ap"] - float(choice["patient_ez_auprc"])) > 1e-7 or abs(correct["mrr"] - float(choice["patient_ez_mrr"])) > 1e-7:
                    raise RuntimeError("Selected-checkpoint correct-context replay mismatch")
                shuffled.append({"fold": fold, "variant": variant, "subject": subject,
                                 "correct_ap": correct["ap"], "shuffled_ap": wrong["ap"],
                                 "correct_mrr": correct["mrr"], "shuffled_mrr": wrong["mrr"],
                                 "correct_ndcg": correct["ndcg"], "shuffled_ndcg": wrong["ndcg"]})
                coefficients.append(a)
                same_epoch_coeff[epoch].append(a)
                diagnostics.append({"fold": fold, "variant": variant, "subject": subject,
                                    "a_norm": float(np.linalg.norm(a)), "a": a,
                                    "saturated_90": float(np.mean(np.abs(a) > .90)),
                                    "saturated_98": float(np.mean(np.abs(a) > .98)),
                                    "relative_change": norm_ratio, "embedding_cosine": cosine})
                for fraction in (.5, .7):
                    for repeat in range(20):
                        k = max(2, int(round(fraction * len(target_h))))
                        rng = np.random.default_rng(stable_seed(42, fold, epoch, fraction, repeat, subject))
                        indices = rng.permutation(len(target_h))[:k]
                        reduced_logit, reduced_a, _ = score_with_context(adapter, classifier, target_h, target_h, indices)
                        subset = metrics(subject, target["y_ez"], reduced_logit)
                        robust_cosine = float(np.dot(a, reduced_a) / max(np.linalg.norm(a) * np.linalg.norm(reduced_a), 1e-12))
                        robustness.append({"fold": fold, "variant": variant, "subject": subject,
                                           "fraction": fraction, "repeat": repeat,
                                           "a_cosine_to_full": robust_cosine,
                                           "ap": subset["ap"], "mrr": subset["mrr"], "top1": subset["top1"],
                                           "full_ap": correct["ap"], "full_mrr": correct["mrr"], "full_top1": correct["top1"]})
                robustness.append({"fold": fold, "variant": variant, "subject": subject,
                                   "fraction": 1.0, "repeat": 0, "a_cosine_to_full": 1.0,
                                   "ap": correct["ap"], "mrr": correct["mrr"], "top1": correct["top1"],
                                   "full_ap": correct["ap"], "full_mrr": correct["mrr"], "full_top1": correct["top1"]})
        per_fold = [r for r in diagnostics if r["fold"] == fold]
        coef = np.stack([r["a"] for r in per_fold])
        pairwise = []
        for same in same_epoch_coeff.values():
            pairwise += [float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
                         for x, y in itertools.combinations(same, 2)]
        pair_stats = {"same_epoch_pair_count": len(pairwise),
                      "pairwise_cosine_mean": float(np.mean(pairwise)) if pairwise else "NA",
                      "pairwise_cosine_median": float(np.median(pairwise)) if pairwise else "NA",
                      "pairwise_cosine_q10": float(np.quantile(pairwise, .10)) if pairwise else "NA",
                      "pairwise_cosine_q90": float(np.quantile(pairwise, .90)) if pairwise else "NA"}
        for row in per_fold:
            row.update(pair_stats)
        write_json(fold_dir / f"{variant}_diagnostic_private.json", {"fold": fold, "patients": len(per_fold),
                   "same_epoch_pairs": len(pairwise), "coefficient_std": coef.std(0).tolist()})
        print(f"[DIAG] fold={fold} {variant} complete", flush=True)
    return shuffled, robustness, diagnostics


def aggregate(shuffled, robustness, diagnostics, variant):
    folder = ROOT / ("p2_conditioned" if variant == "P2" else "p3_geometry_teacher")
    mechanism = []
    for fold in range(1, 6):
        part = [row for row in diagnostics if row["fold"] == fold]
        a = np.stack([row["a"] for row in part])
        mechanism.append({"fold": fold, "variant": variant, "n_patients": len(part),
                          "mean_a_norm": float(np.mean([row["a_norm"] for row in part])),
                          "std_a_norm": float(np.std([row["a_norm"] for row in part])),
                          **{f"component_{k}_std": float(a[:, k].std()) for k in range(4)},
                          "fraction_abs_gt_0_90": float(np.mean([r["saturated_90"] for r in part])),
                          "fraction_abs_gt_0_98": float(np.mean([r["saturated_98"] for r in part])),
                          "mean_relative_embedding_change": float(np.mean([r["relative_change"] for r in part])),
                          "mean_embedding_cosine": float(np.mean([r["embedding_cosine"] for r in part])),
                          **{key: part[0][key] for key in ("same_epoch_pair_count", "pairwise_cosine_mean",
                                                          "pairwise_cosine_median", "pairwise_cosine_q10", "pairwise_cosine_q90")}})
    write_csv(folder / f"{variant}_ADAPTER_DIAGNOSTICS.csv", mechanism)
    robustness_public = []
    for fold in range(1, 6):
        for fraction in (.5, .7, 1.0):
            part = [row for row in robustness if row["fold"] == fold and row["fraction"] == fraction]
            robustness_public.append({"fold": fold, "variant": variant, "context_fraction": fraction,
                                      "n_patient_subsets": len(part),
                                      **{f"mean_{metric}": float(np.mean([row[metric] for row in part]))
                                         for metric in ("a_cosine_to_full", "ap", "mrr", "top1", "full_ap", "full_mrr", "full_top1")}})
    write_csv(folder / f"{variant}_CONTEXT_ROBUSTNESS.csv", robustness_public)
    shuf = []
    for fold in range(1, 6):
        part = [r for r in shuffled if r["fold"] == fold]
        shuf.append({"fold": fold, "variant": variant, "n_patients": 13,
                     **{f"mean_{metric}": float(np.mean([row[metric] for row in part]))
                        for metric in ("correct_ap", "shuffled_ap", "correct_mrr", "shuffled_mrr", "correct_ndcg", "shuffled_ndcg")}})
    return shuf, mechanism


def main():
    preflight()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controls = []
    for variant in ("P2", "P3"):
        shuffled, robustness, diagnostics = run_variant(variant, device)
        rows, _ = aggregate(shuffled, robustness, diagnostics, variant)
        controls += rows
    write_csv(ROOT / "controls" / "SHUFFLED_CONTEXT_COMPARISON.csv", controls)


if __name__ == "__main__":
    main()
