"""Aggregate-only diagnostic probes on private frozen FIT/validation arrays."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import EXPERIMENT, RUNTIME, ensure_source, selected_epochs, write_json

METRICS = ("ap", "auc", "mrr", "top1", "ndcg")
BASE = ("log_bp_delta", "log_bp_theta", "log_bp_beta", "log_bp_low_gamma", "log_bp_high_gamma", "rms", "variance", "line_length_per_sec", "spectral_entropy")
VIEWS = ("ABS", "DELTA", "ZDELTA", "RATIO")


def csv_out(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Empty public CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ranking(y, score) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(y) != len(score) or len(y) == 0 or len(np.unique(y)) < 2 or not np.all(np.isfinite(score)):
        raise RuntimeError("Ranking requires finite two-class patient/channel subset")
    order = np.argsort(-score, kind="stable")
    rank = int(np.flatnonzero(y[order] == 1)[0] + 1)
    discount = 1.0 / np.log2(np.arange(2, len(y) + 2))
    ideal = float(discount[: int(y.sum())].sum())
    return {"ap": float(average_precision_score(y, score)),
            "auc": float(roc_auc_score(y, score)),
            "mrr": 1.0 / rank,
            "top1": float(y[order[0]] == 1),
            "ndcg": float((y[order] * discount).sum() / ideal)}


def align(x: np.ndarray, mode: str, stats=None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "native":
        return x
    if mode == "global_fit":
        mu, sd = stats
        return (x - mu) / np.maximum(sd, 1e-6)
    if mode == "patient_z":
        return (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)
    if mode == "patient_robust":
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        return (x - med) / np.maximum(1.4826 * mad, 1e-6)
    if mode == "patient_rank":
        ranks = np.apply_along_axis(rankdata, 0, x, method="average")
        return norm.ppf(np.clip((ranks - 0.5) / len(x), 1e-4, 1 - 1e-4))
    raise ValueError(mode)


def align_pair(fit, val, rep, mode):
    all_fit = np.concatenate([row[rep] for row in fit.values()])
    stats = (all_fit.mean(0), all_fit.std(0)) if mode == "global_fit" else None
    return ({p: {"x": align(row[rep], mode, stats), "y": row["y"]} for p, row in fit.items()},
            {p: {"x": align(row[rep], mode, stats), "y": row["y"]} for p, row in val.items()})


def fit_shared(fit):
    x, y, w = [], [], []
    n = len(fit)
    for row in fit.values():
        yi = np.asarray(row["y"], dtype=np.int8)
        if set(np.unique(yi)) != {0, 1}:
            raise RuntimeError("FIT patient lacks one class")
        x.append(row["x"])
        y.append(yi)
        w.append(np.where(yi == 1, 0.5 / (n * yi.sum()), 0.5 / (n * (len(yi) - yi.sum()))))
    model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(np.concatenate(x), np.concatenate(y), sample_weight=np.concatenate(w))
    return model


def load_fold(fold):
    folder = RUNTIME / "private" / f"fold_{fold}"
    with (folder / "raw.pkl").open("rb") as stream:
        raw = pickle.load(stream)
    epochs = {}
    for epoch in sorted(set(selected_epochs(fold).values())):
        with (folder / f"epoch_{epoch:02d}.pkl").open("rb") as stream:
            epochs[epoch] = pickle.load(stream)
    return raw, epochs


def by_selected(fold, raw, epochs, rep, mode):
    selected = selected_epochs(fold)
    rows, models, aligned = [], {}, {}
    for epoch in sorted(set(selected.values())) if rep in ("R2", "R3", "R4", "R5") else [0]:
        source = epochs[epoch] if epoch else raw
        fit, val = align_pair(source["fit"], source["val"], rep, mode)
        model = None if rep == "R5" else fit_shared(fit)
        models[epoch] = model
        aligned[epoch] = (fit, val)
        for subject in val:
            if selected[subject] != epoch and epoch != 0:
                continue
            score = val[subject]["x"].reshape(-1) if rep == "R5" else model.decision_function(val[subject]["x"])
            rows.append({"fold": fold, "subject": subject, "epoch": epoch, "representation": rep,
                         "alignment": mode, **ranking(val[subject]["y"], score)})
    if len(rows) != 13:
        raise RuntimeError(f"Expected 13 selected validation patients for {rep}/{mode}, got {len(rows)}")
    return rows, models, aligned


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys, key, strict=True)), "n_patients": len(part),
             **{metric: float(np.mean([r[metric] for r in part])) for metric in METRICS}}
            for key, part in sorted(groups.items())]


def feature_directions(raw_folds):
    private, public = [], []
    for fold, raw in raw_folds.items():
        fit, val = raw["fit"], raw["val"]
        for feature in range(36):
            fit_effect = {p: 2 * roc_auc_score(row["y"], row["R0"][:, feature]) - 1 for p, row in fit.items()}
            median = float(np.median(list(fit_effect.values())))
            orientation = 1 if median >= 0 else -1
            val_effect = {p: 2 * roc_auc_score(row["y"], row["R0"][:, feature]) - 1 for p, row in val.items()}
            for patient, effect in val_effect.items():
                private.append({"fold": fold, "feature": feature, "patient": patient, "effect": effect, "oriented": orientation * effect})
            effects = np.asarray(list(val_effect.values()), dtype=float)
            agree = float(np.mean(orientation * effects > 0))
            entropy = 0.0 if agree in (0, 1) else float(-agree * np.log2(agree) - (1 - agree) * np.log2(1 - agree))
            public.append({"fold": fold, "feature": f"{VIEWS[feature // 9]}:{BASE[feature % 9]}",
                           "view": VIEWS[feature // 9], "base_descriptor": BASE[feature % 9],
                           "fit_median_effect": median,
                           "fit_positive_sign_fraction": float(np.mean(np.asarray(list(fit_effect.values())) > 0)),
                           "validation_sign_agreement_fraction": agree,
                           "validation_median_oriented_effect": float(np.median(orientation * effects)),
                           "validation_mean_absolute_effect": float(np.mean(np.abs(effects))),
                           "sign_entropy_bits": entropy,
                           "interpatient_iqr": float(np.quantile(effects, 0.75) - np.quantile(effects, 0.25))})
    csv_out(EXPERIMENT / "single_feature" / "FEATURE_DIRECTION_STABILITY.csv", public)
    for key, name in (("view", "VIEW_DIRECTION_STABILITY.csv"), ("base_descriptor", "BASE_DESCRIPTOR_DIRECTION_STABILITY.csv")):
        grouped = defaultdict(list)
        for row in public:
            grouped[row[key]].append(row)
        csv_out(EXPERIMENT / "single_feature" / name,
                [{key: group, "n_fold_features": len(part),
                  **{metric: float(np.mean([p[metric] for p in part])) for metric in
                     ("validation_sign_agreement_fraction", "validation_median_oriented_effect", "validation_mean_absolute_effect", "sign_entropy_bits", "interpatient_iqr")}}
                 for group, part in sorted(grouped.items())])


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def direction(row):
    return unit(row["x"][row["y"] == 1].mean(0) - row["x"][row["y"] == 0].mean(0))


def geometry(fold, rep, selected, aligned):
    private = []
    for epoch, (fit, val) in aligned.items():
        fd = {p: direction(row) for p, row in fit.items()}
        consensus = unit(np.mean(list(fd.values()), axis=0))
        pair = np.asarray([float(np.dot(a, b)) for a, b in itertools.combinations(fd.values(), 2)])
        for patient, row in val.items():
            if selected[patient] != epoch and epoch != 0:
                continue
            d = direction(row)
            score = row["x"] @ consensus
            transfers = [ranking(row["y"], row["x"] @ source) for source in fd.values()]
            private.append({"fold": fold, "representation": rep, "patient": patient,
                            "direction_cosine": float(np.dot(consensus, d)),
                            "consensus_ap": ranking(row["y"], score)["ap"],
                            "consensus_auc": ranking(row["y"], score)["auc"],
                            "pair_mean": float(pair.mean()), "pair_median": float(np.median(pair)),
                            "pair_std": float(pair.std()), "pair_q10": float(np.quantile(pair, .10)),
                            "pair_q25": float(np.quantile(pair, .25)), "pair_q75": float(np.quantile(pair, .75)),
                            "pair_q90": float(np.quantile(pair, .90)),
                            "pair_neg": float(np.mean(pair < 0)), "pair_low": float(np.mean(pair < .2)),
                            "pair_high": float(np.mean(pair > .5)),
                            "transfer_ap_mean": float(np.mean([t["ap"] for t in transfers])),
                            "transfer_ap_median": float(np.median([t["ap"] for t in transfers])),
                            "transfer_auc_mean": float(np.mean([t["auc"] for t in transfers])),
                            "transfer_reverse_fraction": float(np.mean([t["auc"] < .5 for t in transfers])),
                            "transfer_below_prevalence_fraction": float(np.mean([t["ap"] < row["y"].mean() for t in transfers])),
                            "between_source_ap_variance": float(np.var([t["ap"] for t in transfers]))})
    if len(private) != 13:
        raise RuntimeError("Geometry selected-patient count changed")
    return private


def local_mlp(xtrain, ytrain, xtest, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Linear(xtrain.shape[1], 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x = torch.tensor(xtrain, dtype=torch.float32)
    y = torch.tensor(ytrain, dtype=torch.float32)
    model.train()
    for _ in range(50):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xtest, dtype=torch.float32)).squeeze(-1).numpy()


def local_probes(fold, rep, selected, models, aligned):
    patients = []
    not_estimable = 0
    for epoch, (fit, val) in aligned.items():
        shared = models[epoch]
        for patient, row in val.items():
            if selected[patient] != epoch and epoch != 0:
                continue
            x, y = row["x"], np.asarray(row["y"], dtype=np.int8)
            k = int(min(5, y.sum(), len(y) - y.sum()))
            if k < 2:
                not_estimable += 1
                continue
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            totals = defaultdict(float)
            for train_idx, test_idx in splitter.split(x, y):
                # No inner test-channel label reaches either local optimizer.
                local = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
                local.fit(x[train_idx], y[train_idx])
                scores = {"linear": local.decision_function(x[test_idx]),
                          "mlp": local_mlp(x[train_idx], y[train_idx], x[test_idx], 42),
                          "shared": shared.decision_function(x[test_idx])}
                for model, score in scores.items():
                    metric = ranking(y[test_idx], score)
                    for key in ("ap", "auc"):
                        totals[f"{model}_{key}"] += len(test_idx) * metric[key] / len(x)
            patients.append({"fold": fold, "representation": rep, "patient": patient, "n_channels": len(x),
                             "n_splits": k, **totals,
                             "linear_delta": totals["linear_ap"] - totals["shared_ap"],
                             "mlp_delta": totals["mlp_ap"] - totals["shared_ap"]})
    return patients, not_estimable


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def run():
    ensure_source()
    raw_folds = {}
    shared_private, geometry_private, local_private = [], [], []
    not_estimable = defaultdict(int)
    for fold in range(1, 6):
        raw, epochs = load_fold(fold)
        raw_folds[fold] = raw
        selected = selected_epochs(fold)
        configs = [("R0", mode) for mode in ("global_fit", "patient_z", "patient_robust", "patient_rank")]
        configs += [("R1", mode) for mode in ("global_fit", "patient_z", "patient_robust", "patient_rank")]
        configs += [(rep, "native") for rep in ("R2", "R3", "R4", "R5")]
        for rep, mode in configs:
            rows, models, aligned = by_selected(fold, raw, epochs, rep, mode)
            shared_private += rows
            if (rep, mode) in (("R1", "patient_z"), ("R3", "native"), ("R4", "native")):
                geometry_private += geometry(fold, rep, selected, aligned)
                local, skipped = local_probes(fold, rep, selected, models, aligned)
                local_private += local
                not_estimable[rep] += skipped
        print(f"[ANALYZE] fold={fold} complete", flush=True)
    feature_directions(raw_folds)
    csv_out(EXPERIMENT / "shared_linear" / "SHARED_LINEAR_BY_REPRESENTATION.csv",
            summarize(shared_private, ("fold", "representation", "alignment")))
    ladder = [r for r in summarize(shared_private, ("fold", "representation", "alignment")) if r["representation"] == "R1"]
    csv_out(EXPERIMENT / "shared_linear" / "ALIGNMENT_LADDER_BY_FOLD.csv", ladder)
    grouped = defaultdict(list)
    for r in ladder:
        grouped[r["alignment"]].append(r)
    global_fold = {r["fold"]: r for r in grouped["global_fit"]}
    align_gate = {}
    for mode in ("patient_z", "patient_robust", "patient_rank"):
        delta = [r["ap"] - global_fold[r["fold"]]["ap"] for r in grouped[mode]]
        avg = {m: float(np.mean([r[m] for r in grouped[mode]])) for m in METRICS}
        base = {m: float(np.mean([r[m] for r in grouped["global_fit"]])) for m in METRICS}
        align_gate[mode] = {"delta_ap": float(np.mean(delta)), "positive_folds": int(np.sum(np.asarray(delta) > 0)),
                            "pass": bool(np.mean(delta) >= .020 and np.sum(np.asarray(delta) > 0) >= 4 and
                                         all(avg[m] >= base[m] for m in ("auc", "mrr", "top1")))}
    write_json(EXPERIMENT / "shared_linear" / "ALIGNMENT_GATE.json", {"variants": align_gate,
               "terminal": "SIMPLE_FEATURE_ALIGNMENT_SUPPORTED" if any(v["pass"] for v in align_gate.values()) else "SIMPLE_FEATURE_ALIGNMENT_NOT_SUPPORTED"})

    for prefix, columns, output in (
        ("direction", ("direction_cosine", "consensus_ap", "consensus_auc"), "CENTROID_DIRECTION_SUMMARY.csv"),
        ("pair", ("pair_mean", "pair_median", "pair_std", "pair_q10", "pair_q25", "pair_q75", "pair_q90", "pair_neg", "pair_low", "pair_high"), "FIT_PAIRWISE_COSINE_SUMMARY.csv"),
        ("transfer", ("transfer_ap_mean", "transfer_ap_median", "transfer_auc_mean", "transfer_reverse_fraction", "transfer_below_prevalence_fraction", "between_source_ap_variance"), "FIT_TO_VALIDATION_TRANSFER_SUMMARY.csv"),
    ):
        groups = defaultdict(list)
        for row in geometry_private:
            groups[(row["fold"], row["representation"])].append(row)
        csv_out(EXPERIMENT / "geometry" / output,
                [{"fold": fold, "representation": rep, "n_patients": len(part),
                  **{name: float(np.median([r[name] for r in part])) if name == "direction_cosine" else float(np.mean([r[name] for r in part])) for name in columns}}
                 for (fold, rep), part in sorted(groups.items())])

    local_summary, gap, gates = [], [], {}
    for rep in ("R1", "R3", "R4"):
        patients = [r for r in local_private if r["representation"] == rep]
        matching = [r for r in geometry_private if r["representation"] == rep and r["patient"] in {p["patient"] for p in patients}]
        for method in ("linear", "mlp"):
            delta = np.asarray([p[f"{method}_delta"] for p in patients])
            low, high = bootstrap(delta)
            gap.append({"representation": rep, "method": method, "n_patients": len(patients),
                        "mean_delta_ap": float(delta.mean()), "median_delta_ap": float(np.median(delta)),
                        "ci_low": low, "ci_high": high, "positive_patient_fraction": float(np.mean(delta > 0)),
                        "fraction_over_0_05": float(np.mean(delta > .05)), "fraction_over_0_10": float(np.mean(delta > .10))})
            local_summary.append({"representation": rep, "method": method, "n_patients": len(patients),
                                  "not_estimable": not_estimable[rep],
                                  "mean_ap": float(np.mean([p[f"{method}_ap"] for p in patients])),
                                  "mean_auc": float(np.mean([p[f"{method}_auc"] for p in patients])),
                                  "matched_shared_ap": float(np.mean([p["shared_ap"] for p in patients])),
                                  "matched_shared_auc": float(np.mean([p["shared_auc"] for p in patients]))})
        line = next(r for r in local_summary if r["representation"] == rep and r["method"] == "linear")
        line_gap = next(r for r in gap if r["representation"] == rep and r["method"] == "linear")
        cosine = float(np.median([r["direction_cosine"] for r in matching]))
        reverse = float(np.mean([r["transfer_reverse_fraction"] for r in matching]))
        cross_gap = line["mean_ap"] - float(np.mean([r["transfer_ap_mean"] for r in matching]))
        gates[rep] = {"median_direction_cosine": cosine, "reversed_transfer_fraction": reverse,
                      "local_to_cross_ap_gap": cross_gap,
                      "pass": bool(line_gap["mean_delta_ap"] >= .05 and line_gap["ci_low"] > 0 and
                                   line_gap["positive_patient_fraction"] >= .65 and line["mean_ap"] >= .6 and
                                   (cosine <= .4 or reverse >= .2 or cross_gap >= .05))}
    csv_out(EXPERIMENT / "patient_specific" / "PATIENT_SPECIFIC_LINEAR_SUMMARY.csv", [r for r in local_summary if r["method"] == "linear"])
    csv_out(EXPERIMENT / "patient_specific" / "PATIENT_SPECIFIC_MLP_SUMMARY.csv", [r for r in local_summary if r["method"] == "mlp"])
    csv_out(EXPERIMENT / "patient_specific" / "MATCHED_SHARED_VS_PATIENT_SPECIFIC.csv", local_summary)
    csv_out(EXPERIMENT / "patient_specific" / "GEOMETRY_GAP_BOOTSTRAP.csv", gap)
    write_json(EXPERIMENT / "patient_specific" / "GEOMETRY_GATE.json", {"representations": gates,
               "terminal": "CROSS_PATIENT_GEOMETRY_INSTABILITY_SUPPORTED" if any(v["pass"] for v in gates.values()) else "CROSS_PATIENT_GEOMETRY_INSTABILITY_NOT_SUPPORTED"})
    write_json(EXPERIMENT / "SANITY_AUDIT.json", {"fit_validation_overlap": False, "outer_loader_constructed": False,
               "label_free_alignments": True, "patient_specific_inner_test_labels_in_training": False,
               "representation_repeat_check": "checked per extracted checkpoint; see private extraction_status.json",
               "remaining_controls": ["FIT-label permutation", "channel-permutation invariance", "richer feature construction"]})
    print("[ANALYZE] aggregate outputs complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
