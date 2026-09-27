"""Frozen-inventory richer-feature probe; FIT-only PCA and shared scorer."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA

from analyze import BASE, align_pair, csv_out, fit_shared, ranking
from common import EXPERIMENT, RUNTIME, ensure_source, make_experiment, write_json


def source_views_all(window_features, centers, names, args):
    """Call the exact source four-view implementation with inventory-fixed all-feature indices."""
    from neuroez_c import evidence_views
    original = evidence_views._feature_group_indices
    try:
        evidence_views._feature_group_indices = lambda dim, _args: (np.arange(dim, dtype=np.int64), tuple(names))
        return evidence_views.b0_self_reference_features(window_features, centers, args)
    finally:
        evidence_views._feature_group_indices = original


def rich_patient(samples, patient_meta, names, args):
    from neuroez_c.evidence_views import b0_self_reference_features
    canonical = list(patient_meta["canonical_channels"])
    label_mask = np.asarray(patient_meta.get("label_mask", np.ones(len(canonical), dtype=bool)), dtype=bool)
    labels = np.asarray(patient_meta["labels"], dtype=np.int8)
    valid = label_mask & (labels >= 0)
    lookup = {name: i for i, name in enumerate(canonical)}
    nine = [[] for _ in canonical]
    all_features = [[] for _ in canonical]
    for sample in samples:
        features = np.asarray(sample["window_features"], dtype=np.float32)
        if list(sample["window_feature_names"]) != names or features.shape[2] != len(names):
            raise RuntimeError("Frozen cache feature inventory changed")
        centers = np.asarray(sample["window_relative_centers_sec"], dtype=np.float32)
        v9 = b0_self_reference_features(features, centers, args)
        va = source_views_all(features, centers, names, args)
        for local, channel_name in enumerate(sample["channel_names_norm"]):
            channel = lookup.get(channel_name)
            if channel is None or not valid[channel]:
                continue
            nine[channel].append(v9[:, local].mean(0))
            all_features[channel].append(va[:, local].mean(0))
    def aggregate(per_channel):
        output = []
        for channel in np.flatnonzero(valid):
            if not per_channel[channel]:
                raise RuntimeError("Valid channel has no cache features")
            values = np.stack(per_channel[channel])
            output.append(np.r_[values.mean(0), values.std(0)])
        return np.asarray(output, dtype=np.float32)
    return {"RICH9": aggregate(nine), "RICHALL": aggregate(all_features), "y": labels[valid]}


def eval_fold(fold, raw, exp, split, names):
    from exp_ez_hybrid import flatten_window_samples
    prepared = {}
    for role, subjects in (("fit", split["fit_subjects"]), ("val", split["validation_subjects"])):
        samples = flatten_window_samples(exp.run_records, subject_ids=subjects)
        groups = defaultdict(list)
        for sample in samples:
            groups[sample["subject_id"]].append(sample)
        prepared[role] = {subject: rich_patient(groups[subject], exp.patient_index[subject], names, exp.args)
                          for subject in subjects}
        if set(prepared[role]) != set(raw[role]):
            raise RuntimeError("Richer extraction patient membership changed")
        for subject in subjects:
            if not np.array_equal(prepared[role][subject]["y"], raw[role][subject]["y"]):
                raise RuntimeError("RICH9 label alignment changed")
            if not np.allclose(prepared[role][subject]["RICH9"], raw[role][subject]["R1"], atol=1e-5, rtol=1e-5):
                raise RuntimeError("RICH9 does not replay source RAW72 view")
    return prepared


def score_variant(fold, fit, val, name):
    model = fit_shared(fit)
    metrics = [ranking(row["y"], model.decision_function(row["x"])) for row in val.values()]
    return {"fold": fold, "variant": name, "n_patients": len(metrics),
            **{key: float(np.mean([m[key] for m in metrics])) for key in ("ap", "auc", "mrr", "top1", "ndcg")}}


def run():
    ensure_source()
    inventory = json.loads((EXPERIMENT / "FEATURE_INVENTORY.json").read_text(encoding="utf-8"))
    names = inventory["cache_feature_names_in_order"]
    exp = make_experiment()
    outputs = defaultdict(list)
    for split in exp.outer_splits:
        fold = int(split["fold_idx"])
        with (RUNTIME / "private" / f"fold_{fold}" / "raw.pkl").open("rb") as stream:
            raw = pickle.load(stream)
        prepared = eval_fold(fold, raw, exp, split, names)
        for rep in ("RICH9", "RICHALL"):
            fit, val = align_pair(prepared["fit"], prepared["val"], rep, "patient_z")
            outputs[rep].append(score_variant(fold, fit, val, rep))
            if rep == "RICH9":
                from analyze import by_selected, load_fold
                # Direct sanity: same R1 patient-z inputs imply same shared scores.
                if not all(np.array_equal(fit[p]["y"], raw["fit"][p]["y"]) for p in fit):
                    raise RuntimeError("RICH9 shared-probe alignment differs")
            xfit = np.concatenate([row["x"] for row in fit.values()])
            pca = PCA(n_components=72, svd_solver="full")
            pca.fit(xfit)
            f_pca = {p: {"x": pca.transform(row["x"]), "y": row["y"]} for p, row in fit.items()}
            v_pca = {p: {"x": pca.transform(row["x"]), "y": row["y"]} for p, row in val.items()}
            outputs[rep + "_PCA72"].append(score_variant(fold, f_pca, v_pca, rep + "_PCA72"))
        print(f"[RICHER] fold={fold} complete", flush=True)
    folder = EXPERIMENT / "richer_features"
    for key, filename in (("RICH9", "RICH9_BY_FOLD.csv"), ("RICHALL", "RICHALL_BY_FOLD.csv"),
                          ("RICH9_PCA72", "RICH9_PCA_BY_FOLD.csv"), ("RICHALL_PCA72", "RICHALL_PCA_BY_FOLD.csv")):
        csv_out(folder / filename, outputs[key])
    comparisons = []
    for fold in range(1, 6):
        rows = {name: next(r for r in values if r["fold"] == fold) for name, values in outputs.items()}
        comparisons.append({"fold": fold,
                            **{f"delta_{key}": rows["RICHALL"][key] - rows["RICH9"][key] for key in ("ap", "auc", "mrr", "top1")},
                            "delta_pca_ap": rows["RICHALL_PCA72"]["ap"] - rows["RICH9_PCA72"]["ap"]})
    csv_out(folder / "RICHER_FEATURE_COMPARISON.csv", comparisons)
    means = {key: float(np.mean([row[key] for row in comparisons])) for key in ("delta_ap", "delta_auc", "delta_mrr", "delta_top1", "delta_pca_ap")}
    pass_gate = (means["delta_ap"] >= .020 and sum(r["delta_ap"] > 0 for r in comparisons) >= 4 and
                 means["delta_auc"] >= 0 and means["delta_mrr"] >= -.005 and means["delta_top1"] >= 0 and
                 means["delta_pca_ap"] >= .010)
    write_json(folder / "RICHER_FEATURE_GATE.json", {"mean_deltas": means,
               "positive_folds": sum(r["delta_ap"] > 0 for r in comparisons), "pass": bool(pass_gate),
               "terminal": "RICHER_HANDCRAFTED_FEATURE_HEADROOM_SUPPORTED" if pass_gate else "RICHER_HANDCRAFTED_FEATURE_HEADROOM_NOT_SUPPORTED"})


if __name__ == "__main__":
    run()
