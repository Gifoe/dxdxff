"""Fusion, permutation, bootstrap, boundary, cross-seizure, and LOCO analyses."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .evaluation import patient_metrics, select_threshold


KEYS = ["seed", "outer_fold", "patient_id", "center", "channel_name", "label_nez"]


def audit_primary_oof_ledger(
    ledger: pd.DataFrame,
    expected_patients: int = 80,
    expected_channels: int = 7635,
) -> None:
    """Reject partial, duplicated, or fold-inconsistent primary OOF predictions."""
    required = set(KEYS + ["partition"])
    if not required.issubset(ledger):
        raise ValueError("Prediction ledger is missing audit columns")
    test = ledger.query("partition == 'test'")
    for seed, group in test.groupby("seed"):
        if group.patient_id.nunique() != expected_patients or len(group) != expected_channels:
            raise ValueError(
                f"Seed {seed} is incomplete: expected "
                f"{expected_patients} patients/{expected_channels} channels"
            )
        identity = ["patient_id", "channel_name"]
        if group.duplicated(identity).any():
            raise ValueError(f"Seed {seed} contains duplicated held-out channels")
        folds_per_patient = group.groupby("patient_id").outer_fold.nunique()
        if not folds_per_patient.eq(1).all():
            raise ValueError(f"Seed {seed} assigns a patient to multiple test folds")


def align_branches(prq: pd.DataFrame, bcr: pd.DataFrame) -> pd.DataFrame:
    """Strict one-to-one alignment; partial or duplicated ledgers are rejected."""
    required_prq = set(KEYS + ["probability_nez"])
    required_bcr = set(KEYS + ["logit_ez"])
    if not required_prq.issubset(prq) or not required_bcr.issubset(bcr):
        raise ValueError("Branch ledgers do not satisfy the required schema")
    if prq.duplicated(KEYS).any() or bcr.duplicated(KEYS).any():
        raise ValueError("Duplicate branch rows")
    aligned = prq[KEYS + ["partition", "probability_nez"]].merge(
        bcr[KEYS + ["partition", "logit_ez"]],
        on=KEYS,
        suffixes=("_prq", "_bcr"),
        validate="one_to_one",
    )
    if len(aligned) != len(prq) or len(aligned) != len(bcr):
        raise ValueError("PRQ-Net and BCR-Net ledgers are not exactly aligned")
    if not (aligned["partition_prq"] == aligned["partition_bcr"]).all():
        raise ValueError("Partition disagreement between branches")
    aligned["partition"] = aligned.pop("partition_prq")
    aligned = aligned.drop(columns="partition_bcr")
    return aligned


def add_cdel_probability(aligned: pd.DataFrame, bcr_weight: float = 0.20) -> pd.DataFrame:
    if not np.isclose(bcr_weight, 0.20):
        raise ValueError("Primary CDEL requires the locked BCR weight 0.20")
    result = aligned.copy()
    result["bcr_probability_nez"] = 1.0 / (1.0 + np.exp(result["logit_ez"]))
    result["probability_nez"] = (
        0.80 * result["probability_nez"] + 0.20 * result["bcr_probability_nez"]
    )
    return result


def fusion_weight_sensitivity(
    aligned: pd.DataFrame,
    weights: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> pd.DataFrame:
    """Post hoc sweep using frozen ledgers and validation-only thresholds."""
    rows = []
    for weight in weights:
        for (seed, fold), group in aligned.groupby(["seed", "outer_fold"]):
            fused = group.copy()
            fused["probability_nez"] = (
                (1 - weight) * fused["probability_nez"]
                + weight / (1 + np.exp(fused["logit_ez"]))
            )
            threshold = select_threshold(fused.query("partition == 'validation'")).threshold
            patient = patient_metrics(fused.query("partition == 'test'"), threshold)
            rows.append(
                {
                    "bcr_weight": weight,
                    "seed": seed,
                    "outer_fold": fold,
                    "patient_macro_f1": patient["macro_f1"].mean(),
                }
            )
    return pd.DataFrame(rows)


def evaluate_cdel_by_fold(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one threshold per validation fold and reuse it unchanged on test."""
    threshold_rows = []
    test_rows = []
    for (seed, fold), group in aligned.groupby(["seed", "outer_fold"]):
        validation = add_cdel_probability(
            group.loc[group["partition"] == "validation"]
        )
        test = add_cdel_probability(group.loc[group["partition"] == "test"])
        selection = select_threshold(validation)
        test["selected_threshold"] = selection.threshold
        threshold_rows.append(
            {"seed": seed, "outer_fold": fold, "threshold": selection.threshold}
        )
        test_rows.append(test)
    return pd.concat(test_rows, ignore_index=True), pd.DataFrame(threshold_rows)


def within_patient_bcr_permutation(
    aligned: pd.DataFrame,
    threshold_by_fold: pd.DataFrame,
    repeats: int = 1000,
    seed: int = 2027,
) -> pd.DataFrame:
    """Shuffle only BCR-Net scores within each patient; all else stays fixed."""
    random = np.random.default_rng(seed)
    rows = []
    thresholds = {
        (int(row.seed), int(row.outer_fold)): float(row.threshold)
        for row in threshold_by_fold.itertuples()
    }
    test = aligned.loc[aligned["partition"] == "test"].copy()
    for repeat in range(repeats):
        shuffled = test.copy()
        shuffled["logit_ez"] = shuffled.groupby(
            ["seed", "outer_fold", "patient_id"]
        )["logit_ez"].transform(lambda values: random.permutation(values.to_numpy()))
        fused = add_cdel_probability(shuffled)
        patient_parts = []
        for (model_seed, fold), group in fused.groupby(["seed", "outer_fold"]):
            patient_parts.append(
                patient_metrics(group, thresholds[(int(model_seed), int(fold))])
            )
        patient = pd.concat(patient_parts, ignore_index=True)
        rows.append({"repeat": repeat, "patient_macro_f1": patient["macro_f1"].mean()})
    return pd.DataFrame(rows)


def paired_patient_bootstrap(
    patient_table: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    repeats: int = 10000,
    seed: int = 2027,
) -> dict[str, float]:
    """Average across seeds first, then resample patients with replacement."""
    averaged = (
        patient_table.groupby(["patient_id", "model"], as_index=False)[metric]
        .mean()
        .pivot(index="patient_id", columns="model", values=metric)
        .dropna(subset=[model_a, model_b])
    )
    delta = (averaged[model_a] - averaged[model_b]).to_numpy()
    random = np.random.default_rng(seed)
    draws = delta[random.integers(0, len(delta), size=(repeats, len(delta)))].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_repeats": int(repeats),
        "patients": int(len(delta)),
    }


def boundary_channel_analysis(
    prq_test: pd.DataFrame,
    bcr_test: pd.DataFrame,
    cdel_test: pd.DataFrame,
    prq_thresholds: pd.DataFrame,
    bcr_thresholds: pd.DataFrame | None = None,
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Evaluate the same PRQ-defined hardest channels for all three models."""
    prq_map = {
        (int(row.seed), int(row.outer_fold)): float(row.threshold)
        for row in prq_thresholds.itertuples()
    }
    bcr_map = (
        {
            (int(row.seed), int(row.outer_fold)): float(row.threshold)
            for row in bcr_thresholds.itertuples()
        }
        if bcr_thresholds is not None
        else prq_map
    )
    keys = ["seed", "outer_fold", "patient_id", "channel_name", "label_nez"]
    joined = prq_test[keys + ["probability_nez"]].rename(
        columns={"probability_nez": "prq"}
    ).merge(
        bcr_test[keys + ["probability_nez"]].rename(columns={"probability_nez": "bcr"}),
        on=keys,
        validate="one_to_one",
    ).merge(
        cdel_test[keys + ["probability_nez", "selected_threshold"]].rename(
            columns={"probability_nez": "cdel"}
        ),
        on=keys,
        validate="one_to_one",
    )
    rows = []
    for fraction in fractions:
        pieces = []
        for (seed, fold, patient), group in joined.groupby(
            ["seed", "outer_fold", "patient_id"]
        ):
            threshold = prq_map[(int(seed), int(fold))]
            count = max(1, int(np.ceil(fraction * len(group))))
            pieces.append(
                group.assign(difficulty=np.abs(group["prq"] - threshold))
                .nsmallest(count, "difficulty")
            )
        subset = pd.concat(pieces)
        for model in ("prq", "bcr", "cdel"):
            if model == "cdel":
                threshold_column = subset["selected_threshold"]
            else:
                mapping = prq_map if model == "prq" else bcr_map
                threshold_column = subset.apply(
                    lambda row: mapping[(int(row.seed), int(row.outer_fold))], axis=1
                )
            label_ez = 1 - subset["label_nez"].to_numpy()
            prediction_ez = (subset[model].to_numpy() < np.asarray(threshold_column)).astype(int)
            true_positive = ((label_ez == 1) & (prediction_ez == 1)).sum()
            false_positive = ((label_ez == 0) & (prediction_ez == 1)).sum()
            false_negative = ((label_ez == 1) & (prediction_ez == 0)).sum()
            rows.append(
                {
                    "hardest_fraction": fraction,
                    "model": model,
                    "ez_recall": true_positive / max((label_ez == 1).sum(), 1),
                    "ez_f1": 2 * true_positive / max(
                        2 * true_positive + false_positive + false_negative, 1
                    ),
                }
            )
    return pd.DataFrame(rows)


def fixed_seizure_subsets(
    seizure_ids: list[str],
    count: int,
    repeats: int = 10,
    seed: int = 2027,
) -> list[tuple[str, ...]]:
    """Generate deterministic, shared subsets for one-/two-seizure inference."""
    if count > len(seizure_ids):
        raise ValueError("Requested more seizures than are available")
    random = np.random.default_rng(seed)
    return [
        tuple(sorted(random.choice(seizure_ids, size=count, replace=False).tolist()))
        for _ in range(repeats)
    ]


def run_cross_seizure_inference(
    patient_seizures: dict[str, list[str]],
    infer: Callable[[str, tuple[str, ...]], pd.DataFrame],
    counts: tuple[int, ...] = (1, 2),
    repeats: int = 10,
    seed: int = 2027,
) -> pd.DataFrame:
    """Run frozen-checkpoint inference on matched patients and shared subsets."""
    matched = {
        patient: seizures
        for patient, seizures in patient_seizures.items()
        if len(seizures) >= max(counts)
    }
    outputs = []
    for patient, seizures in matched.items():
        for count in counts:
            subsets = fixed_seizure_subsets(
                seizures, count, repeats, seed + sum(map(ord, patient)) + count
            )
            for repeat, subset in enumerate(subsets):
                frame = infer(patient, subset).copy()
                frame["seizure_count"] = count
                frame["repeat"] = repeat
                frame["patient_id"] = patient
                outputs.append(frame)
        frame = infer(patient, tuple(sorted(seizures))).copy()
        frame["seizure_count"] = "all"
        frame["repeat"] = 0
        frame["patient_id"] = patient
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def loco_patient_bootstrap(
    patient_table: pd.DataFrame,
    repeats: int = 2000,
    seed: int = 2027,
) -> pd.DataFrame:
    """Average seeds, then bootstrap patients separately within held-out centers."""
    random = np.random.default_rng(seed)
    averaged = patient_table.groupby(
        ["held_out_center", "model", "patient_id"], as_index=False
    )["macro_f1"].mean()
    rows = []
    for (center, model), group in averaged.groupby(["held_out_center", "model"]):
        values = group["macro_f1"].to_numpy()
        draws = values[
            random.integers(0, len(values), size=(repeats, len(values)))
        ].mean(axis=1)
        rows.append(
            {
                "held_out_center": center,
                "model": model,
                "macro_f1": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "patients": len(values),
            }
        )
    return pd.DataFrame(rows)


def paired_condition_bootstrap(
    patient_table: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    metric: str,
    repeats: int = 10000,
    seed: int = 2027,
) -> dict[str, float]:
    """Average seeds/subset repeats, then bootstrap matched patient differences."""
    averaged = (
        patient_table.groupby(["patient_id", "seizure_count"], as_index=False)[metric]
        .mean()
        .pivot(index="patient_id", columns="seizure_count", values=metric)
        .dropna(subset=[condition_a, condition_b])
    )
    delta = (averaged[condition_a] - averaged[condition_b]).to_numpy()
    random = np.random.default_rng(seed)
    draws = delta[
        random.integers(0, len(delta), size=(repeats, len(delta)))
    ].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_repeats": repeats,
        "patients": len(delta),
    }
