from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from outcome_hifos.metrics import compute_patient_metrics
from outcome_hifos.collate import collate_outcome_patients
from outcome_hifos.dataset import OutcomePatientExample


def perturb_model_input(
    model_input: dict[str, torch.Tensor],
    *,
    channel_dropout_fraction: float = 0.0,
    drop_one_seizure: bool = False,
    channel_permutation: bool = False,
    window_order_shuffle: bool = False,
    shuffled_signal: bool = False,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Create a mask-only robustness perturbation without consulting outcomes."""

    output = {key: value.clone() if torch.is_tensor(value) else value for key, value in model_input.items()}
    generator = torch.Generator(device=output["channel_mask"].device).manual_seed(int(seed))
    if shuffled_signal:
        for patient in range(output["feature_x"].shape[0]):
            valid = output["window_channel_mask"][patient]
            selected = output["feature_x"][patient][valid].clone()
            if selected.shape[0] > 1:
                order = torch.randperm(selected.shape[0], generator=generator, device=selected.device)
                output["feature_x"][patient][valid] = selected[order]
    if channel_permutation:
        for patient in range(output["channel_mask"].shape[0]):
            order = torch.randperm(output["channel_mask"].shape[1], generator=generator, device=output["channel_mask"].device)
            output["feature_x"][patient] = output["feature_x"][patient].index_select(2, order)
            output["channel_mask"][patient] = output["channel_mask"][patient].index_select(0, order)
            output["seizure_channel_mask"][patient] = output["seizure_channel_mask"][patient].index_select(1, order)
            output["window_channel_mask"][patient] = output["window_channel_mask"][patient].index_select(2, order)
    if window_order_shuffle:
        for patient in range(output["window_mask"].shape[0]):
            for seizure in range(output["window_mask"].shape[1]):
                valid = torch.nonzero(output["window_mask"][patient, seizure], as_tuple=False).flatten()
                if valid.numel() > 1:
                    order = valid[torch.randperm(valid.numel(), generator=generator, device=valid.device)]
                    original_features = output["feature_x"][patient, seizure].clone()
                    original_mask = output["window_channel_mask"][patient, seizure].clone()
                    output["feature_x"][patient, seizure, valid] = original_features[order]
                    output["window_channel_mask"][patient, seizure, valid] = original_mask[order]
    if channel_dropout_fraction > 0:
        for patient in range(output["channel_mask"].shape[0]):
            valid = torch.nonzero(output["channel_mask"][patient], as_tuple=False).flatten()
            drop_count = min(max(0, int(round(valid.numel() * float(channel_dropout_fraction)))), max(valid.numel() - 1, 0))
            if drop_count:
                order = torch.randperm(valid.numel(), generator=generator, device=valid.device)[:drop_count]
                dropped = valid[order]
                output["channel_mask"][patient, dropped] = False
                output["seizure_channel_mask"][patient, :, dropped] = False
                output["window_channel_mask"][patient, :, :, dropped] = False
    if drop_one_seizure:
        for patient in range(output["seizure_mask"].shape[0]):
            valid = torch.nonzero(output["seizure_mask"][patient], as_tuple=False).flatten()
            if valid.numel() > 1:
                selected = valid[torch.randint(valid.numel(), (1,), generator=generator, device=valid.device)]
                output["seizure_mask"][patient, selected] = False
                output["window_mask"][patient, selected] = False
                output["seizure_channel_mask"][patient, selected] = False
                output["window_channel_mask"][patient, selected] = False
    return output


@torch.no_grad()
def replay_model_perturbations(
    model: torch.nn.Module,
    examples: list[OutcomePatientExample],
    output_dir: str | Path,
    device: torch.device,
    *,
    calibrator: Any,
    threshold: float,
    seed: int,
    outer_fold_idx: int,
    checkpoint_path: str | Path,
) -> pd.DataFrame:
    if not examples:
        raise ValueError("Shortcut replay requires non-empty held-out examples.")
    model.eval()
    loader = DataLoader(examples, batch_size=1, shuffle=False, collate_fn=partial(collate_outcome_patients, padding_value=0.0))
    perturbations = {
        "original": {},
        "shuffled_signal": {"shuffled_signal": True},
        "channel_permutation": {"channel_permutation": True},
        "window_order_shuffle": {"window_order_shuffle": True},
        "channel_dropout_10": {"channel_dropout_fraction": 0.10},
        "channel_dropout_20": {"channel_dropout_fraction": 0.20},
        "channel_dropout_30": {"channel_dropout_fraction": 0.30},
        "seizure_dropout": {"drop_one_seizure": True},
    }
    rows = []
    for batch in loader:
        base_input = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch["model_input"].items()}
        for offset, (name, options) in enumerate(perturbations.items()):
            perturbed = perturb_model_input(base_input, seed=int(seed) + offset, **options) if options else base_input
            logit = float(model(perturbed, persist_diagnostics=False)["logits"][0].detach().cpu())
            raw_probability = float(torch.sigmoid(torch.tensor(logit)))
            probability = float(np.asarray(calibrator.predict(np.asarray([logit], dtype=np.float64))).reshape(-1)[0])
            rows.append(
                {
                    "subject_id": str(batch["subject_id"][0]),
                    "center": str(batch["center"][0]),
                    "outcome": int(float(batch["outcome"][0])),
                    "perturbation": name,
                    "outer_fold_idx": int(outer_fold_idx),
                    "seed": int(seed),
                    "raw_logit": logit,
                    "raw_probability": raw_probability,
                    "calibrated_probability": probability,
                    "probability": probability,
                    "selected_threshold": float(threshold),
                    "predicted": int(probability >= float(threshold)),
                    "calibration_method": "platt_inner_oof",
                    "threshold_source": "inner_oof",
                    "checkpoint_path": str(checkpoint_path),
                }
            )
    predictions = pd.DataFrame(rows)
    metric_rows = []
    for name, group in predictions.groupby("perturbation", sort=True):
        bundle = compute_patient_metrics(group["outcome"].to_numpy(), group["probability"].to_numpy(), predicted=group["predicted"].to_numpy())
        metric_rows.append({"perturbation": name, "status": "evaluated", **bundle.values, "undefined_reasons": json.dumps(bundle.undefined_reasons, sort_keys=True)})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "shortcut_replay_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "shortcut_replay_metrics.csv", index=False)
    return predictions


SHORTCUT_MODEL_NAMES = {
    "center_only",
    "channel_count_only",
    "seizure_count_only",
    "window_count_only",
    "record_duration_only",
    "quality_only",
    "sampling_rate_device_only",
    "metadata_combined",
}


def compute_shortcut_risk(main_macro_f1: float, metrics: pd.DataFrame) -> tuple[bool, list[str]]:
    eligible = metrics[
        (metrics["status"].astype(str) == "evaluated")
        & metrics["shortcut"].astype(str).isin(SHORTCUT_MODEL_NAMES)
    ].copy()
    considered = sorted(eligible["shortcut"].astype(str).unique().tolist())
    values = pd.to_numeric(eligible.get("macro_f1", pd.Series(dtype=float)), errors="coerce").dropna()
    risk = bool(np.isfinite(main_macro_f1) and not values.empty and float(values.max()) >= float(main_macro_f1) - 0.02)
    return risk, considered


def _cross_validated_metadata_probability(manifest: pd.DataFrame, ledger: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    base = manifest.merge(ledger[["subject_id", "fold_idx"]], on="subject_id", validate="one_to_one")
    design = pd.get_dummies(base[columns], columns=[column for column in columns if base[column].dtype == object], dtype=float)
    probabilities = np.zeros(len(base), dtype=np.float64)
    for fold in sorted(base["fold_idx"].unique().tolist()):
        train = base["fold_idx"] != fold
        test = ~train
        target = base.loc[train, "outcome_label"].to_numpy(dtype=np.int64)
        if np.unique(target).size < 2:
            probabilities[test] = float(np.mean(target))
            continue
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        estimator.fit(design.loc[train].to_numpy(dtype=float), target)
        probabilities[test] = estimator.predict_proba(design.loc[test].to_numpy(dtype=float))[:, 1]
    return pd.DataFrame(
        {
            "subject_id": base["subject_id"],
            "outcome": base["outcome_label"],
            "probability": probabilities,
            "fold_idx": base["fold_idx"],
        }
    )


def run_shortcut_audit(input_dir: str | Path) -> dict[str, object]:
    root = Path(input_dir)
    manifest_path = root / "manifests" / "outcome_training_patient_manifest.csv"
    ledger_path = root / "folds" / "outcome_fold_assignments_feature.csv"
    if not manifest_path.exists() or not ledger_path.exists():
        raise FileNotFoundError("Shortcut audit requires the training patient manifest and frozen feature fold ledger.")
    manifest = pd.read_csv(manifest_path)
    ledger = pd.read_csv(ledger_path)
    definitions = {
        "center_only": ["center"],
        "channel_count_only": ["channel_count"],
        "seizure_count_only": ["seizure_count"],
        "window_count_only": ["window_count"],
        "metadata_combined": ["center", "channel_count", "seizure_count", "window_count"],
    }
    metric_rows = []
    prediction_rows = []
    unavailable: dict[str, str] = {}
    for name, columns in definitions.items():
        missing = [column for column in columns if column not in manifest]
        if missing:
            unavailable[name] = f"missing patient-manifest columns: {missing}"
            metric_rows.append({"shortcut": name, "status": "unavailable", "undefined_reasons": unavailable[name]})
            continue
        prediction = _cross_validated_metadata_probability(manifest, ledger, columns)
        prediction["shortcut"] = name
        prediction_rows.append(prediction)
        bundle = compute_patient_metrics(prediction["outcome"].to_numpy(), prediction["probability"].to_numpy(), 0.5)
        metric_rows.append({"shortcut": name, "status": "evaluated", **bundle.values, "undefined_reasons": json.dumps(bundle.undefined_reasons, sort_keys=True)})
    for name, columns in {
        "record_duration_only": ["record_duration_sec"],
        "quality_only": ["quality"],
        "sampling_rate_device_only": ["sampling_rate", "device"],
    }.items():
        missing = [column for column in columns if column not in manifest]
        if missing:
            unavailable[name] = f"missing patient-manifest columns: {missing}"
            metric_rows.append({"shortcut": name, "status": "unavailable", "undefined_reasons": unavailable[name]})
        else:
            prediction = _cross_validated_metadata_probability(manifest, ledger, columns)
            prediction["shortcut"] = name
            prediction_rows.append(prediction)
            bundle = compute_patient_metrics(prediction["outcome"].to_numpy(), prediction["probability"].to_numpy(), 0.5)
            metric_rows.append({"shortcut": name, "status": "evaluated", **bundle.values, "undefined_reasons": json.dumps(bundle.undefined_reasons, sort_keys=True)})

    replay_metric_paths = [path for path in root.rglob("shortcut_replay_metrics.csv") if path.parent != root]
    replay_prediction_paths = [path for path in root.rglob("shortcut_replay_predictions.csv") if path.parent != root]
    replay_metrics = pd.concat([pd.read_csv(path) for path in replay_metric_paths], ignore_index=True) if replay_metric_paths else pd.DataFrame()
    replay_predictions = pd.concat([pd.read_csv(path) for path in replay_prediction_paths], ignore_index=True) if replay_prediction_paths else pd.DataFrame()
    robustness_metrics = pd.DataFrame()
    if replay_metrics.empty:
        unavailable["frozen_model_replay"] = "no completed frozen-checkpoint replay artifacts"
    else:
        robustness_metrics = replay_metrics.groupby("perturbation", as_index=False).agg(
            macro_f1=("macro_f1", "mean"),
            auroc=("auroc", "mean"),
            brier=("brier", "mean"),
        )
        robustness_metrics["status"] = "evaluated"
        replay_predictions = replay_predictions.rename(columns={"perturbation": "shortcut"})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(root / "outcome_shortcut_metrics.csv", index=False)
    robustness_metrics.to_csv(root / "outcome_robustness_metrics.csv", index=False)
    (pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()).to_csv(root / "outcome_shortcut_predictions.csv", index=False)
    replay_predictions.to_csv(root / "outcome_robustness_predictions.csv", index=False)
    main_metrics_path = root / "reports" / "outcome_metrics_summary.csv"
    main_macro = float(pd.read_csv(main_metrics_path)["macro_f1"].max()) if main_metrics_path.exists() else float("nan")
    shortcut_risk, shortcut_models_considered = compute_shortcut_risk(main_macro, metrics)
    evaluated_macro = pd.to_numeric(metrics.loc[metrics["shortcut"].isin(shortcut_models_considered), "macro_f1"], errors="coerce")
    shortcut_macro = float(evaluated_macro.max()) if evaluated_macro.notna().any() else float("nan")
    audit = {
        "shortcut_risk": shortcut_risk,
        "main_best_macro_f1": main_macro,
        "metadata_best_macro_f1": shortcut_macro,
        "frozen_replay_file_count": len(replay_prediction_paths),
        "unavailable": unavailable,
        "shortcut_models_considered": shortcut_models_considered,
        "robustness_perturbations_excluded_from_shortcut_risk": sorted(replay_predictions["shortcut"].astype(str).unique().tolist()) if not replay_predictions.empty else [],
        "model_input_policy": "center, counts, quality, duration, sampling rate, and device are excluded from the main model",
    }
    (root / "outcome_shortcut_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    embedding_path = root / "reports" / "outcome_patient_embeddings.npy"
    embedding_index_path = root / "reports" / "outcome_patient_embeddings_index.csv"
    center_rows = []
    if embedding_path.exists() and embedding_index_path.exists():
        embeddings = np.load(embedding_path)
        embedding_index = pd.read_csv(embedding_index_path)
        if len(embeddings) == len(embedding_index):
            embedding_frame = pd.DataFrame(embeddings)
            embedding_frame["subject_id"] = embedding_index["subject_id"].astype(str).to_numpy()
            averaged = embedding_frame.groupby("subject_id", as_index=False).mean(numeric_only=True)
            probe = averaged.merge(manifest[["subject_id", "center"]], on="subject_id", validate="one_to_one").merge(ledger[["subject_id", "fold_idx"]], on="subject_id", validate="one_to_one")
            feature_columns = [column for column in averaged.columns if column != "subject_id"]
            for fold in sorted(probe["fold_idx"].unique()):
                train = probe["fold_idx"] != fold
                test = ~train
                if probe.loc[train, "center"].nunique() < 2:
                    continue
                estimator = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
                estimator.fit(probe.loc[train, feature_columns], probe.loc[train, "center"])
                predicted_center = estimator.predict(probe.loc[test, feature_columns])
                for subject, actual, predicted in zip(probe.loc[test, "subject_id"], probe.loc[test, "center"], predicted_center):
                    center_rows.append({"subject_id": subject, "fold_idx": int(fold), "actual_center": actual, "predicted_center": predicted, "correct": int(actual == predicted), "status": "evaluated"})
    if not center_rows:
        center_rows.append({"status": "unavailable", "reason": "frozen patient embeddings or multi-center train folds unavailable"})
    pd.DataFrame(center_rows).to_csv(root / "outcome_center_probe.csv", index=False)

    def replay_export(name: str, output_name: str) -> None:
        selected = replay_predictions[replay_predictions.get("shortcut", pd.Series(dtype=str)) == name].copy() if not replay_predictions.empty else pd.DataFrame()
        if selected.empty:
            selected = pd.DataFrame([{"status": "unavailable", "reason": unavailable.get("frozen_model_replay", "perturbation unavailable")}])
        else:
            selected["status"] = "evaluated"
        selected.to_csv(root / output_name, index=False)

    replay_export("channel_dropout_10", "outcome_robustness_channel_dropout_10.csv")
    channel_dropout = replay_predictions[replay_predictions["shortcut"].isin(["channel_dropout_10", "channel_dropout_20", "channel_dropout_30"])].copy() if not replay_predictions.empty else pd.DataFrame()
    (channel_dropout if not channel_dropout.empty else pd.DataFrame([{"status": "unavailable", "reason": unavailable.get("frozen_model_replay", "replay unavailable")}])).to_csv(root / "outcome_robustness_channel_dropout.csv", index=False)
    replay_export("seizure_dropout", "outcome_robustness_seizure_dropout.csv")
    replay_export("window_order_shuffle", "outcome_window_shuffle.csv")
    replay_export("channel_permutation", "outcome_channel_permutation.csv")
    return audit


__all__ = ["compute_shortcut_risk", "perturb_model_input", "replay_model_perturbations", "run_shortcut_audit"]
