from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from outcome_hifos.collate import collate_outcome_patients
from outcome_hifos.dataset import OutcomePatientExample


@torch.no_grad()
def export_interpretability(
    model: torch.nn.Module,
    examples: Sequence[OutcomePatientExample],
    output_dir: str | Path,
    device: torch.device,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(list(examples), batch_size=1, shuffle=False, collate_fn=partial(collate_outcome_patients, padding_value=0.0))
    seizure_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    responsibility_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    embedding_subjects: list[str] = []
    plan_payload: dict[str, np.ndarray] = {}
    model.eval()
    for batch in loader:
        subject_id = str(batch["subject_id"][0])
        model_input = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch["model_input"].items()}
        result = model(model_input, persist_diagnostics=True)
        patient_embedding = result["patient_embedding"][0].detach().cpu().numpy().astype(np.float32)
        embeddings.append(patient_embedding)
        embedding_subjects.append(subject_id)
        seizure_attention = result.get("seizure_attention")
        if torch.is_tensor(seizure_attention):
            for seizure_index, value in enumerate(seizure_attention[0].detach().cpu().reshape(-1).tolist()):
                seizure_rows.append({"subject_id": subject_id, "seizure_idx": seizure_index, "contribution": float(value)})
        focality = result.get("focality_descriptors")
        transport_costs = result.get("transport_costs")
        if torch.is_tensor(focality):
            focality_np = focality[0].detach().cpu().numpy()
            for seizure_index in range(focality_np.shape[0]):
                for window_index in range(focality_np.shape[1]):
                    window_rows.append(
                        {
                            "subject_id": subject_id,
                            "seizure_idx": seizure_index,
                            "window_idx": window_index,
                            "focality_norm": float(np.linalg.norm(focality_np[seizure_index, window_index])),
                            "transport_cost_to_next": float(transport_costs[0, seizure_index, window_index].cpu()) if torch.is_tensor(transport_costs) and window_index < transport_costs.shape[2] else float("nan"),
                        }
                    )
        responsibilities = result.get("responsibilities")
        if torch.is_tensor(responsibilities):
            resp = responsibilities[0].detach().cpu().numpy()
            mean_resp = resp.mean(axis=(0, 1))
            channels = batch["canonical_channels"][0]
            for channel_index, channel_name in enumerate(channels):
                for core_index in range(mean_resp.shape[-1]):
                    responsibility_rows.append(
                        {
                            "subject_id": subject_id,
                            "channel_name": channel_name,
                            "core_idx": core_index if core_index < mean_resp.shape[-1] - 1 else "background",
                            "mean_responsibility": float(mean_resp[channel_index, core_index]),
                        }
                    )
        anchors = result.get("core_anchors")
        if torch.is_tensor(anchors):
            for core_index, vector in enumerate(anchors[0].detach().cpu().numpy()):
                for dimension, value in enumerate(vector):
                    anchor_rows.append({"subject_id": subject_id, "core_idx": core_index, "dimension": dimension, "value": float(value)})
        masses = result.get("core_masses")
        if torch.is_tensor(masses):
            mass_np = masses[0].detach().cpu().numpy()
            for seizure_index in range(mass_np.shape[0]):
                for window_index in range(mass_np.shape[1]):
                    for core_index in range(mass_np.shape[2]):
                        mass_rows.append({"subject_id": subject_id, "seizure_idx": seizure_index, "window_idx": window_index, "core_idx": core_index, "mass": float(mass_np[seizure_index, window_index, core_index])})
        descriptors = result.get("transport_descriptors")
        if torch.is_tensor(descriptors):
            descriptor_np = descriptors[0].detach().cpu().numpy()
            for seizure_index in range(descriptor_np.shape[0]):
                for interval_index in range(descriptor_np.shape[1]):
                    row = {"subject_id": subject_id, "seizure_idx": seizure_index, "interval_idx": interval_index}
                    row.update({f"descriptor_{dimension}": float(value) for dimension, value in enumerate(descriptor_np[seizure_index, interval_index])})
                    descriptor_rows.append(row)
        recurrence = result.get("recurrence_distances")
        if torch.is_tensor(recurrence):
            recurrence_np = recurrence[0].detach().cpu().numpy()
            for first in range(recurrence_np.shape[0]):
                for second in range(first + 1, recurrence_np.shape[1]):
                    recurrence_rows.append({"subject_id": subject_id, "seizure_a": first, "seizure_b": second, "uot_distance": float(recurrence_np[first, second])})
        plans = result.get("transport_plans")
        if torch.is_tensor(plans):
            plan_payload[subject_id] = plans[0].detach().cpu().numpy().astype(np.float32)

    pd.DataFrame(seizure_rows, columns=["subject_id", "seizure_idx", "contribution"]).to_csv(output / "outcome_seizure_contributions.csv", index=False)
    pd.DataFrame(window_rows, columns=["subject_id", "seizure_idx", "window_idx", "focality_norm", "transport_cost_to_next"]).to_csv(output / "outcome_window_contributions.csv", index=False)
    pd.DataFrame(responsibility_rows, columns=["subject_id", "channel_name", "core_idx", "mean_responsibility"]).to_csv(output / "outcome_channel_responsibilities.csv", index=False)
    pd.DataFrame(anchor_rows, columns=["subject_id", "core_idx", "dimension", "value"]).to_csv(output / "outcome_core_anchors.csv", index=False)
    pd.DataFrame(mass_rows, columns=["subject_id", "seizure_idx", "window_idx", "core_idx", "mass"]).to_csv(output / "outcome_core_mass_trajectories.csv", index=False)
    descriptor_columns = ["subject_id", "seizure_idx", "interval_idx", *[f"descriptor_{index}" for index in range(10)]]
    pd.DataFrame(descriptor_rows, columns=descriptor_columns).to_csv(output / "outcome_transport_descriptors.csv", index=False)
    pd.DataFrame(recurrence_rows, columns=["subject_id", "seizure_a", "seizure_b", "uot_distance"]).to_csv(output / "outcome_cross_seizure_recurrence.csv", index=False)
    np.savez_compressed(output / "outcome_transport_plans.npz", **plan_payload)
    np.save(output / "outcome_patient_embeddings.npy", np.stack(embeddings).astype(np.float32))
    pd.DataFrame({"row_idx": range(len(embedding_subjects)), "subject_id": embedding_subjects}).to_csv(output / "outcome_patient_embeddings_index.csv", index=False)


def aggregate_interpretability_artifacts(run_root: str | Path, output_dir: str | Path) -> None:
    root = Path(run_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_names = (
        "outcome_seizure_contributions.csv",
        "outcome_window_contributions.csv",
        "outcome_channel_responsibilities.csv",
        "outcome_core_anchors.csv",
        "outcome_core_mass_trajectories.csv",
        "outcome_transport_descriptors.csv",
        "outcome_cross_seizure_recurrence.csv",
        "outcome_patient_embeddings_index.csv",
    )
    for name in csv_names:
        paths = list(root.rglob(name))
        if paths:
            frames = [pd.read_csv(path) for path in paths]
            non_empty = [frame for frame in frames if not frame.empty]
            (pd.concat(non_empty, ignore_index=True) if non_empty else frames[0]).to_csv(output / name, index=False)
    embedding_paths = list(root.rglob("outcome_patient_embeddings.npy"))
    if embedding_paths:
        np.save(output / "outcome_patient_embeddings.npy", np.concatenate([np.load(path) for path in embedding_paths], axis=0))
    plan_paths = list(root.rglob("outcome_transport_plans.npz"))
    if plan_paths:
        combined: dict[str, np.ndarray] = {}
        for path_index, path in enumerate(plan_paths):
            with np.load(path) as payload:
                for key in payload.files:
                    combined[f"run{path_index}_{key}"] = payload[key]
        np.savez_compressed(output / "outcome_transport_plans.npz", **combined)


__all__ = ["aggregate_interpretability_artifacts", "export_interpretability"]
