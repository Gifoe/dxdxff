from __future__ import annotations

from collections import defaultdict
import copy
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .evidence_views import (
    EvidenceNormalizer,
    MultiViewEvidenceNormalizer,
    b0_self_reference_features,
    fit_normalizer,
    physics_state_features,
)
from .quality import quality_weight, resolve_quality_label

CENTER_TO_ID = {
    "hup": 0,
    "lzu": 1,
    "multicenter": 2,
    "pediatric": 3,
    "unknown": 4,
}


def _canonical_center(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if raw in CENTER_TO_ID:
        return raw
    if raw.startswith("hup"):
        return "hup"
    if raw.startswith("lzu"):
        return "lzu"
    if raw.startswith("ped"):
        return "pediatric"
    if raw in {"multi", "multi-center", "multi_center"} or raw.startswith("multicenter"):
        return "multicenter"
    return "unknown"


def infer_patient_center(subject_id: str, patient_meta: dict[str, Any], samples: Sequence[dict[str, Any]] | None = None) -> str:
    for source in (patient_meta, *((samples or [])[:1])):
        for key in ("source_center", "center", "source_dataset"):
            center = _canonical_center(source.get(key) if isinstance(source, dict) else None)
            if center != "unknown":
                return center
    if ":" in str(subject_id):
        return _canonical_center(str(subject_id).split(":", 1)[0])
    return "unknown"


def center_to_id(center: str) -> int:
    return int(CENTER_TO_ID.get(_canonical_center(center), CENTER_TO_ID["unknown"]))


def _as_window_tensors(sample: dict[str, Any], feature_dim_fallback: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
    window_adjacency = np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32)
    window_centers = np.asarray(sample.get("window_relative_centers_sec", np.zeros((0,))), dtype=np.float32)

    if window_features.ndim == 3 and window_features.shape[0] > 0 and window_features.shape[-1] > 0:
        num_windows, num_channels = int(window_features.shape[0]), int(window_features.shape[1])
        if (
            window_adjacency.ndim != 3
            or window_adjacency.shape[0] != num_windows
            or window_adjacency.shape[1] != num_channels
            or window_adjacency.shape[2] != num_channels
        ):
            window_adjacency = np.zeros((num_windows, num_channels, num_channels), dtype=np.float32)
        if window_centers.shape[0] != window_features.shape[0]:
            window_centers = np.arange(window_features.shape[0], dtype=np.float32)
        return window_features, window_adjacency.astype(np.float32, copy=False), window_centers.astype(np.float32, copy=False)

    num_channels = int(np.asarray(sample.get("labels", []), dtype=np.float32).shape[0])
    features = np.zeros((1, num_channels, max(1, int(feature_dim_fallback))), dtype=np.float32)
    adjacency = np.zeros((features.shape[0], num_channels, num_channels), dtype=np.float32)
    centers = np.zeros((features.shape[0],), dtype=np.float32)
    return features, adjacency, centers


def _args_with_window_feature_names(args: Any | None, sample: dict[str, Any]) -> Any | None:
    feature_names = sample.get("window_feature_names")
    if feature_names is None:
        return args
    view_args = copy.copy(args) if args is not None else type("Args", (), {})()
    setattr(view_args, "window_feature_names", list(feature_names))
    return view_args


def _prepared_views(sample: dict[str, Any], args: Any | None = None) -> dict[str, np.ndarray]:
    raw_features, adjacency, centers = _as_window_tensors(sample, feature_dim_fallback=1)
    view_args = _args_with_window_feature_names(args, sample)
    return {
        "b0": b0_self_reference_features(raw_features, centers, view_args),
        "physics": physics_state_features(raw_features, centers, view_args),
        "adjacency": np.asarray(adjacency, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float32),
    }


def fit_window_tensor_normalizer(window_samples: Iterable[dict[str, Any]], args: Any | None = None) -> MultiViewEvidenceNormalizer:
    samples = list(window_samples)
    views = [_prepared_views(sample, args=args) for sample in samples]
    return MultiViewEvidenceNormalizer(
        b0=fit_normalizer((view["b0"] for view in views)),
        physics=fit_normalizer((view["physics"] for view in views)),
    )


def build_patient_examples(
    window_samples: Sequence[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    *,
    normalizer: EvidenceNormalizer | MultiViewEvidenceNormalizer | None = None,
    subject_ids: Sequence[str] | None = None,
    args: Any | None = None,
    raw_alignment_store: Any | None = None,
    causal_propagation_store: Any | None = None,
    v3_anchor_store: Any | None = None,
) -> list[dict[str, Any]]:
    selected_subjects = set(subject_ids) if subject_ids is not None else None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in window_samples:
        subject_id = str(sample["subject_id"])
        if selected_subjects is not None and subject_id not in selected_subjects:
            continue
        grouped[subject_id].append(sample)

    positive_label = str(getattr(args, "positive_label", "nez") if args is not None else "nez").lower()
    examples: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        if subject_id not in patient_index:
            continue
        patient_meta = patient_index[subject_id]
        canonical_channels = list(patient_meta["canonical_channels"])
        channel_to_idx = {name: idx for idx, name in enumerate(canonical_channels)}
        labels_ez = np.asarray(patient_meta["labels"], dtype=np.float32)
        labels_nez = np.where(labels_ez >= 0.0, 1.0 - labels_ez, -1.0).astype(np.float32, copy=False)
        labels = labels_nez if positive_label == "nez" else labels_ez
        channel_mask = np.asarray(patient_meta.get("label_mask", np.ones(len(canonical_channels), dtype=bool)), dtype=bool)
        valid_label_mask = channel_mask & (labels_ez >= 0.0)
        valid_channel_count = int(np.sum(valid_label_mask))
        ez_channel_count = int(np.sum((labels_ez > 0.5) & valid_label_mask))
        ez_fraction = float(ez_channel_count / max(valid_channel_count, 1))
        center = infer_patient_center(subject_id, patient_meta, grouped[subject_id])
        num_patient_channels = len(canonical_channels)
        use_diffusion_residual = bool(getattr(args, "use_diffusion_residual", False)) if args is not None else False

        b0_items: list[np.ndarray] = []
        physics_items: list[np.ndarray] = []
        diffusion_adjacency_items: list[np.ndarray] = []
        window_centers_items: list[np.ndarray] = []
        window_masks: list[np.ndarray] = []
        seizure_channel_masks: list[np.ndarray] = []
        run_ids: list[str] = []
        sample_ids: list[str] = []
        record_quality_labels: list[str] = []
        record_quality_weights: list[float] = []
        raw_items: list[np.ndarray] = []
        raw_masks: list[np.ndarray] = []
        raw_channel_masks: list[np.ndarray] = []
        use_quality_weighting = bool(getattr(args, "use_edf_quality_weighting", False)) if args is not None else False
        quality_field = str(getattr(args, "edf_quality_field", "") if args is not None else "").strip()
        review_weight = float(getattr(args, "review_weight", 0.7) if args is not None else 0.7)
        poor_weight = float(getattr(args, "poor_weight", 0.0) if args is not None else 0.0)

        for sample in grouped[subject_id]:
            local_channels = list(sample["channel_names_norm"])
            local_to_patient = [channel_to_idx.get(channel_name) for channel_name in local_channels]
            views = _prepared_views(sample, args=args)
            b0 = normalizer.transform_b0(views["b0"]) if hasattr(normalizer, "transform_b0") else normalizer.transform(views["b0"]) if normalizer is not None else views["b0"]
            physics = (
                normalizer.transform_physics(views["physics"])
                if hasattr(normalizer, "transform_physics")
                else views["physics"]
            )
            adjacency = np.asarray(views["adjacency"], dtype=np.float32)

            t = int(b0.shape[0])
            aligned_b0 = np.zeros((t, num_patient_channels, b0.shape[-1]), dtype=np.float32)
            aligned_physics = np.zeros((t, num_patient_channels, physics.shape[-1]), dtype=np.float32)
            aligned_adjacency = np.zeros((t, num_patient_channels, num_patient_channels), dtype=np.float32)
            aligned_channel_mask = np.zeros((num_patient_channels,), dtype=bool)

            for local_idx, patient_idx in enumerate(local_to_patient):
                if patient_idx is None:
                    continue
                aligned_b0[:, patient_idx, :] = b0[:, local_idx, :]
                aligned_physics[:, patient_idx, :] = physics[:, local_idx, :]
                aligned_channel_mask[patient_idx] = True
            for src_local_idx, src_patient_idx in enumerate(local_to_patient):
                if src_patient_idx is None:
                    continue
                for dst_local_idx, dst_patient_idx in enumerate(local_to_patient):
                    if dst_patient_idx is None:
                        continue
                    aligned_adjacency[:, src_patient_idx, dst_patient_idx] = adjacency[:, src_local_idx, dst_local_idx]

            b0_items.append(aligned_b0)
            physics_items.append(aligned_physics)
            if use_diffusion_residual:
                diffusion_adjacency_items.append(aligned_adjacency)
            window_centers_items.append(np.asarray(views["centers"], dtype=np.float32))
            window_masks.append(np.ones((t,), dtype=bool))
            seizure_channel_masks.append(aligned_channel_mask)
            if raw_alignment_store is not None:
                local_raw, local_raw_mask, _ = raw_alignment_store.aligned_windows(sample)
                if local_raw.shape[:2] != (t, len(local_channels)):
                    raise ValueError(f"Raw alignment shape mismatch for {subject_id}/{sample['run_id']}")
                aligned_raw = np.zeros((t, num_patient_channels, local_raw.shape[-1]), dtype=np.float32)
                aligned_raw_mask = np.zeros((t, num_patient_channels), dtype=bool)
                for local_idx, patient_idx in enumerate(local_to_patient):
                    if patient_idx is None:
                        continue
                    aligned_raw[:, patient_idx] = local_raw[:, local_idx]
                    aligned_raw_mask[:, patient_idx] = local_raw_mask[:, local_idx]
                raw_items.append(aligned_raw)
                raw_masks.append(aligned_raw_mask)
                raw_channel_masks.append(aligned_raw_mask.any(axis=0))
            run_ids.append(str(sample["run_id"]))
            sample_ids.append(str(sample["sample_id"]))
            if use_quality_weighting:
                label = resolve_quality_label(sample, quality_field)
                weight = quality_weight(label, review_weight=review_weight, poor_weight=poor_weight)
            else:
                label = "good"
                weight = 1.0
            record_quality_labels.append(label)
            record_quality_weights.append(float(weight))

        if not b0_items:
            continue
        examples.append(
            {
                "subject_id": subject_id,
                "center": center,
                "center_id": center_to_id(center),
                "ez_fraction": ez_fraction,
                "valid_channel_count": valid_channel_count,
                "ez_channel_count": ez_channel_count,
                "canonical_channels": canonical_channels,
                "channel_meta": list(patient_meta.get("channel_meta", [])),
                "labels": labels,
                "labels_nez": labels_nez,
                "labels_ez": labels_ez,
                "label_semantics": "1=NEZ,0=EZ" if positive_label == "nez" else "1=EZ,0=NEZ",
                "channel_mask": channel_mask,
                "b0_features": b0_items,
                "features": b0_items,
                "physics_features": physics_items,
                "window_centers": window_centers_items,
                "window_mask": window_masks,
                "seizure_channel_mask": seizure_channel_masks,
                "run_ids": run_ids,
                "sample_ids": sample_ids,
                "record_quality_labels": record_quality_labels,
                "record_quality_weights": np.asarray(record_quality_weights, dtype=np.float32),
            }
        )
        if causal_propagation_store is not None:
            causal_features, causal_valid, causal_audit = causal_propagation_store.align(
                subject_id, canonical_channels
            )
            examples[-1].update({
                "causal_propagation_features": causal_features,
                "causal_propagation_valid": causal_valid,
                "causal_propagation_audit": causal_audit,
                "cp_valid_window_fraction": causal_audit["cp_valid_window_fraction"],
                "cp_valid_seizure_count": causal_audit["cp_valid_seizure_count"],
                "cp_mean_var_stability": causal_audit["cp_mean_var_stability"],
            })
        if v3_anchor_store is not None:
            v3_score, v3_logit, v3_valid, v3_audit = v3_anchor_store.align(subject_id, canonical_channels)
            examples[-1].update({
                "v3_score_nez": v3_score,
                "v3_logit_nez": v3_logit,
                "v3_anchor_valid": v3_valid,
                "v3_anchor_audit": v3_audit,
            })
        if raw_alignment_store is not None:
            examples[-1].update({
                "raw_windows": raw_items,
                "raw_window_mask": raw_masks,
                "raw_seizure_channel_mask": raw_channel_masks,
                "raw_alignment_metadata": {"source": "compound_key", "raw_target_samples": int(raw_alignment_store.raw_target_samples)},
            })
        if use_diffusion_residual:
            examples[-1]["diffusion_adjacency"] = diffusion_adjacency_items
    return examples


class PatientNeuroEZCDataset(Dataset):
    def __init__(self, patient_examples: Sequence[dict[str, Any]]) -> None:
        self.patient_examples = list(patient_examples)

    def __len__(self) -> int:
        return len(self.patient_examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.patient_examples[index]


def _allocate_view(batch: Sequence[dict[str, Any]], key: str, max_seizures: int, max_windows: int, max_channels: int) -> torch.Tensor:
    dim = max(int(item[key][s].shape[-1]) for item in batch for s in range(len(item[key])))
    return torch.zeros((len(batch), max_seizures, max_windows, max_channels, dim), dtype=torch.float32)


def collate_patient_ez_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("collate_patient_ez_batch received an empty batch.")

    batch_size = len(batch)
    max_seizures = max(len(item["b0_features"]) for item in batch)
    max_windows = max(int(arr.shape[0]) for item in batch for arr in item["b0_features"])
    max_channels = max(int(item["labels"].shape[0]) for item in batch)

    b0_features = _allocate_view(batch, "b0_features", max_seizures, max_windows, max_channels)
    physics_features = _allocate_view(batch, "physics_features", max_seizures, max_windows, max_channels)
    has_diffusion_adjacency = any("diffusion_adjacency" in item for item in batch)
    diffusion_adjacency = (
        torch.zeros((batch_size, max_seizures, max_windows, max_channels, max_channels), dtype=torch.float32)
        if has_diffusion_adjacency
        else None
    )
    labels = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_nez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_ez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    seizure_mask = torch.zeros((batch_size, max_seizures), dtype=torch.bool)
    seizure_channel_mask = torch.zeros((batch_size, max_seizures, max_channels), dtype=torch.bool)
    window_mask = torch.zeros((batch_size, max_seizures, max_windows), dtype=torch.bool)
    window_centers = torch.zeros((batch_size, max_seizures, max_windows), dtype=torch.float32)
    use_raw = any("raw_windows" in item for item in batch)
    raw_target_samples = max((int(np.asarray(raw).shape[-1]) for item in batch for raw in item.get("raw_windows", [])), default=1)
    raw_windows = torch.zeros((batch_size, max_seizures, max_channels, max_windows, raw_target_samples), dtype=torch.float32) if use_raw else None
    raw_window_mask = torch.zeros((batch_size, max_seizures, max_channels, max_windows), dtype=torch.bool) if use_raw else None
    raw_seizure_channel_mask = torch.zeros((batch_size, max_seizures, max_channels), dtype=torch.bool) if use_raw else None
    use_causal = any("causal_propagation_features" in item for item in batch)
    causal_propagation_features = torch.zeros((batch_size, max_channels, 6), dtype=torch.float32) if use_causal else None
    causal_propagation_valid = torch.zeros((batch_size, max_channels), dtype=torch.bool) if use_causal else None
    cp_valid_window_fraction = torch.zeros((batch_size, max_channels), dtype=torch.float32) if use_causal else None
    cp_valid_seizure_count = torch.zeros((batch_size, max_channels), dtype=torch.float32) if use_causal else None
    cp_mean_var_stability = torch.zeros((batch_size, max_channels), dtype=torch.float32) if use_causal else None
    use_v3 = any("v3_logit_nez" in item for item in batch)
    v3_score_nez = torch.zeros((batch_size, max_channels), dtype=torch.float32) if use_v3 else None
    v3_logit_nez = torch.zeros((batch_size, max_channels), dtype=torch.float32) if use_v3 else None
    v3_anchor_valid = torch.zeros((batch_size, max_channels), dtype=torch.bool) if use_v3 else None

    subject_ids: list[str] = []
    canonical_channels: list[list[str]] = []
    channel_meta: list[list[dict[str, Any]]] = []
    run_ids: list[list[str]] = []
    sample_ids: list[list[str]] = []
    record_quality_labels: list[list[str]] = []
    label_semantics: list[str] = []
    centers: list[str] = []
    center_ids = torch.zeros((batch_size,), dtype=torch.long)
    ez_fraction = torch.zeros((batch_size,), dtype=torch.float32)
    valid_channel_count = torch.zeros((batch_size,), dtype=torch.long)
    ez_channel_count = torch.zeros((batch_size,), dtype=torch.long)
    record_quality_weight = torch.ones((batch_size, max_seizures), dtype=torch.float32)

    for batch_idx, item in enumerate(batch):
        c = int(item["labels"].shape[0])
        labels[batch_idx, :c] = torch.as_tensor(item["labels"], dtype=torch.float32)
        labels_nez[batch_idx, :c] = torch.as_tensor(item["labels_nez"], dtype=torch.float32)
        labels_ez[batch_idx, :c] = torch.as_tensor(item["labels_ez"], dtype=torch.float32)
        channel_mask[batch_idx, :c] = torch.as_tensor(item["channel_mask"], dtype=torch.bool)
        subject_ids.append(str(item["subject_id"]))
        canonical_channels.append(list(item["canonical_channels"]))
        channel_meta.append(list(item.get("channel_meta", [])))
        run_ids.append(list(item.get("run_ids", [])))
        sample_ids.append(list(item.get("sample_ids", [])))
        record_quality_labels.append(list(item.get("record_quality_labels", ["good"] * len(item["b0_features"]))))
        label_semantics.append(str(item.get("label_semantics", "1=NEZ,0=EZ")))
        center = _canonical_center(item.get("center", "unknown"))
        centers.append(center)
        center_ids[batch_idx] = int(item.get("center_id", center_to_id(center)))
        ez_fraction[batch_idx] = float(item.get("ez_fraction", 0.0))
        valid_channel_count[batch_idx] = int(item.get("valid_channel_count", int(item["channel_mask"].sum())))
        ez_channel_count[batch_idx] = int(item.get("ez_channel_count", int((item["labels_ez"] > 0.5).sum())))
        if causal_propagation_features is not None and "causal_propagation_features" in item:
            cp = np.asarray(item["causal_propagation_features"], dtype=np.float32)
            cp_valid = np.asarray(item["causal_propagation_valid"], dtype=bool)
            if cp.shape != (c, 6) or cp_valid.shape != (c,):
                raise ValueError("Causal propagation tensors must align with canonical patient channels")
            if not np.isfinite(cp).all():
                raise ValueError("Causal propagation model features must be finite")
            causal_propagation_features[batch_idx, :c] = torch.as_tensor(cp)
            causal_propagation_valid[batch_idx, :c] = torch.as_tensor(cp_valid) & channel_mask[batch_idx, :c]
            cp_valid_window_fraction[batch_idx, :c] = torch.as_tensor(item["cp_valid_window_fraction"], dtype=torch.float32)
            cp_valid_seizure_count[batch_idx, :c] = torch.as_tensor(item["cp_valid_seizure_count"], dtype=torch.float32)
            cp_mean_var_stability[batch_idx, :c] = torch.as_tensor(item["cp_mean_var_stability"], dtype=torch.float32)
        if v3_score_nez is not None and "v3_score_nez" in item:
            for name, tensor in (("v3_score_nez", v3_score_nez), ("v3_logit_nez", v3_logit_nez)):
                values = np.asarray(item[name], dtype=np.float32)
                if values.shape != (c,) or not np.isfinite(values).all():
                    raise ValueError(f"{name} must be finite and align with canonical channels")
                tensor[batch_idx, :c] = torch.as_tensor(values)
            valid = np.asarray(item["v3_anchor_valid"], dtype=bool)
            if valid.shape != (c,):
                raise ValueError("v3_anchor_valid must align with canonical channels")
            v3_anchor_valid[batch_idx, :c] = torch.as_tensor(valid) & channel_mask[batch_idx, :c]

        for seizure_idx, b0 in enumerate(item["b0_features"]):
            physics = np.asarray(item["physics_features"][seizure_idx], dtype=np.float32)
            seizure_centers = np.asarray(item["window_centers"][seizure_idx], dtype=np.float32)
            mask = np.asarray(item["seizure_channel_mask"][seizure_idx], dtype=bool)
            win_mask = np.asarray(item["window_mask"][seizure_idx], dtype=bool)
            b0 = np.asarray(b0, dtype=np.float32)
            t, current_c = int(b0.shape[0]), int(b0.shape[1])
            b0_features[batch_idx, seizure_idx, :t, :current_c, : b0.shape[-1]] = torch.as_tensor(b0)
            physics_features[batch_idx, seizure_idx, :t, :current_c, : physics.shape[-1]] = torch.as_tensor(physics)
            if diffusion_adjacency is not None and "diffusion_adjacency" in item:
                adjacency = np.asarray(item["diffusion_adjacency"][seizure_idx], dtype=np.float32)
                diffusion_adjacency[batch_idx, seizure_idx, :t, :current_c, :current_c] = torch.as_tensor(adjacency)
            window_centers[batch_idx, seizure_idx, :t] = torch.as_tensor(seizure_centers[:t], dtype=torch.float32)
            seizure_channel_mask[batch_idx, seizure_idx, :current_c] = torch.as_tensor(mask, dtype=torch.bool)
            window_mask[batch_idx, seizure_idx, :t] = torch.as_tensor(win_mask, dtype=torch.bool)
            seizure_mask[batch_idx, seizure_idx] = True
            if raw_windows is not None and "raw_windows" in item:
                raw = np.asarray(item["raw_windows"][seizure_idx], dtype=np.float32)
                raw_mask = np.asarray(item["raw_window_mask"][seizure_idx], dtype=bool)
                if raw.shape[:2] != (t, current_c) or raw_mask.shape != (t, current_c):
                    raise ValueError("raw window tensors must align with feature windows and canonical channels")
                raw_windows[batch_idx, seizure_idx, :current_c, :t, : raw.shape[-1]] = torch.as_tensor(raw.transpose(1, 0, 2))
                raw_window_mask[batch_idx, seizure_idx, :current_c, :t] = torch.as_tensor(raw_mask.T)
                raw_seizure_channel_mask[batch_idx, seizure_idx, :current_c] = torch.as_tensor(raw_mask.any(axis=0), dtype=torch.bool)
            weights = np.asarray(item.get("record_quality_weights", np.ones(len(item["b0_features"]), dtype=np.float32)), dtype=np.float32)
            if seizure_idx < weights.shape[0]:
                record_quality_weight[batch_idx, seizure_idx] = float(weights[seizure_idx])

    collated = {
        "features": b0_features,
        "b0_features": b0_features,
        "physics_features": physics_features,
        "window_centers": window_centers,
        "labels": labels,
        "labels_nez": labels_nez,
        "labels_ez": labels_ez,
        "channel_mask": channel_mask,
        "seizure_mask": seizure_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "window_mask": window_mask,
        "subject_id": subject_ids,
        "center": centers,
        "center_id": center_ids,
        "ez_fraction": ez_fraction,
        "valid_channel_count": valid_channel_count,
        "ez_channel_count": ez_channel_count,
        "canonical_channels": canonical_channels,
        "channel_meta": channel_meta,
        "run_ids": run_ids,
        "sample_ids": sample_ids,
        "record_quality_label": record_quality_labels,
        "record_quality_weight": record_quality_weight,
        "record_quality_weight_active": record_quality_weight.clone(),
        "label_semantics": label_semantics,
    }
    if diffusion_adjacency is not None:
        collated["diffusion_adjacency"] = diffusion_adjacency
    if raw_windows is not None:
        collated.update({
            "raw_windows": raw_windows,
            "raw_window_mask": raw_window_mask,
            "raw_seizure_channel_mask": raw_seizure_channel_mask,
            "raw_channel_available": raw_window_mask.any(dim=(1, 3)),
        })
    if causal_propagation_features is not None:
        collated.update({
            "causal_propagation_features": causal_propagation_features,
            "causal_propagation_valid": causal_propagation_valid,
            "cp_valid_window_fraction": cp_valid_window_fraction,
            "cp_valid_seizure_count": cp_valid_seizure_count,
            "cp_mean_var_stability": cp_mean_var_stability,
        })
    if v3_score_nez is not None:
        collated.update({
            "v3_score_nez": v3_score_nez,
            "v3_logit_nez": v3_logit_nez,
            "v3_anchor_valid": v3_anchor_valid,
        })
    return collated


__all__ = [
    "EvidenceNormalizer",
    "CENTER_TO_ID",
    "PatientNeuroEZCDataset",
    "build_patient_examples",
    "center_to_id",
    "collate_patient_ez_batch",
    "fit_window_tensor_normalizer",
    "infer_patient_center",
]
