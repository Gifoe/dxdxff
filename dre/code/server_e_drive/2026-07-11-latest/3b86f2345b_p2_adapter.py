from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch

from .simple_q10 import compute_simple_q10


_RUNTIME_PACKAGE = "_npam_p23_runtime"
_TOP_LEVEL_SUPPORT = ("graph_spectral_encoder", "patient_channel_ranker", "seizure_aggregator", "temporal_encoder")
_STEP4B_NAMES = (
    "early_high_gamma_slope", "early_line_length_slope", "onset_latency_high_gamma",
    "onset_latency_line_length", "onset_rank_high_gamma", "onset_rank_line_length",
    "high_gamma_top20pct_mean", "line_length_top20pct_mean",
)


def _load_module_from_file(name: str, path: Path) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        existing_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_file == path.resolve():
            return existing
        raise RuntimeError(f"Module {name} already loaded from incompatible runtime: {existing_file}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class P23Runtime:
    """Loads the repository's exact P2_TEMPORAL_Q10 implementation under an alias."""

    def __init__(self, runtime_root: str | Path) -> None:
        self.root = Path(runtime_root).expanduser().resolve()
        if not (self.root / "neuroez_c" / "model.py").exists():
            raise FileNotFoundError(f"P23 runtime root is invalid: {self.root}")
        for module_name in _TOP_LEVEL_SUPPORT:
            _load_module_from_file(module_name, self.root / f"{module_name}.py")
        if _RUNTIME_PACKAGE not in sys.modules:
            package = types.ModuleType(_RUNTIME_PACKAGE)
            package.__path__ = [str(self.root / "neuroez_c")]
            package.__package__ = _RUNTIME_PACKAGE
            sys.modules[_RUNTIME_PACKAGE] = package
        self.model_module = importlib.import_module(f"{_RUNTIME_PACKAGE}.model")
        self.dataset_module = importlib.import_module(f"{_RUNTIME_PACKAGE}.dataset")
        self.ez_features_module = _load_module_from_file("_npam_p23_ez_features", self.root / "ez_features.py")

    def _ensure_step4b_features(self, features: Any, centers: Any) -> tuple[Any, list[str] | None]:
        import numpy as np

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != 20:
            return features, None
        times = np.asarray(centers, dtype=np.float32)
        clinical = self.ez_features_module._clinical_onset_core_features(values, times)
        burst = self.ez_features_module._burstness_features(values, times)
        clinical_index = {name: idx for idx, name in enumerate(self.ez_features_module.CLINICAL_ONSET_CORE_FEATURE_NAMES)}
        burst_index = {name: idx for idx, name in enumerate(self.ez_features_module.BURSTNESS_FEATURE_NAMES)}
        derived = np.concatenate((
            clinical[:, :, [clinical_index[name] for name in _STEP4B_NAMES[:6]]],
            burst[:, :, [burst_index[name] for name in _STEP4B_NAMES[6:]]],
        ), axis=-1).astype(np.float32, copy=False)
        base_names = list(self.ez_features_module.WINDOW_NODE_FEATURE_NAMES)
        return np.concatenate((values, derived), axis=-1), base_names + list(_STEP4B_NAMES)

    def flatten_window_samples(self, run_records: Iterable[dict[str, Any]], subject_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        selected = set(map(str, subject_ids)) if subject_ids is not None else None
        samples: list[dict[str, Any]] = []
        for record in run_records:
            subject = str(record.get("subject_id", ""))
            if selected is not None and subject not in selected:
                continue
            sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
            features, derived_names = self._ensure_step4b_features(sample.get("window_features"), sample.get("window_relative_centers_sec"))
            samples.append({
                "subject_id": subject,
                "run_id": str(record.get("run_id", sample.get("run_id", ""))),
                "sample_id": str(sample.get("sample_id", record.get("run_id", ""))),
                "source_center": record.get("source_center", sample.get("source_center")),
                "center": record.get("center", sample.get("center")),
                "source_dataset": record.get("source_dataset", sample.get("source_dataset")),
                "channel_names_norm": list(record.get("channel_names_norm", sample.get("channel_names_norm", []))),
                "labels": sample.get("labels", record.get("labels", [])),
                "window_features": features,
                "window_adjacency": sample.get("window_adjacency"),
                "window_relative_centers_sec": sample.get("window_relative_centers_sec"),
                "window_feature_names": derived_names or sample.get("window_feature_names", record.get("window_feature_names")),
                "quality_metadata": {},
            })
        return samples

    def fit_normalizer(self, samples: Iterable[dict[str, Any]], args: Any) -> Any:
        return self.dataset_module.fit_window_tensor_normalizer(samples, args=args)

    def build_examples(self, samples: Sequence[dict[str, Any]], patient_index: dict[str, dict[str, Any]], *, normalizer: Any, subject_ids: Sequence[str], args: Any) -> list[dict[str, Any]]:
        return self.dataset_module.build_patient_examples(samples, patient_index, normalizer=normalizer, subject_ids=subject_ids, args=args)

    def collate(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return self.dataset_module.collate_patient_ez_batch(examples)


def load_p2_args(config_path: str | Path, *, feature_cache: str | Path | None = None) -> SimpleNamespace:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    payload.update({
        "use_p23_trn_nez": True,
        "p23_profile": "P2_TEMPORAL_Q10",
        "positive_label": "nez",
        "use_p2_rtc_shift": False,
        "use_p2_atc": False,
        "use_p2_scope_v2": False,
        "use_causal_propagation_residual": False,
        "p23_use_causal": False,
        "outcome_subset": "all",
        "drop_high_ez_fraction_lzu": False,
        "allowed_subjects_ledger": None,
        "allowed_subjects_file": None,
        "require_n_patients": 0,
    })
    if feature_cache is not None:
        payload["window_cache_path"] = str(Path(feature_cache).expanduser())
    return SimpleNamespace(**payload)


def locate_p2_config(checkpoint_root: str | Path, explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    root = Path(checkpoint_root).expanduser()
    candidates = (root / "run_args_p23.json", root / "run_config.json", root.parent / "run_args_p23.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No P2 run_args_p23.json/run_config.json under {root}")


class P2ExportAdapter(torch.nn.Module):
    """One P2 backbone plus a read-only export hook for Task-2 tensors."""

    ALLOWED_BATCH_KEYS = ("b0_features", "physics_features", "window_centers", "window_mask", "seizure_mask", "seizure_channel_mask", "channel_mask")

    def __init__(
        self,
        runtime: P23Runtime,
        args: Any,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        limited_finetune: bool = False,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.args = args
        self.device_value = torch.device(device)
        self.backbone = runtime.model_module.NeuroEZCModel(args)
        try:
            payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(Path(checkpoint_path), map_location="cpu")
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
        self.backbone.load_state_dict(state, strict=True)
        self.checkpoint_outer_fold = int(payload.get("outer_fold", 0) or 0)
        self.backbone.to(self.device_value)
        self._captured: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.backbone.p23_temporal_contrast.register_forward_pre_hook(self._capture_temporal_input)
        self.set_limited_finetune(limited_finetune)

    def _capture_temporal_input(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self._captured = (inputs[0], inputs[1], inputs[2], inputs[3])

    def set_limited_finetune(self, enabled: bool) -> list[str]:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        unfrozen: list[str] = []
        if enabled:
            # Exactly the last learned projection in each P23 temporal,
            # seizure-channel, and cross-seizure evidence branch.
            allowed_fragments = (
                "p23_temporal_contrast.head.3",
                "channel_classifier.classifier.3",
                "p23_fusion.gate.3",
            )
            for name, parameter in self.backbone.named_parameters():
                if any(fragment in name for fragment in allowed_fragments):
                    parameter.requires_grad = True
                    unfrozen.append(name)
        self.unfrozen_parameter_names = unfrozen
        return unfrozen

    @staticmethod
    def _phase_embeddings(window: torch.Tensor, centers: torch.Tensor, window_mask: torch.Tensor, seizure_channel_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(window).all(dim=-1)
        base = window_mask.bool().unsqueeze(-1) & seizure_channel_mask.bool().unsqueeze(2) & finite
        t = centers.unsqueeze(-1)
        masks = torch.stack((base & (t < 0.0), base & (t >= 0.0) & (t <= 10.0), base & (t > 10.0) & (t <= 30.0), base & (t > 30.0)), dim=2)
        expanded = window.unsqueeze(2)
        safe = torch.where(masks.unsqueeze(-1), expanded, torch.zeros_like(expanded))
        count = masks.sum(dim=3).clamp_min(1).unsqueeze(-1).to(window.dtype)
        phase = safe.sum(dim=3) / count
        valid = masks.any(dim=3)
        return phase.masked_fill(~valid.unsqueeze(-1), 0.0), valid

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        inputs = {key: batch[key].to(self.device_value) for key in self.ALLOWED_BATCH_KEYS}
        self._captured = None
        output = self.backbone(inputs)
        if self._captured is None:
            raise RuntimeError("P2 temporal export hook did not capture window embeddings")
        window_embedding, centers, window_mask, seizure_channel_mask = self._captured
        phase_embedding, phase_mask = self._phase_embeddings(window_embedding, centers, window_mask, seizure_channel_mask)
        export = {
            "window_embedding": window_embedding,
            "phase_channel_embedding": phase_embedding,
            "seizure_channel_embedding": output["seizure_channel_embedding"],
            "seizure_nez_logit": output["seizure_nez_logit"],
            "seizure_nez_probability": output["seizure_nez_probability"],
            "patient_channel_embedding": output["patient_channel_embedding"],
            "final_nez_logit": output["final_nez_logit"],
            "final_score_nez": output["final_score_nez"],
            "final_score_ez": output["final_score_ez"],
            "direct_nez_logit": output["direct_nez_logit"],
            "temporal_delta_norm": output["temporal_delta_norm"],
            "delta_onset_norm": output["delta_onset_norm"],
            "delta_spread_norm": output["delta_spread_norm"],
            "window_mask": inputs["window_mask"].bool(),
            "phase_mask": phase_mask,
            "seizure_mask": inputs["seizure_mask"].bool(),
            "seizure_channel_mask": inputs["seizure_channel_mask"].bool(),
            "channel_mask": inputs["channel_mask"].bool(),
        }
        export.update(compute_simple_q10(export["seizure_nez_probability"], export["seizure_mask"], export["seizure_channel_mask"], export["channel_mask"]))
        self.audit_shapes(export)
        return export

    @staticmethod
    def audit_shapes(export: dict[str, torch.Tensor]) -> dict[str, list[int]]:
        window = export["window_embedding"]
        if window.ndim != 5:
            raise ValueError("window_embedding must be [B,S,T,C,D]")
        b, s, t, c, _ = window.shape
        expected = {
            "phase_channel_embedding": (b, s, 4, c),
            "seizure_channel_embedding": (b, s, c),
            "seizure_nez_probability": (b, s, c),
            "patient_channel_embedding": (b, c),
            "window_mask": (b, s, t),
            "phase_mask": (b, s, 4, c),
            "seizure_mask": (b, s),
            "seizure_channel_mask": (b, s, c),
            "channel_mask": (b, c),
        }
        for key, prefix in expected.items():
            if tuple(export[key].shape[: len(prefix)]) != prefix:
                raise ValueError(f"{key} shape {tuple(export[key].shape)} violates prefix {prefix}")
        return {key: list(value.shape) for key, value in export.items() if torch.is_tensor(value)}


__all__ = ["P23Runtime", "P2ExportAdapter", "load_p2_args", "locate_p2_config"]
