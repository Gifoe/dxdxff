from __future__ import annotations

import sys
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class FrozenFMExtractor:
    name: str = "base"
    target_sfreq: int = 200
    window_sec: float = 4.0
    embedding_dim: Optional[int] = None
    loaded: bool = False
    audit_info: Dict[str, Any]

    def load(self, checkpoint_path: str | None, external_repo_path: str | None, device: str):
        raise NotImplementedError

    def encode_batch(self, batch_waveforms: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _comparison_audit_fields(fm_model: str, *, true_pretrained_fm: bool, debug_only: bool, paper_baseline: bool) -> Dict[str, Any]:
    return {
        "fm_model": fm_model,
        "comparison_only_baseline": True,
        "true_pretrained_fm": bool(true_pretrained_fm),
        "debug_only": bool(debug_only),
        "paper_baseline": bool(paper_baseline),
        "frozen_or_finetuned": "frozen",
        "frozen_encoder": True,
        "fine_tuned": False,
        "adapter_tuned": False,
        "test_time_selection": False,
        "backbone_trainable_params": 0,
        "trainable_component": "logistic_l2_or_linear_svm_head_only",
        "seeg_adaptation": "single-channel SEEG contact mode; no scalp montage remapping",
        "warning": "FM is used only as a frozen comparison baseline, not as the proposed method.",
    }


def _require_existing_path(path: str | None, label: str) -> Path:
    if not path:
        raise FileNotFoundError(f"{label} is required.")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_device(device: str) -> str:
    requested = str(device or "auto").lower()
    if requested == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _prepend_repo_path(repo: Path) -> bool:
    repo_str = str(repo)
    if repo_str in sys.path:
        return False
    sys.path.insert(0, repo_str)
    return True


def _clear_stale_modules(*module_roots: str) -> None:
    for root in module_roots:
        for name in list(sys.modules):
            if name == root or name.startswith(root + "."):
                sys.modules.pop(name, None)


def _load_checkpoint_state_dict(checkpoint_path: Path) -> Dict[str, Any]:
    import torch

    obj = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict):
                return dict(value)
        if all(hasattr(v, "shape") for v in obj.values()):
            return dict(obj)
    raise RuntimeError(f"checkpoint_path does not contain a loadable state dict: {checkpoint_path}")


def _strip_prefix_once(key: str, prefix: str) -> str:
    return key[len(prefix) :] if key.startswith(prefix) else key


def _normalize_state_dict(state: Dict[str, Any], *, prefer_prefix: str | None = None) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    items = list(state.items())
    if prefer_prefix:
        prefixed = [(k, v) for k, v in items if k.startswith(prefer_prefix)]
        if prefixed:
            items = prefixed
    for key, value in items:
        out_key = str(key)
        for prefix in ("module.", "model."):
            out_key = _strip_prefix_once(out_key, prefix)
        if prefer_prefix:
            out_key = _strip_prefix_once(out_key, prefer_prefix)
        for prefix in ("module.", "model."):
            out_key = _strip_prefix_once(out_key, prefix)
        normalized[out_key] = value
    return normalized


def _load_model_state(model: Any, state: Dict[str, Any], *, strict: bool, model_name: str) -> tuple[list[str], list[str]]:
    try:
        result = model.load_state_dict(state, strict=strict)
    except RuntimeError:
        if strict:
            raise
        result = model.load_state_dict(state, strict=False)
    missing = [str(item) for item in getattr(result, "missing_keys", [])]
    unexpected = [str(item) for item in getattr(result, "unexpected_keys", [])]
    if not state:
        raise RuntimeError(f"{model_name} checkpoint state dict is empty.")
    return missing, unexpected


def _freeze_eval(model: Any, device: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)


def _tensor_to_2d_float32(output: Any, *, model_name: str) -> np.ndarray:
    import torch

    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError(f"{model_name} returned an empty output tuple/list.")
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"{model_name} returned non-tensor output: {type(output)!r}")
    if output.ndim == 1:
        output = output.unsqueeze(0)
    if output.ndim > 2:
        output = output.reshape(output.shape[0], -1)
    return output.detach().cpu().numpy().astype(np.float32, copy=False)


class RandomProjectionExtractor(FrozenFMExtractor):
    name = "random_projection"

    def __init__(self, embedding_dim: int = 256, random_seed: int = 42, target_sfreq: int = 200, window_sec: float = 4.0):
        self.embedding_dim = int(embedding_dim)
        self.random_seed = int(random_seed)
        self.target_sfreq = int(target_sfreq)
        self.window_sec = float(window_sec)
        self._projection: np.ndarray | None = None
        self.audit_info = _comparison_audit_fields(
            "random_projection",
            true_pretrained_fm=False,
            debug_only=True,
            paper_baseline=False,
        )

    def load(self, checkpoint_path: str | None, external_repo_path: str | None, device: str):
        self.loaded = True
        self.audit_info.update({"adapter_status": "debug_random_projection", "checkpoint_loaded": False})
        return self

    def encode_batch(self, batch_waveforms: np.ndarray) -> np.ndarray:
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim == 3:
            x = x[:, 0, :]
        if x.ndim != 2:
            raise ValueError("RandomProjectionExtractor expects [B,T] or [B,1,T].")
        if self._projection is None or self._projection.shape[0] != x.shape[1]:
            rng = np.random.default_rng(self.random_seed + x.shape[1])
            self._projection = rng.normal(0.0, 1.0 / np.sqrt(max(x.shape[1], 1)), size=(x.shape[1], self.embedding_dim)).astype(np.float32)
        stats = np.stack([x.mean(axis=1), x.std(axis=1), x.max(axis=1), x.min(axis=1)], axis=1).astype(np.float32)
        projected = x @ self._projection
        projected[:, : min(4, self.embedding_dim)] += stats[:, : min(4, self.embedding_dim)]
        return projected.astype(np.float32)


class BIOTFrozenExtractor(FrozenFMExtractor):
    name = "biot"

    def __init__(
        self,
        target_sfreq: int = 200,
        window_sec: float = 4.0,
        n_fft: int = 200,
        hop_length: int = 100,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        biot_n_channels: str | int = "auto",
        n_channel_offset: int = 0,
        strict_load: bool = True,
    ):
        self.target_sfreq = int(target_sfreq)
        self.window_sec = float(window_sec)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.emb_size = int(emb_size)
        self.heads = int(heads)
        self.depth = int(depth)
        self.biot_n_channels = biot_n_channels
        self.n_channel_offset = int(n_channel_offset)
        self.strict_load = bool(strict_load)
        self.model = None
        self.device = "cpu"
        self.embedding_dim = self.emb_size
        self.audit_info = _comparison_audit_fields("biot", true_pretrained_fm=True, debug_only=False, paper_baseline=True)

    def load(self, checkpoint_path: str | None, external_repo_path: str | None, device: str):
        repo = _require_existing_path(external_repo_path, "external_repo_path")
        checkpoint = _require_existing_path(checkpoint_path, "checkpoint_path")
        _prepend_repo_path(repo)
        _clear_stale_modules("model")
        importlib.invalidate_caches()
        try:
            try:
                from model import BIOTEncoder  # type: ignore
            except ImportError:
                from model.biot import BIOTEncoder  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"BIOT adapter could not import BIOTEncoder from external_repo_path: {repo}") from exc

        raw_state = _load_checkpoint_state_dict(checkpoint)
        state = _normalize_state_dict(raw_state, prefer_prefix="biot.")
        audit_warning = ""
        inferred_n_channels = None
        for key, value in state.items():
            if key.endswith("channel_tokens.weight") and hasattr(value, "shape") and len(value.shape) >= 1:
                inferred_n_channels = int(value.shape[0])
                break
        if str(self.biot_n_channels).lower() == "auto":
            if inferred_n_channels is None:
                name = checkpoint.name.lower()
                if "16" in name:
                    inferred_n_channels = 16
                elif "18" in name:
                    inferred_n_channels = 18
                else:
                    inferred_n_channels = 18
                    audit_warning = "BIOT n_channels could not be inferred from checkpoint; defaulted to 18."
            n_channels = inferred_n_channels
        else:
            n_channels = int(self.biot_n_channels)

        try:
            self.model = BIOTEncoder(
                emb_size=self.emb_size,
                heads=self.heads,
                depth=self.depth,
                n_channels=int(n_channels),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        except TypeError as exc:
            raise RuntimeError("BIOTEncoder constructor did not accept the expected frozen-baseline arguments.") from exc
        missing, unexpected = _load_model_state(self.model, state, strict=self.strict_load, model_name="BIOT")
        for component in ("patch_embedding", "transformer", "channel_tokens"):
            if not any(str(key).startswith(component) for key in state):
                raise RuntimeError(f"BIOT checkpoint did not load required component: {component}")
        self.device = _resolve_device(device)
        _freeze_eval(self.model, self.device)
        self.loaded = True
        self.audit_info.update(
            {
                "adapter_status": "implemented",
                "biot_n_channels": int(n_channels),
                "biot_n_fft": self.n_fft,
                "biot_hop_length": self.hop_length,
                "biot_emb_size": self.emb_size,
                "biot_heads": self.heads,
                "biot_depth": self.depth,
                "biot_n_channel_offset": self.n_channel_offset,
                "biot_strict_load": self.strict_load,
                "biot_missing_keys": missing,
                "biot_unexpected_keys": unexpected,
                "biot_checkpoint_loaded": True,
                "seeg_input_mode": "single_channel_contact",
                "adapter_warning": audit_warning,
            }
        )
        return self

    def encode_batch(self, batch_waveforms: np.ndarray) -> np.ndarray:
        if self.model is None or not self.loaded:
            raise RuntimeError("BIOTFrozenExtractor is not loaded.")
        import torch

        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError("BIOTFrozenExtractor expects [B,T] or [B,1,T].")
        if x.shape[1] != 1:
            raise ValueError("BIOTFrozenExtractor only supports single-channel SEEG contact input for now.")
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            output = self.model(tensor, n_channel_offset=self.n_channel_offset)
        return _tensor_to_2d_float32(output, model_name="BIOT")


class CBraModFrozenExtractor(FrozenFMExtractor):
    name = "cbramod"

    def __init__(
        self,
        target_sfreq: int = 200,
        window_sec: float = 4.0,
        cbramod_points_per_patch: int = 200,
        cbramod_num_segments: str | int = "auto",
        strict_load: bool = True,
    ):
        self.target_sfreq = int(target_sfreq)
        self.window_sec = float(window_sec)
        self.points_per_patch = int(cbramod_points_per_patch)
        self.num_segments = cbramod_num_segments
        self.strict_load = bool(strict_load)
        self.model = None
        self.device = "cpu"
        self.audit_info = _comparison_audit_fields("cbramod", true_pretrained_fm=True, debug_only=False, paper_baseline=True)

    def load(self, checkpoint_path: str | None, external_repo_path: str | None, device: str):
        repo = _require_existing_path(external_repo_path, "external_repo_path")
        checkpoint = _require_existing_path(checkpoint_path, "checkpoint_path")
        _prepend_repo_path(repo)
        _clear_stale_modules("models")
        importlib.invalidate_caches()
        try:
            from models.cbramod import CBraMod  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"CBraMod adapter could not import CBraMod from external_repo_path: {repo}") from exc

        self.model = CBraMod()
        state = _normalize_state_dict(_load_checkpoint_state_dict(checkpoint))
        missing, unexpected = _load_model_state(self.model, state, strict=self.strict_load, model_name="CBraMod")
        proj_replaced = hasattr(self.model, "proj_out")
        if proj_replaced:
            import torch

            self.model.proj_out = torch.nn.Identity()
        self.device = _resolve_device(device)
        _freeze_eval(self.model, self.device)
        self.loaded = True
        self.audit_info.update(
            {
                "adapter_status": "implemented",
                "cbramod_points_per_patch": self.points_per_patch,
                "cbramod_num_segments": self.num_segments,
                "cbramod_checkpoint_loaded": True,
                "cbramod_strict_load": self.strict_load,
                "cbramod_missing_keys": missing,
                "cbramod_unexpected_keys": unexpected,
                "seeg_input_mode": "single_channel_contact",
                "proj_out_replaced_by_identity": bool(proj_replaced),
            }
        )
        return self

    def encode_batch(self, batch_waveforms: np.ndarray) -> np.ndarray:
        if self.model is None or not self.loaded:
            raise RuntimeError("CBraModFrozenExtractor is not loaded.")
        import torch

        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError("CBraModFrozenExtractor only supports single-channel SEEG contact input for now.")
            x = x[:, 0, :]
        if x.ndim != 2:
            raise ValueError("CBraModFrozenExtractor expects [B,T] or [B,1,T].")
        if str(self.num_segments).lower() == "auto":
            segments = int(np.ceil(x.shape[1] / float(self.points_per_patch)))
        else:
            segments = int(self.num_segments)
        target_len = int(segments * self.points_per_patch)
        if x.shape[1] < target_len:
            x = np.pad(x, ((0, 0), (0, target_len - x.shape[1])), mode="constant")
        elif x.shape[1] > target_len:
            x = x[:, :target_len]
        x = x.reshape(x.shape[0], 1, segments, self.points_per_patch)
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            output = self.model(tensor)
        return _tensor_to_2d_float32(output, model_name="CBraMod")


class LaBraMFrozenExtractor(FrozenFMExtractor):
    name = "labram"

    def __init__(
        self,
        target_sfreq: int = 200,
        window_sec: float = 4.0,
        labram_model_name: str = "labram_base_patch200_200",
        labram_patch_size: int = 200,
        labram_num_channels: int = 1,
        labram_channel_order: str = "single_seeg",
        strict_load: bool = True,
    ):
        self.target_sfreq = int(target_sfreq)
        self.window_sec = float(window_sec)
        self.model_name = str(labram_model_name)
        self.patch_size = int(labram_patch_size)
        self.num_channels = int(labram_num_channels)
        self.channel_order = str(labram_channel_order)
        self.strict_load = bool(strict_load)
        self.model = None
        self.device = "cpu"
        self.audit_info = _comparison_audit_fields("labram", true_pretrained_fm=True, debug_only=False, paper_baseline=True)

    def _locate_factory(self, repo: Path):
        _prepend_repo_path(repo)
        _clear_stale_modules("models", "modeling_pretrain", "modeling_finetune")
        importlib.invalidate_caches()
        for module_name in ("models", "modeling_pretrain", "modeling_finetune"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            factory = getattr(module, self.model_name, None)
            if callable(factory):
                return factory, module_name
        try:
            import timm  # type: ignore
        except Exception:
            timm = None
        if timm is not None:
            return lambda **kwargs: timm.create_model(self.model_name, pretrained=False, **kwargs), "timm"
        raise RuntimeError("LaBraM adapter could not locate a supported model factory in external_repo_path.")

    def load(self, checkpoint_path: str | None, external_repo_path: str | None, device: str):
        repo = _require_existing_path(external_repo_path, "external_repo_path")
        checkpoint = _require_existing_path(checkpoint_path, "checkpoint_path")
        factory, factory_module = self._locate_factory(repo)
        try:
            self.model = factory(num_channels=self.num_channels, patch_size=self.patch_size)
        except TypeError:
            self.model = factory()
        state = _normalize_state_dict(_load_checkpoint_state_dict(checkpoint))
        missing, unexpected = _load_model_state(self.model, state, strict=self.strict_load, model_name="LaBraM")
        if not self.strict_load:
            total_state_keys = max(1, len(state))
            mismatch_ratio = (len(missing) + len(unexpected)) / float(total_state_keys)
            if mismatch_ratio > 0.5:
                raise RuntimeError("LaBraM checkpoint has too many missing/unexpected keys under strict=False.")
        head_replaced = False
        import torch

        for attr in ("head", "fc", "classifier"):
            if hasattr(self.model, attr) and isinstance(getattr(self.model, attr), torch.nn.Module):
                setattr(self.model, attr, torch.nn.Identity())
                head_replaced = True
                break
        self.device = _resolve_device(device)
        _freeze_eval(self.model, self.device)
        self.loaded = True
        try:
            test_output = self.encode_batch(np.ones((2, int(round(self.target_sfreq * self.window_sec))), dtype=np.float32))
            if test_output.ndim != 2 or test_output.shape[0] != 2 or test_output.shape[1] <= 0:
                raise RuntimeError(f"invalid LaBraM forward output shape: {test_output.shape}")
        except Exception as exc:
            self.loaded = False
            raise RuntimeError(
                "LaBraM adapter cannot run without unsupported scalp montage/channel metadata; "
                "refusing to fabricate scalp montage."
            ) from exc
        self.audit_info.update(
            {
                "adapter_status": "implemented",
                "labram_factory_module": factory_module,
                "labram_factory_name": self.model_name,
                "labram_model_name": self.model_name,
                "labram_patch_size": self.patch_size,
                "labram_num_channels": self.num_channels,
                "labram_channel_order": self.channel_order,
                "labram_checkpoint_loaded": True,
                "labram_strict_load": self.strict_load,
                "labram_missing_keys": missing,
                "labram_unexpected_keys": unexpected,
                "labram_forward_test_passed": True,
                "labram_safe_for_paper_baseline": True,
                "seeg_input_mode": "single_channel_contact",
                "classification_head_replaced_by_identity": bool(head_replaced),
            }
        )
        return self

    def encode_batch(self, batch_waveforms: np.ndarray) -> np.ndarray:
        if self.model is None or not self.loaded:
            raise RuntimeError("LaBraMFrozenExtractor is not loaded.")
        import torch

        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError("LaBraMFrozenExtractor expects [B,T] or [B,1,T].")
        if x.shape[1] != 1:
            raise ValueError("LaBraMFrozenExtractor only supports single-channel SEEG contact input for now.")
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            try:
                output = self.model(tensor, channel_order=self.channel_order)
            except TypeError:
                output = self.model(tensor)
        return _tensor_to_2d_float32(output, model_name="LaBraM")


def build_extractor(fm_model: str, *, target_sfreq: int, window_sec: float, **kwargs: Any) -> FrozenFMExtractor:
    name = str(fm_model).lower()
    if name == "random_projection":
        return RandomProjectionExtractor(target_sfreq=target_sfreq, window_sec=window_sec)
    if name == "biot":
        return BIOTFrozenExtractor(
            target_sfreq=target_sfreq,
            window_sec=window_sec,
            n_fft=kwargs.get("biot_n_fft", 200),
            hop_length=kwargs.get("biot_hop_length", 100),
            emb_size=kwargs.get("biot_emb_size", 256),
            heads=kwargs.get("biot_heads", 8),
            depth=kwargs.get("biot_depth", 4),
            biot_n_channels=kwargs.get("biot_n_channels", "auto"),
            n_channel_offset=kwargs.get("biot_n_channel_offset", 0),
            strict_load=kwargs.get("biot_strict_load", True),
        )
    if name == "cbramod":
        return CBraModFrozenExtractor(
            target_sfreq=target_sfreq,
            window_sec=window_sec,
            cbramod_points_per_patch=kwargs.get("cbramod_points_per_patch", 200),
            cbramod_num_segments=kwargs.get("cbramod_num_segments", "auto"),
            strict_load=kwargs.get("cbramod_strict_load", True),
        )
    if name == "labram":
        return LaBraMFrozenExtractor(
            target_sfreq=target_sfreq,
            window_sec=window_sec,
            labram_model_name=kwargs.get("labram_model_name", "labram_base_patch200_200"),
            labram_patch_size=kwargs.get("labram_patch_size", 200),
            labram_num_channels=kwargs.get("labram_num_channels", 1),
            labram_channel_order=kwargs.get("labram_channel_order", "single_seeg"),
            strict_load=kwargs.get("labram_strict_load", True),
        )
    raise ValueError(f"Unknown fm_model: {fm_model}")
