"""Checkpoint compatibility checks for the final Q10-free BCR-Net."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


FORBIDDEN_BCR_Q10_TOKENS = ("q10", "quantile", "lower_tail")


def inspect_bcr_state_dict(state_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Return incompatible legacy-Q10 keys without silently dropping them."""
    legacy = [str(key) for key in state_dict if any(token in str(key).lower() for token in FORBIDDEN_BCR_Q10_TOKENS)]
    return {"legacy_q10_keys": sorted(legacy)}


def load_bcr_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    allow_non_strict_debug: bool = False,
) -> dict[str, list[str]]:
    """Load a BCR checkpoint, rejecting historical BCR-Q10 weights by default."""
    payload = torch.load(Path(path), map_location="cpu")
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError("BCR checkpoint must contain a model state dictionary")
    report = inspect_bcr_state_dict(state_dict)
    if report["legacy_q10_keys"] and not allow_non_strict_debug:
        raise RuntimeError(
            "Incompatible legacy BCR-Q10 checkpoint. Retrain BCR-Net without Q10; "
            f"legacy keys={report['legacy_q10_keys']}"
        )
    incompatible = model.load_state_dict(state_dict, strict=not allow_non_strict_debug)
    report["missing_keys"] = list(incompatible.missing_keys)
    report["unexpected_keys"] = list(incompatible.unexpected_keys)
    if allow_non_strict_debug and (report["legacy_q10_keys"] or report["missing_keys"] or report["unexpected_keys"]):
        report["status"] = "DEBUG_NON_STRICT_INCOMPATIBLE"
    else:
        report["status"] = "COMPATIBLE"
    return report


def inspect_bcr_checkpoint_path(path: str | Path) -> dict[str, list[str]]:
    """Inspect a checkpoint before model construction in runners and audits."""
    payload = torch.load(Path(path), map_location="cpu")
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError("BCR checkpoint must contain a model state dictionary")
    return inspect_bcr_state_dict(state_dict)


__all__ = [
    "FORBIDDEN_BCR_Q10_TOKENS", "inspect_bcr_state_dict", "inspect_bcr_checkpoint_path",
    "load_bcr_checkpoint",
]
