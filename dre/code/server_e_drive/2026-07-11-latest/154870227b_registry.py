from __future__ import annotations

from typing import Any

from .biot_adapter import BIOTAdapter
from .brainbert_adapter import BrainBERTAdapter
from .cbramod_adapter import CBraModAdapter, LaBraMAdapter


def build_fm_adapter(name: str, config: dict[str, Any]):
    normalized = str(name).strip().lower()
    registry = {
        "biot": BIOTAdapter,
        "cbramod": CBraModAdapter,
        "labram": LaBraMAdapter,
        "brainbert": BrainBERTAdapter,
    }
    if normalized not in registry:
        raise ValueError(f"Unsupported frozen FM {name!r}; expected one of {tuple(registry)}.")
    return registry[normalized](dict(config))


__all__ = ["build_fm_adapter"]
