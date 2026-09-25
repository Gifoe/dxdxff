from __future__ import annotations

from typing import Any, Dict

import torch

try:
    from TeChEZ import Model as TeChEZModel
except Exception:  # pragma: no cover - TeChEZ may live outside this module bundle.
    TeChEZModel = None

try:
    from neuroez_hybrid_model import NeuroEZHybridModel
except Exception:  # pragma: no cover
    NeuroEZHybridModel = None


class Exp_Basic:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.device = self._acquire_device()
        self.model_dict: Dict[str, Any] = {}
        if TeChEZModel is not None:
            self.model_dict.update(
                {
                    "TeChEZ": TeChEZModel,
                    "TeChEZFeature": TeChEZModel,
                    "TeChEZHybrid": TeChEZModel,
                    "PatientChannelRanker": TeChEZModel,
                }
            )
        if NeuroEZHybridModel is not None:
            self.model_dict.update(
                {
                    "NeuroEZHybrid": NeuroEZHybridModel,
                    "BNPDGS": NeuroEZHybridModel,
                    "CNNGNNMIL": NeuroEZHybridModel,
                }
            )

    def _acquire_device(self) -> torch.device:
        preferred = str(getattr(self.args, "device", "auto")).lower()
        if preferred == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if preferred.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(preferred)

    def _build_model(self):
        raise NotImplementedError


__all__ = ["Exp_Basic"]
