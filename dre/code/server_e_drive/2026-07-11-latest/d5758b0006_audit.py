from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def file_identity(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser().resolve(); stat = value.stat()
    return {"absolute_path": str(value), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def cache_signature(
    input_path: str | Path, *, patient: str, seizure: str, sampling_rate: float,
    algorithm_version: str, parameters: dict[str, Any], negative_control: str,
) -> tuple[str, dict[str, Any]]:
    payload = {"input": file_identity(input_path), "patient": str(patient), "seizure": str(seizure), "sampling_rate": float(sampling_rate), "algorithm_version": str(algorithm_version), "parameters": parameters, "negative_control": str(negative_control)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


class BiomarkerCache:
    def __init__(self, root: str | Path, resume: bool = True) -> None:
        self.root = Path(root).expanduser(); self.root.mkdir(parents=True, exist_ok=True); self.resume = bool(resume)

    def path(self, family: str, patient: str, seizure: str) -> Path:
        token = hashlib.sha256(f"{patient}|{seizure}".encode()).hexdigest()[:20]
        path = self.root / family / f"{token}.pkl"; path.parent.mkdir(parents=True, exist_ok=True); return path

    def load(self, family: str, patient: str, seizure: str, signature: str, *, recompute: bool = False) -> Any | None:
        path = self.path(family, patient, seizure)
        if recompute or not self.resume or not path.exists(): return None
        try:
            with path.open("rb") as handle: value = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError): return None
        return value.get("value") if isinstance(value, dict) and value.get("signature") == signature else None

    def save(self, family: str, patient: str, seizure: str, signature: str, signature_payload: dict, value: Any) -> None:
        path = self.path(family, patient, seizure); temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle: pickle.dump({"signature": signature, "signature_payload": signature_payload, "value": value}, handle, pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return (int(seed) + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


__all__ = ["BiomarkerCache", "cache_signature", "file_identity", "stable_seed"]
