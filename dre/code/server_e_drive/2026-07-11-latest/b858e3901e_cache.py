from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np

CACHE_VERSION = "mosaic_atomic_cache_v1"
CACHE_FAMILIES = ("pilot_manifest", "p2_oof", "fragility_trajectory", "pte", "cii",
                  "recruitment", "spectral", "virtual_intervention", "patient_features")


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    if not path: return None
    value = Path(path); stat = value.stat()
    return {"path": str(value.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def channel_hash(names, mask=None) -> str:
    payload = {"names": list(map(str, names)), "mask": None if mask is None else np.asarray(mask, dtype=np.uint8).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def trajectory_signature(record, mechanism: str, algorithm_version: str, params: dict,
                         raw_identity=None, feature_identity=None) -> dict[str, Any]:
    payload = {"cache_version": CACHE_VERSION, "mechanism": mechanism,
               "patient_key": record.patient_key, "seizure_id": record.seizure_id,
               "sampling_rate": float(record.sampling_rate), "onset_sample": int(record.onset_sample),
               "n_samples": int(record.signal.shape[1]),
               "channel_hash": channel_hash(record.channel_names, record.valid_channel_mask),
               "true_ez_hash": channel_hash(record.channel_names, record.true_ez_mask),
               "algorithm_version": algorithm_version, "params": params,
               "raw_identity": raw_identity, "feature_identity": feature_identity}
    payload["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return payload


class AtomicCache:
    def __init__(self, root: str | Path, resume: bool = True, external_root: str | Path | None = None):
        self.root, self.resume = Path(root), bool(resume)
        self.external_root = Path(external_root) if external_root else None
        self._external_index: dict[str, dict[str, Path]] = {}
        for name in CACHE_FAMILIES: (self.root / name).mkdir(parents=True, exist_ok=True)

    def path(self, family: str, patient: str, seizure: str, suffix: str = ".pkl") -> Path:
        safe = lambda x: hashlib.sha1(str(x).encode()).hexdigest()[:12]
        return self.root / family / f"{safe(patient)}__{safe(seizure)}{suffix}"

    def load(self, family: str, patient: str, seizure: str, signature: dict, recompute: bool = False):
        if recompute or not self.resume: return None
        candidates = [self.path(family, patient, seizure)]
        if self.external_root:
            candidates += [self.external_root / family / candidates[0].name,
                           self.external_root / candidates[0].name]
            if family == "fragility_trajectory":
                candidates.append(self.external_root / "fragility" / candidates[0].name)
        for path in candidates:
            if not path.exists(): continue
            with path.open("rb") as handle: payload = pickle.load(handle)
            found = payload.get("signature", {}) if isinstance(payload, dict) else {}
            if found.get("digest") == signature.get("digest"): return payload.get("value")
        # A prior compatible implementation may use another deterministic file
        # name. Index only self-describing caches; legacy summary caches without
        # the full MOSAIC signature are deliberately rejected.
        if self.external_root:
            if family not in self._external_index:
                index = {}
                roots = [self.external_root / family, self.external_root]
                if family == "fragility_trajectory": roots.append(self.external_root / "fragility")
                for root in roots:
                    if not root.is_dir(): continue
                    for path in root.glob("*.pkl"):
                        try:
                            with path.open("rb") as handle: payload = pickle.load(handle)
                            found = payload.get("signature", {}) if isinstance(payload, dict) else {}
                            if isinstance(found, dict) and found.get("digest"): index[found["digest"]] = path
                        except (OSError, pickle.UnpicklingError, EOFError): continue
                self._external_index[family] = index
            path = self._external_index[family].get(signature.get("digest"))
            if path:
                with path.open("rb") as handle: return pickle.load(handle).get("value")
        return None

    def save(self, family: str, patient: str, seizure: str, signature: dict, value: Any) -> Path:
        path = self.path(family, patient, seizure); tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with tmp.open("wb") as handle: pickle.dump({"signature": signature, "value": value}, handle, pickle.HIGHEST_PROTOCOL)
        tmp.replace(path); return path


__all__ = ["CACHE_VERSION", "CACHE_FAMILIES", "AtomicCache", "file_identity", "channel_hash", "trajectory_signature"]
