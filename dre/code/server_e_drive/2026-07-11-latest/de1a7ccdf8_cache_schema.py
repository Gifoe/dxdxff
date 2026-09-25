from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CacheContractError(ValueError):
    """Raised when a cache does not satisfy the top-level run/patient contract."""


@dataclass(frozen=True)
class CachePayload:
    source_path: Path | None
    payload: Mapping[str, Any]
    run_records: tuple[Mapping[str, Any], ...]
    patient_index: Mapping[str, Mapping[str, Any]]

    @property
    def top_level_keys(self) -> tuple[str, ...]:
        return tuple(sorted(str(key) for key in self.payload))


def _validate_payload(payload: Any, source_path: Path | None = None) -> CachePayload:
    if not isinstance(payload, Mapping):
        raise CacheContractError("Cache top level must be a mapping.")
    if "run_records" not in payload:
        raise CacheContractError("Cache is missing required top-level key: run_records")
    if "patient_index" not in payload:
        raise CacheContractError("Cache is missing required top-level key: patient_index")
    run_records = payload["run_records"]
    patient_index = payload["patient_index"]
    if not isinstance(run_records, (list, tuple)):
        raise CacheContractError("Cache run_records must be a list or tuple.")
    if not isinstance(patient_index, Mapping):
        raise CacheContractError("Cache patient_index must be a mapping.")
    bad_records = [index for index, record in enumerate(run_records) if not isinstance(record, Mapping)]
    if bad_records:
        raise CacheContractError(f"Cache run_records contain non-mapping entries at indices: {bad_records[:10]}")
    bad_patients = [subject for subject, meta in patient_index.items() if not isinstance(meta, Mapping)]
    if bad_patients:
        raise CacheContractError(f"Cache patient_index contains non-mapping metadata for: {bad_patients[:10]}")
    return CachePayload(
        source_path=source_path,
        payload=payload,
        run_records=tuple(run_records),
        patient_index={str(subject): meta for subject, meta in patient_index.items()},
    )


def load_cache_contract(source: str | Path | Mapping[str, Any]) -> CachePayload:
    if isinstance(source, Mapping):
        return _validate_payload(source)
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Cache does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return _validate_payload(payload, source_path=path.resolve())


def file_sha256(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


__all__ = ["CacheContractError", "CachePayload", "file_sha256", "load_cache_contract"]
