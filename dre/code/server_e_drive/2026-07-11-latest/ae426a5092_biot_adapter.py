from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .base import AdapterSpec, FrozenFMAdapter


class RepositoryExtractorAdapter(FrozenFMAdapter):
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        target_sfreq = float(config.get("target_sfreq", 200))
        window_sec = float(config.get("window_sec", 4.0))
        output_dim = int(config.get("embedding_dim", 256)) if name == "biot" else None
        super().__init__(
            AdapterSpec(
                name=name,
                expected_sampling_rate=target_sfreq,
                input_duration_sec=window_sec,
                channel_handling="single_seeg_contact",
                normalization="robust_median_iqr",
                embedding_layer="pre_classification_representation",
                output_dim=output_dim,
                version=str(config.get("version", "repository-adapter-v1")),
            )
        )
        self.config = dict(config)
        self.extractor: Any | None = None

    def load(self, *, checkpoint_path: str | Path, external_repo_path: str | Path | None, device: str) -> "RepositoryExtractorAdapter":
        from scripts.fm_baselines.extractors import build_extractor

        self.extractor = build_extractor(
            self.spec.name,
            target_sfreq=int(round(self.spec.expected_sampling_rate)),
            window_sec=self.spec.input_duration_sec,
            **self.config,
        )
        self.extractor.load(str(checkpoint_path), None if external_repo_path is None else str(external_repo_path), str(device))
        self.model = getattr(self.extractor, "model", None)
        if self.model is None:
            raise RuntimeError(f"{self.spec.name} repository extractor did not expose its frozen model.")
        self.freeze()
        self.audit_info.update(dict(getattr(self.extractor, "audit_info", {})))
        checkpoint = Path(checkpoint_path).resolve()
        repository = Path(external_repo_path).resolve() if external_repo_path is not None else None
        self.audit_info["checkpoint_path"] = str(checkpoint)
        self.audit_info["checkpoint_sha256"] = self.checkpoint_sha256(checkpoint)
        self.audit_info["checkpoint_size"] = checkpoint.stat().st_size
        if self.spec.name == "cbramod":
            if repository is None:
                raise ValueError("CBraMod requires the checked-out official repository path.")
            try:
                commit = subprocess.run(
                    ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
                ).stdout.strip()
                remote = subprocess.run(
                    ["git", "-C", str(repository), "remote", "get-url", "origin"], check=True, capture_output=True, text=True
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ValueError("CBraMod external repository must be a Git checkout with an origin remote.") from exc
            normalized_remote = remote.removesuffix(".git").rstrip("/")
            official = "https://github.com/wjq-learning/CBraMod"
            if normalized_remote != official:
                raise ValueError(f"CBraMod repository origin must be {official!r}, found {remote!r}.")
            checkpoint_source = str(self.config.get("checkpoint_source", "")).strip()
            license_name = str(self.config.get("license", "")).strip()
            if not checkpoint_source or not license_name:
                raise ValueError("CBraMod config must record checkpoint_source and license.")
            self.audit_info.update(
                {
                    "official_repository": official,
                    "repository_commit_sha": commit,
                    "checkpoint_source": checkpoint_source,
                    "license": license_name,
                    "missing_keys": list(self.audit_info.get("cbramod_missing_keys", [])),
                    "unexpected_keys": list(self.audit_info.get("cbramod_unexpected_keys", [])),
                }
            )
        return self

    def encode_batch(self, waveforms: np.ndarray) -> np.ndarray:
        if self.extractor is None or not self.loaded:
            raise RuntimeError(f"{self.spec.name} adapter must be loaded before encoding.")
        output = np.asarray(self.extractor.encode_batch(waveforms), dtype=np.float32)
        if output.ndim != 2 or output.shape[0] != np.asarray(waveforms).shape[0]:
            raise RuntimeError(f"{self.spec.name} returned invalid embedding shape {output.shape}.")
        return output


class BIOTAdapter(RepositoryExtractorAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("biot", config)


__all__ = ["BIOTAdapter", "RepositoryExtractorAdapter"]
