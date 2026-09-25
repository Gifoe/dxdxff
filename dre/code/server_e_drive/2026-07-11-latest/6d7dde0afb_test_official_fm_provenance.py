from __future__ import annotations

import json
from pathlib import Path

import pytest

from outcome_hifos.fm.embedding_cache import validate_official_fm_embedding_cache


def _write_audit(root: Path, **updates: object) -> None:
    payload = {
        "name": "cbramod",
        "official_repository": "https://github.com/wjq-learning/CBraMod",
        "repository_commit_sha": "a" * 40,
        "checkpoint_source": "https://huggingface.co/wjq-learning/CBraMod",
        "checkpoint_path": "/models/cbramod.pth",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_size": 123,
        "pretrained_modality": "EEG",
        "expected_sampling_rate": 200,
        "input_duration": 4.0,
        "input_representation": "single-channel waveform",
        "embedding_layer": "pre_classification_representation",
        "embedding_dimension": 200,
        "license": "official repository license",
        "missing_keys": [],
        "unexpected_keys": [],
        "frozen": True,
        "fine_tuned": False,
    }
    payload.update(updates)
    root.mkdir(exist_ok=True)
    (root / "audit.json").write_text(json.dumps(payload), encoding="utf-8")


def test_official_fm_cache_validation_accepts_complete_cbramod_audit(tmp_path: Path) -> None:
    _write_audit(tmp_path)
    audit = validate_official_fm_embedding_cache(tmp_path, "cbramod")
    assert audit["checkpoint_sha256"] == "b" * 64


def test_official_fm_cache_validation_rejects_wrong_repository_and_key_mismatch(tmp_path: Path) -> None:
    _write_audit(tmp_path, official_repository="https://example.com/reupload", missing_keys=["encoder.weight"])
    with pytest.raises(ValueError, match="official_repository"):
        validate_official_fm_embedding_cache(tmp_path, "cbramod")


def test_official_fm_cache_validation_rejects_internal_brainbert_checkpoint(tmp_path: Path) -> None:
    _write_audit(
        tmp_path,
        name="brainbert",
        official_repository="https://github.com/czlwang/BrainBERT",
        version="rawbrainbert-v1",
    )
    with pytest.raises(ValueError, match="internal RawBrainBERT"):
        validate_official_fm_embedding_cache(tmp_path, "brainbert")
