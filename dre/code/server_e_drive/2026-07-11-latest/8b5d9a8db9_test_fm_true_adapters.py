from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.build_fm_raw_window_manifest import build_raw_window_manifest
from scripts.extract_fm_embeddings_from_manifest import extract_fm_embeddings
from scripts.fm_baselines.extractors import build_extractor
from tests.test_fm_raw_waveform_prep import _raw_cache, _write_cache


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fake_biot(root: Path) -> tuple[Path, Path]:
    repo = root / "BIOT"
    _write(repo / "model" / "__init__.py", "from .biot import BIOTEncoder\n")
    _write(
        repo / "model" / "biot.py",
        """
import torch

class BIOTEncoder(torch.nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, n_fft=200, hop_length=100):
        super().__init__()
        self.channel_tokens = torch.nn.Embedding(n_channels, emb_size)
        self.patch_embedding = torch.nn.Linear(1, emb_size)
        self.transformer = torch.nn.Linear(emb_size, emb_size)

    def forward(self, x, n_channel_offset=0):
        pooled = x.mean(dim=-1).transpose(1, 0).transpose(1, 0).unsqueeze(-1)
        return self.transformer(self.patch_embedding(pooled).mean(dim=1) + self.channel_tokens.weight[n_channel_offset])
""",
    )
    import importlib
    import sys

    sys.path.insert(0, str(repo))
    try:
        mod = importlib.import_module("model")
        model = mod.BIOTEncoder(n_channels=18)
        ckpt = repo / "fake_biot.pt"
        torch.save({"state_dict": {"biot." + k: v for k, v in model.state_dict().items()}}, ckpt)
    finally:
        sys.path.remove(str(repo))
        for name in ["model.biot", "model"]:
            sys.modules.pop(name, None)
    return repo, ckpt


def _make_fake_cbramod(root: Path) -> tuple[Path, Path]:
    repo = root / "CBraMod"
    _write(repo / "models" / "__init__.py", "")
    _write(
        repo / "models" / "cbramod.py",
        """
import torch

class CBraMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.proj_out = torch.nn.Linear(1, 1)

    def forward(self, x):
        return x * self.scale
""",
    )
    import importlib
    import sys

    sys.path.insert(0, str(repo))
    try:
        for name in ["models.cbramod", "models"]:
            sys.modules.pop(name, None)
        mod = importlib.import_module("models.cbramod")
        model = mod.CBraMod()
        ckpt = repo / "fake_cbramod.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt)
    finally:
        sys.path.remove(str(repo))
        for name in ["models.cbramod", "models"]:
            sys.modules.pop(name, None)
    return repo, ckpt


def _make_fake_labram(root: Path) -> tuple[Path, Path]:
    repo = root / "LaBraM"
    _write(
        repo / "models" / "__init__.py",
        """
from .fake_labram import labram_base_patch200_200
""",
    )
    _write(
        repo / "models" / "fake_labram.py",
        """
import torch

class FakeLaBraM(torch.nn.Module):
    def __init__(self, num_channels=1, patch_size=200, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.head = torch.nn.Linear(1, 1)

    def forward(self, x, **kwargs):
        return x.mean(dim=-1) * self.weight

def labram_base_patch200_200(**kwargs):
    return FakeLaBraM(**kwargs)
""",
    )
    import importlib
    import sys

    sys.path.insert(0, str(repo))
    try:
        for name in ["models.fake_labram", "models.cbramod", "models"]:
            sys.modules.pop(name, None)
        mod = importlib.import_module("models")
        model = mod.labram_base_patch200_200()
        ckpt = repo / "fake_labram.pt"
        torch.save({"model": model.state_dict()}, ckpt)
    finally:
        sys.path.remove(str(repo))
        for name in ["models.fake_labram", "models"]:
            sys.modules.pop(name, None)
    return repo, ckpt


def _manifest_for_true_fm(root: Path) -> tuple[Path, Path]:
    cache_path = _write_cache(root, _raw_cache())
    build_raw_window_manifest(
        cache_path,
        root / "manifest",
        target_sfreq=200,
        window_sec=2.0,
        stride_sec=1.0,
        max_windows_per_record=1,
        positive_label="ez",
        split_strategy="5fold",
        n_splits=2,
        random_seed=42,
    )
    return cache_path, root / "manifest" / "fm_raw_window_manifest.csv"


class FMTrueAdapterTests(unittest.TestCase):
    def test_biot_frozen_adapter_loads_fake_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, ckpt = _make_fake_biot(Path(tmpdir))

            extractor = build_extractor("biot", target_sfreq=200, window_sec=2.0)
            extractor.load(str(ckpt), str(repo), "cpu")

            self.assertTrue(extractor.loaded)
            self.assertTrue(all(not p.requires_grad for p in extractor.model.parameters()))

    def test_biot_frozen_adapter_encodes_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, ckpt = _make_fake_biot(Path(tmpdir))
            extractor = build_extractor("biot", target_sfreq=200, window_sec=2.0)
            extractor.load(str(ckpt), str(repo), "cpu")

            out = extractor.encode_batch(np.ones((2, 400), dtype=np.float32))

            self.assertEqual(out.shape, (2, 256))

    def test_cbramod_frozen_adapter_loads_fake_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, ckpt = _make_fake_cbramod(Path(tmpdir))
            extractor = build_extractor("cbramod", target_sfreq=200, window_sec=2.0)
            extractor.load(str(ckpt), str(repo), "cpu")

            self.assertTrue(extractor.loaded)
            self.assertTrue(all(not p.requires_grad for p in extractor.model.parameters()))

    def test_cbramod_frozen_adapter_encodes_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, ckpt = _make_fake_cbramod(Path(tmpdir))
            extractor = build_extractor("cbramod", target_sfreq=200, window_sec=2.0)
            extractor.load(str(ckpt), str(repo), "cpu")

            out = extractor.encode_batch(np.ones((2, 400), dtype=np.float32))

            self.assertEqual(out.shape[0], 2)
            self.assertGreater(out.shape[1], 0)

    def test_labram_frozen_adapter_loads_fake_checkpoint_if_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, ckpt = _make_fake_labram(Path(tmpdir))
            extractor = build_extractor("labram", target_sfreq=200, window_sec=2.0)
            extractor.load(str(ckpt), str(repo), "cpu")

            self.assertTrue(extractor.loaded)
            self.assertTrue(all(not p.requires_grad for p in extractor.model.parameters()))

    def test_true_fm_extract_embeddings_writes_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo, ckpt = _make_fake_biot(root)
            cache_path, manifest_path = _manifest_for_true_fm(root)

            audit = extract_fm_embeddings(
                window_cache_path=cache_path,
                manifest_path=manifest_path,
                output_dir=root / "emb",
                fm_model="biot",
                external_repo_path=str(repo),
                checkpoint_path=str(ckpt),
                target_sfreq=200,
                window_sec=2.0,
                batch_size=4,
                device="cpu",
            )

            self.assertTrue((root / "emb" / "fm_window_embeddings.npy").exists())
            self.assertEqual(audit["fm_model"], "biot")
            self.assertTrue(audit["true_pretrained_fm"])
            self.assertTrue(audit["comparison_only_baseline"])

    def test_true_fm_audit_marks_comparison_only_and_frozen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo, ckpt = _make_fake_biot(root)
            cache_path, manifest_path = _manifest_for_true_fm(root)

            audit = extract_fm_embeddings(
                window_cache_path=cache_path,
                manifest_path=manifest_path,
                output_dir=root / "emb",
                fm_model="biot",
                external_repo_path=str(repo),
                checkpoint_path=str(ckpt),
                target_sfreq=200,
                window_sec=2.0,
                batch_size=4,
                device="cpu",
            )

            self.assertTrue(audit["comparison_only_baseline"])
            self.assertTrue(audit["frozen_encoder"])
            self.assertFalse(audit["fine_tuned"])
            self.assertFalse(audit["adapter_tuned"])
            self.assertEqual(audit["backbone_trainable_params"], 0)

    def test_true_fm_does_not_train_backbone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, ckpt = _make_fake_cbramod(Path(tmpdir))
            extractor = build_extractor("cbramod", target_sfreq=200, window_sec=2.0)
            extractor.load(str(ckpt), str(repo), "cpu")

            self.assertTrue(all(not p.requires_grad for p in extractor.model.parameters()))

    def test_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo, _ = _make_fake_biot(Path(tmpdir))

            with self.assertRaisesRegex(FileNotFoundError, "checkpoint_path"):
                build_extractor("biot", target_sfreq=200, window_sec=2.0).load(str(Path(tmpdir) / "missing.pt"), str(repo), "cpu")

    def test_missing_external_repo_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "external_repo_path"):
                build_extractor("cbramod", target_sfreq=200, window_sec=2.0).load(str(Path(tmpdir) / "x.pt"), str(Path(tmpdir) / "missing_repo"), "cpu")

    def test_no_random_fallback_for_true_fm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = _manifest_for_true_fm(root)

            with self.assertRaises(FileNotFoundError):
                extract_fm_embeddings(
                    window_cache_path=cache_path,
                    manifest_path=manifest_path,
                    output_dir=root / "emb",
                    fm_model="biot",
                    external_repo_path=str(root / "missing_repo"),
                    checkpoint_path=str(root / "missing.pt"),
                    target_sfreq=200,
                    window_sec=2.0,
                    batch_size=4,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
