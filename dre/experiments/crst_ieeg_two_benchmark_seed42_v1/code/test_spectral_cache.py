"""Synthetic physical-frequency, edge symmetry and padded-window tests."""

import json
from pathlib import Path

import numpy as np

from spectral_cache import FREQUENCIES_HZ, connectivity_edges, spectral_patches


def run():
    results = {}
    for fs in (200, 250, 300, 1000):
        t = np.arange(60 * fs, dtype=np.float64) / fs
        raw = np.stack((np.sin(2 * np.pi * 10 * t),
                        np.sin(2 * np.pi * 20 * t + 0.3))).astype(np.float32)
        patches, fmask, wmask = spectral_patches(raw, fs)
        assert patches.shape == (2, 59, 64, 8)
        assert fmask.shape == (64,) and wmask.shape == (2, 59)
        assert np.isfinite(patches).all() and wmask.all()
        assert np.all(patches[:, :, FREQUENCIES_HZ > 0.45 * fs] == 0)
        edges, bands = connectivity_edges(raw, fs)
        assert edges.shape == (2, 2, 15) and np.isfinite(edges).all()
        assert np.max(np.abs(edges - edges.transpose(1, 0, 2))) < 1e-5
        assert np.all(edges[:, :, np.repeat(bands == 0, 3)] == 0)
        results[str(fs)] = {"valid_frequency_bins": int(fmask.sum()),
                            "valid_connectivity_bands": int(bands.sum())}
    short, _, mask = spectral_patches(raw, 1000, valid_samples=41_000)
    assert mask[0].sum() == 40 and np.all(short[:, 40:] == 0)
    _, _, mask_offset = spectral_patches(raw, 1000, valid_samples=41_000,
                                          valid_start=19_000)
    assert mask_offset[0].sum() == 40 and not mask_offset[0, 0]
    return {"pass": True, "sampling_rates": results,
            "partial_41_second_valid_windows": 40}


if __name__ == "__main__":
    result = run()
    path = Path(__file__).resolve().parents[1] / "TOKENIZER_SYNTHETIC_TEST.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
