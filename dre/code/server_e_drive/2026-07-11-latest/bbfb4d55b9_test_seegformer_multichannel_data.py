from __future__ import annotations

import numpy as np

from task1_baselines.seegformer.multichannel_data import Task1MultichannelWindow, collate_multichannel_windows


def test_multichannel_collate_masks_padding() -> None:
    batch = [
        Task1MultichannelWindow(np.ones((7, 800), dtype=np.float32), np.ones(7, dtype=np.float32), tuple(f"a{n}" for n in range(7)), "p1", "c", "s1", 0, "unknown", "labels"),
        Task1MultichannelWindow(np.ones((3, 800), dtype=np.float32), np.zeros(3, dtype=np.float32), tuple(f"b{n}" for n in range(3)), "p2", "c", "s1", 0, "unknown", "labels"),
    ]
    output = collate_multichannel_windows(batch)
    assert output["waveform"].shape == (2, 7, 800)
    assert output["channel_mask"].tolist() == [[True] * 7, [True] * 3 + [False] * 4]
