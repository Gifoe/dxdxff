"""CLI entrypoint for NeuroEZ-C traditional baselines from engineered features.

This script uses cached window_features only. Foundation-model baselines are
separate because they consume raw_waveform.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.traditional_baselines.core import main


if __name__ == "__main__":
    raise SystemExit(main())
