"""Stable Windows unit-test entry, before any GPU training."""
from pathlib import Path
import psutil
psutil.Process().cpu_affinity([16])
import numpy
import pytest

if __name__=='__main__':raise SystemExit(pytest.main([str(Path(__file__).parent/'test_model.py'),'-q']))
