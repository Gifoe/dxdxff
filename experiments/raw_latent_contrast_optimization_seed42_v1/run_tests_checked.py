"""Outcome-free unit-test entry: stable CPU affinity and NumPy-before-Torch."""
from pathlib import Path
import psutil
psutil.Process().cpu_affinity([16])
import numpy
import pytest

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    raise SystemExit(pytest.main([str(here/'test_model.py'),str(here/'test_engine.py'),'-q']))
