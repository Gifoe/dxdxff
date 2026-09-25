from __future__ import annotations

import subprocess
import sys


def test_direct_baseline_scripts_can_import_repository_packages() -> None:
    for script in (
        "scripts/task1_baselines/run_task1_baseline_suite.py",
        "scripts/task2_baselines/run_task2_baseline_suite.py",
    ):
        completed = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
