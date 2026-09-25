#!/usr/bin/env python3
from __future__ import annotations
import sys
from run_all_confirmatory import main
if __name__ == "__main__":
    sys.argv.insert(1, "--run_loco"); main()
