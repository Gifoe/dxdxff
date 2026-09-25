"""Run a selected A12 variant through the full suite entry point."""

from __future__ import annotations

import sys

from run_a12_vcsn_suite import main


if __name__ == "__main__":
    if "--variants" not in sys.argv:
        sys.argv.extend(["--variants", "A12-V1"])
    main()
