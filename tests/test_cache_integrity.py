import json
from pathlib import Path

import pytest

from reva_dlm.cache_verifier import verify_cache


def test_ready_cache_integrity_if_present():
    cache = Path(__file__).resolve().parents[1] / "cache" / "reva_gsm8k_mini_v1"
    if not (cache / "status.json").is_file():
        pytest.skip("READY cache is built only after G0/G1 pass")
    status = json.loads((cache / "status.json").read_text(encoding="utf-8"))
    if status.get("cache_status") != "READY_FOR_G2":
        # A stale cache may fail an even earlier immutable-version check; the
        # contract is simply that REBUILD_REQUIRED can never verify.
        with pytest.raises(AssertionError):
            verify_cache(cache, verify_source_hashes=False)
    else:
        assert (
            verify_cache(cache, verify_source_hashes=False)["cache_version"]
            == "reva_gsm8k_mini_v1"
        )
