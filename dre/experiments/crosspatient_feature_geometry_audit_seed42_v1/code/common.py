"""Frozen development-only source access for the cross-patient geometry audit."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(PROJECT / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from run_development import assert_architecture_args, assert_cohort  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402

LOCK_SHA256 = "ca845ede5c5ae3504b06a95136a72fca6eb330b673f5b4b54159a97bc97ac462"
INVENTORY_SHA256 = "6db5ca02770ab40b07fcf55c07a0ba8d80e39502f0d18d370362a782cc1b2b22"
SOURCE_LOCK_SHA256 = "6694da1b351d7130015661fcce1ec91bec643821392e409f1d5728f472b3aba0"
A1_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))
RUNTIME = Path(os.environ.get("CROSS_RUNTIME", ""))


def ensure_source() -> None:
    if not os.environ.get("A1_A2_RUNTIME") or not A1_RUNTIME.is_absolute():
        raise RuntimeError("A1_A2_RUNTIME must be an absolute private path")
    if not os.environ.get("CROSS_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("CROSS_RUNTIME must be an absolute private path")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != LOCK_SHA256:
        raise RuntimeError("Frozen protocol lock changed")
    if sha256(PROJECT / "a1_a2_patient_equal_objective_seed42_v1" / "PROTOCOL_LOCK.json") != SOURCE_LOCK_SHA256:
        raise RuntimeError("A1 source protocol changed")
    inv = json.loads((EXPERIMENT / "FEATURE_INVENTORY.json").read_text(encoding="utf-8"))
    payload = "\n".join(inv["cache_feature_names_in_order"]).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != INVENTORY_SHA256:
        raise RuntimeError("Frozen feature inventory changed")
    assert_sources()
    if not json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("A1 reproduction not passed")


def make_experiment():
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    assert_architecture_args(args)
    exp = core.Exp_EZHybridLocalization(args)
    assert_cohort(exp)
    return exp


def source_checkpoint(fold: int, epoch: int) -> Path:
    return A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"


def selected_epochs(fold: int) -> dict[str, int]:
    import csv
    source = RUNTIME / "source_selected" / f"fold_{fold}_A1_SF1_PATIENT.csv"
    if not source.is_file():
        source = Path(os.environ.get("RANK_RUNTIME", "")) / "private" / f"fold_{fold}_A1_SF1_PATIENT.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 13:
        raise RuntimeError("A1 selected validation subjects changed")
    return {row["subject_id"]: int(row["selected_epoch"]) for row in rows}


def assert_development_split(test_loader) -> None:
    if test_loader is not None:
        raise RuntimeError("Outer-test loader constructed")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
