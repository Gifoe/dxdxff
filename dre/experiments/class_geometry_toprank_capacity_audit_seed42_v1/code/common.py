"""Frozen input checks and aggregate-only helpers for the seed-42 ranking audit."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import METRICS, THRESHOLDS, epoch_grid, finalize_fold  # noqa: E402
from run_development import assert_architecture_args, assert_cohort, custom_compute_loss  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402

LOCK_SHA256 = "114c762d45adfb55c634d620339ec202b5aa5223317d3d9d2052b00a6b64b484"
SOURCE_LOCK_SHA256 = "6694da1b351d7130015661fcce1ec91bec643821392e409f1d5728f472b3aba0"
A1_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))
RUNTIME = Path(os.environ.get("GEOM_RUNTIME", ""))


def ensure_source() -> dict:
    if not os.environ.get("A1_A2_RUNTIME") or not A1_RUNTIME.is_absolute():
        raise RuntimeError("A1_A2_RUNTIME must name the private source runtime")
    if not os.environ.get("GEOM_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("GEOM_RUNTIME must name an absolute private output runtime")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != LOCK_SHA256:
        raise RuntimeError("Class geometry audit protocol lock changed")
    source_lock = HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "PROTOCOL_LOCK.json"
    if sha256(source_lock) != SOURCE_LOCK_SHA256:
        raise RuntimeError("A1 source protocol changed")
    assert_sources()
    return json.loads((EXPERIMENT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))


def make_experiment():
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    assert_architecture_args(args)
    exp = core.Exp_EZHybridLocalization(args)
    assert_cohort(exp)
    return exp


def source_grid(fold: int, epoch: int) -> dict:
    path = A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}_validation_grid.json"
    return json.loads(path.read_text(encoding="utf-8"))


def source_checkpoint(fold: int, epoch: int) -> Path:
    return A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(rows: list[dict], metric: str) -> float:
    return sum(float(row[metric]) for row in rows) / len(rows)


def assert_no_outer_loader(test_loader) -> None:
    if test_loader is not None:
        raise RuntimeError("Outer-test loader constructed")
