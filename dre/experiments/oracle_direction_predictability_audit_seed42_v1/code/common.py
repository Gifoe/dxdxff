"""Frozen A1 development-only access. Public outputs never contain patient rows."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(PROJECT / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import choose, epoch_grid, finalize_fold, METRICS, THRESHOLDS  # noqa: E402
from run_development import assert_architecture_args, assert_cohort  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402

LOCK_SHA = "c7ebe090a42dd9f7c2c5361a37b2d5f6d2329972d4440f38d33659029af7639e"
SOURCE_LOCK_SHA = "6694da1b351d7130015661fcce1ec91bec643821392e409f1d5728f472b3aba0"
RUNTIME = Path(os.environ.get("ODPA_RUNTIME", ""))
A1_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))
EXPECTED_FOLDS = (0.6549997115717437, 0.6410958089865288, 0.6101185971670673,
                  0.6329791804569156, 0.5907877503876969)


def preflight():
    if not os.environ.get("ODPA_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("ODPA_RUNTIME must be an absolute private path")
    if not os.environ.get("A1_A2_RUNTIME") or not A1_RUNTIME.is_absolute():
        raise RuntimeError("A1_A2_RUNTIME must be an absolute private path")
    if sha256(ROOT / "PROTOCOL_LOCK.json") != LOCK_SHA:
        raise RuntimeError("Frozen direction-predictability protocol changed")
    if sha256(PROJECT / "a1_a2_patient_equal_objective_seed42_v1" / "PROTOCOL_LOCK.json") != SOURCE_LOCK_SHA:
        raise RuntimeError("A1 source protocol changed")
    assert_sources()


def experiment():
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    assert_architecture_args(args)
    exp = core.Exp_EZHybridLocalization(args)
    assert_cohort(exp)
    return exp


def checkpoint(fold, epoch):
    return A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"


def source_grid(fold, epoch):
    p = A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}_validation_grid.json"
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
