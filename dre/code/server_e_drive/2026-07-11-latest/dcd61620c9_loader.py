from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .audit import ReadAudit
from .bids_common import (
    bids_run_metadata,
    build_sidecar_index,
    discover_edf_files,
    find_participants_path,
    load_participants,
    read_ictal_bounds_from_events,
    resolve_bids_sidecars,
)
from .bids_loader import load_bids_patient_records
from .hup import load_hup_patient_records
from .lzu import load_lzu_patient_records
from .multicenter import load_multicenter_patient_records
from .schemas import DataInterfaceConfig, PatientRecord, coerce_config

try:
    from ..logging_utils import log
except Exception:
    def log(message: str) -> None:
        print(message)


def load_patient_records(args: Any | None = None) -> list[PatientRecord]:
    cfg = coerce_config(args)
    requested = {str(name).strip().lower() for name in cfg.datasets}
    audit = ReadAudit()
    patients: list[PatientRecord] = []
    skipped: list[str] = []
    log(f"Loading patient records. datasets={sorted(requested)}, audit_dir={cfg.read_audit_dir}")

    if "lzu" in requested:
        if cfg.lzu_root.exists() and cfg.lzu_ez_annotations_path.exists() and cfg.lzu_seizure_times_path.exists():
            loaded = load_lzu_patient_records(cfg, audit=audit)
            audit.add_loaded_records("lzu", loaded)
            patients.extend(loaded)
            log(f"LZU loaded: patients={len(loaded)}")
            if cfg.write_read_audit:
                audit.write(cfg.read_audit_dir)
                log(f"LZU read audit updated: {cfg.read_audit_dir}")
        else:
            reason = (
                f"missing_lzu_paths(root={cfg.lzu_root}, ez={cfg.lzu_ez_annotations_path}, "
                f"times={cfg.lzu_seizure_times_path})"
            )
            log(f"LZU skipped: {reason}")
            skipped.append(reason)
            audit.add_skipped_patient("lzu", "ALL", reason)
            if cfg.write_read_audit:
                audit.write(cfg.read_audit_dir)
                log(f"LZU read audit updated: {cfg.read_audit_dir}")

    if "hup" in requested:
        if cfg.hup_root.exists():
            loaded = load_hup_patient_records(cfg, audit=audit)
            audit.add_loaded_records("hup", loaded)
            patients.extend(loaded)
            log(f"HUP loaded: patients={len(loaded)}")
            if cfg.write_read_audit:
                audit.write(cfg.read_audit_dir)
                log(f"HUP read audit updated: {cfg.read_audit_dir}")
        else:
            reason = f"missing_hup_root(root={cfg.hup_root})"
            skipped.append(reason)
            audit.add_skipped_patient("hup", "ALL", reason)
            if cfg.write_read_audit:
                audit.write(cfg.read_audit_dir)
                log(f"HUP read audit updated: {cfg.read_audit_dir}")

    if requested.intersection({"sub", "subxxxx", "sub-xxxx", "multicenter", "bids"}):
        if cfg.multicenter_root.exists():
            loaded = load_multicenter_patient_records(cfg, audit=audit)
            audit.add_loaded_records("multicenter", loaded)
            patients.extend(loaded)
            log(f"Multicenter loaded: patients={len(loaded)}")
            if cfg.write_read_audit:
                audit.write(cfg.read_audit_dir)
                log(f"Multicenter read audit updated: {cfg.read_audit_dir}")
        else:
            reason = f"missing_multicenter_root(root={cfg.multicenter_root})"
            skipped.append(reason)
            audit.add_skipped_patient("multicenter", "ALL", reason)
            if cfg.write_read_audit:
                audit.write(cfg.read_audit_dir)
                log(f"Multicenter read audit updated: {cfg.read_audit_dir}")

    if cfg.debug_limit is not None:
        patients = patients[: int(cfg.debug_limit)]
    if cfg.write_read_audit:
        audit.write(cfg.read_audit_dir)
        log(f"Read audit written: {cfg.read_audit_dir}")
    if not patients and cfg.strict:
        raise FileNotFoundError("No patient records were loaded. Missing/skipped datasets: " + "; ".join(skipped))
    log(f"Finished loading patient records. total_patients={len(patients)}, skipped={len(skipped)}")
    return patients


def discover_bids_run_table(
    root: str | Path,
    sidecar_root: str | Path | None = None,
    participants_path: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(root)
    sidecar_root_path = Path(sidecar_root) if sidecar_root is not None else None
    sidecar_index = build_sidecar_index(sidecar_root_path)
    participants = load_participants(participants_path or find_participants_path(root, sidecar_root_path, "hup"))
    rows: list[dict] = []
    for edf_path in discover_edf_files(root):
        meta = bids_run_metadata(edf_path, root)
        sidecars = resolve_bids_sidecars(edf_path, sidecar_root_path, sidecar_index)
        onset, offset = read_ictal_bounds_from_events(sidecars["events"])
        participant_meta = participants.get(str(meta["subject_id"]), {})
        rows.append(
            {
                **meta,
                "edf_path": str(edf_path),
                "channels_path": str(sidecars["channels"]) if sidecars["channels"] else None,
                "events_path": str(sidecars["events"]) if sidecars["events"] else None,
                "json_path": str(sidecars["json"]) if sidecars["json"] else None,
                "seizure_onset_sec": onset,
                "seizure_offset_sec": offset,
                **{f"participant_{key}": value for key, value in participant_meta.items()},
            }
        )
    return pd.DataFrame(rows)


__all__ = ["discover_bids_run_table", "load_bids_patient_records", "load_lzu_patient_records", "load_patient_records"]
