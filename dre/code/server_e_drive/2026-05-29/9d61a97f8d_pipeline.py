from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

from cache_io import write_cache
from config import VERSIONS
from data_sources import discover_hup_bids_runs, load_quality_table, quality_decision_for_run
from events import parse_all_seizure_intervals, remote_intervals_for_run
from extraction import extract_b0_record, extract_interictal_bank_from_run, extract_strict_ictal_records
from utils import log, parse_csv_set, write_csv
from variants import attach_interictal_and_make_variant


def _limit_worker_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _cache_worker_initializer() -> None:
    _limit_worker_threads()
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


def _resolve_cache_workers(args, task_count: int) -> int:
    requested = int(getattr(args, "cache_num_workers", 16))
    if requested <= 1 or task_count <= 1:
        return 1
    return max(1, min(requested, task_count, os.cpu_count() or requested))


def _run_parallel(tasks, worker, *, args, label: str) -> list[Any]:
    workers = _resolve_cache_workers(args, len(tasks))
    if not tasks:
        return []
    log(f"{label}: tasks={len(tasks)} cache_workers={workers}")
    if workers == 1:
        return [worker(task) for task in tasks]

    results: list[Any] = []
    progress_interval = max(1, len(tasks) // 20)
    with ProcessPoolExecutor(max_workers=workers, initializer=_cache_worker_initializer) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed == 1 or completed == len(tasks) or completed % progress_interval == 0:
                log(f"{label}: progress {completed}/{len(tasks)}")
    return results


def _extract_ictal_task(payload: tuple[Any, Any]) -> dict[str, Any]:
    run, args = payload
    try:
        intervals = parse_all_seizure_intervals(run.events_path)
        b0_record = extract_b0_record(run, args)
        records, strict_qc = extract_strict_ictal_records(run, args)
        return {
            "ok": True,
            "run": run,
            "num_seizures": len(intervals),
            "b0_record": b0_record,
            "strict_records": records,
            "strict_qc": strict_qc,
        }
    except Exception as exc:
        return {
            "ok": False,
            "run": run,
            "error": f"{type(exc).__name__}: {exc}",
            "num_seizures": 0,
            "b0_record": None,
            "strict_records": [],
            "strict_qc": [{"run_id": run.run_id, "drop_reason": "worker_exception", "error": f"{type(exc).__name__}: {exc}"}],
        }


def _extract_explicit_interictal_task(payload: tuple[Any, Any]) -> dict[str, Any]:
    run, args = payload
    try:
        bank, info = extract_interictal_bank_from_run(run, args, source_type="explicit_interictal")
        return {"ok": True, "run": run, "bank": bank, "info": info}
    except Exception as exc:
        return {
            "ok": False,
            "run": run,
            "bank": None,
            "info": {"run_id": run.run_id, "drop_reason": "worker_exception", "error": f"{type(exc).__name__}: {exc}"},
        }


def _extract_remote_interictal_task(payload: tuple[Any, Any]) -> dict[str, Any]:
    run, args = payload
    try:
        intervals, buffer_used = remote_intervals_for_run(run, args)
        if not intervals:
            return {"ok": True, "run": run, "bank": None, "info": {"run_id": run.run_id, "drop_reason": "no_remote_interictal_intervals"}, "remote_intervals": 0}
        bank, info = extract_interictal_bank_from_run(
            run,
            args,
            source_type="remote_interictal",
            intervals=intervals,
            buffer_used_sec=buffer_used,
        )
        return {"ok": True, "run": run, "bank": bank, "info": info, "remote_intervals": len(intervals)}
    except Exception as exc:
        return {
            "ok": False,
            "run": run,
            "bank": None,
            "info": {"run_id": run.run_id, "drop_reason": "worker_exception", "error": f"{type(exc).__name__}: {exc}"},
            "remote_intervals": 0,
        }


def build_caches(args) -> dict[str, Path]:
    _limit_worker_threads()
    allowed = parse_csv_set(args.allowed_edf_quality)
    preview_aliases = parse_csv_set(args.preview_quality_aliases)
    quality_by_run, quality_cols = load_quality_table(
        Path(args.edf_quality_report),
        allowed_quality=allowed,
        preview_aliases=preview_aliases,
    )
    effective_allowed = set(quality_cols["effective_allowed"].split(","))
    runs = discover_hup_bids_runs(Path(args.dataset_dir))
    decisions = {
        run.run_id: quality_decision_for_run(
            run,
            quality_by_run,
            allowed_quality=effective_allowed,
            interictal_quality_policy=str(args.interictal_quality_policy),
        )
        for run in runs
    }
    kept_runs = [run for run in runs if decisions[run.run_id].kept]
    ictal_runs = [run for run in kept_runs if run.phase_group == "ictal"]
    explicit_interictal_runs = [run for run in kept_runs if run.phase_group == "interictal"]
    log(
        f"HUP runs discovered={len(runs)} kept={len(kept_runs)} dropped={len(runs)-len(kept_runs)} "
        f"ictal_kept={len(ictal_runs)} explicit_interictal_kept={len(explicit_interictal_runs)}"
    )

    qc_rows: list[dict[str, Any]] = []
    b0_records: list[dict[str, Any]] = []
    strict_records: list[dict[str, Any]] = []
    skipped_seizures = 0
    ictal_results = _run_parallel([(run, args) for run in ictal_runs], _extract_ictal_task, args=args, label="ictal extraction")
    ictal_result_by_run_id = {result["run"].run_id: result for result in ictal_results}
    for run in ictal_runs:
        decision = decisions[run.run_id]
        result = ictal_result_by_run_id.get(run.run_id)
        if result is None:
            continue
        b0_record = result["b0_record"]
        if b0_record is not None:
            b0_records.append(b0_record)
        records = result["strict_records"]
        strict_qc = result["strict_qc"]
        strict_records.extend(records)
        skipped_seizures += len([row for row in strict_qc if row.get("drop_reason")])
        qc_rows.append(
            {
                "subject_id": run.subject_id,
                "edf_path": str(run.edf_path),
                "run_id": run.run_id,
                "run_type": run.phase_group,
                "quality_status": decision.status,
                "kept_or_dropped": "kept",
                "drop_reason": "" if result.get("ok") else result.get("error", "worker_exception"),
                "has_events": int(run.events_path is not None),
                "num_seizures": result["num_seizures"],
                "num_interictal_windows": 0,
                "num_ictal_windows": sum(int(rec["sample"]["num_ictal_windows"]) for rec in records),
                "low_ictal_coverage": sum(int(rec["sample"].get("low_ictal_coverage", 0)) for rec in records),
                "interictal_source_type": "",
                "quality_matched_by": decision.matched_by,
            }
        )
        for row in strict_qc:
            if row.get("drop_reason"):
                qc_rows.append({"subject_id": run.subject_id, "edf_path": str(run.edf_path), **row})

    for run in runs:
        decision = decisions[run.run_id]
        if not decision.kept:
            qc_rows.append(
                {
                    "subject_id": run.subject_id,
                    "edf_path": str(run.edf_path),
                    "run_id": run.run_id,
                    "run_type": run.phase_group,
                    "quality_status": decision.status,
                    "kept_or_dropped": "dropped",
                    "drop_reason": decision.drop_reason,
                    "has_events": int(run.events_path is not None),
                    "quality_matched_by": decision.matched_by,
                }
            )

    banks_by_subject = {}
    explicit_results = _run_parallel(
        [(run, args) for run in explicit_interictal_runs],
        _extract_explicit_interictal_task,
        args=args,
        label="explicit interictal extraction",
    )
    for result in explicit_results:
        run = result["run"]
        bank = result["bank"]
        info = result["info"]
        qc_rows.append({"subject_id": run.subject_id, "edf_path": str(run.edf_path), "run_type": "interictal", **info})
        if bank is not None:
            banks_by_subject.setdefault(run.subject_id, []).append(bank)

    subjects_needing_remote = {
        subject
        for subject in {rec["subject_id"] for rec in strict_records}
        if sum(bank.features.shape[0] for bank in banks_by_subject.get(subject, [])) < int(args.interictal_min_windows)
    }
    remote_intervals_count = 0
    remote_runs = [run for run in ictal_runs if run.subject_id in subjects_needing_remote]
    remote_results = _run_parallel(
        [(run, args) for run in remote_runs],
        _extract_remote_interictal_task,
        args=args,
        label="remote interictal extraction",
    )
    for result in remote_results:
        run = result["run"]
        bank = result["bank"]
        info = result["info"]
        remote_intervals_count += int(result.get("remote_intervals", 0))
        qc_rows.append({"subject_id": run.subject_id, "edf_path": str(run.edf_path), "run_type": "remote_interictal", **info})
        if bank is not None:
            banks_by_subject.setdefault(run.subject_id, []).append(bank)

    write_csv(Path(args.output_root) / "hup_good_preview_qc.csv", qc_rows)
    if not b0_records:
        raise RuntimeError("No B0 records were produced after HUP quality filtering.")
    if not strict_records:
        raise RuntimeError("No strict ictal records were produced after HUP quality filtering.")

    cache_root = Path(args.output_root) / "_caches"
    caches: dict[str, Path] = {}
    b0_path = cache_root / "B0_HUP_Filtered_window_cache.pkl"
    write_cache(b0_path, source_center="hup", run_records=b0_records, extra={"quality_columns": quality_cols, "sample_mode": "current_baseline"})
    caches["B0_HUP_Filtered"] = b0_path

    version_stats = {}
    for version in VERSIONS[1:]:
        variant_records, stats = attach_interictal_and_make_variant(strict_records, banks_by_subject, variant=version)
        if not variant_records:
            raise RuntimeError(f"{version}: no records have a usable interictal baseline.")
        path = cache_root / f"{version}_window_cache.pkl"
        write_cache(path, source_center="hup", run_records=variant_records, extra={"quality_columns": quality_cols, "sample_mode": "strict_ictal", "variant": version, "variant_stats": stats})
        caches[version] = path
        version_stats[version] = stats

    log(f"Number of HUP patients kept: {len({record['subject_id'] for record in b0_records})}")
    log(f"Number of EDFs good/preview kept: {len(kept_runs)}")
    log(f"Number of explicit interictal runs: {len(explicit_interictal_runs)}")
    log(f"Number of remote interictal intervals: {remote_intervals_count}")
    log(f"Number of ictal runs: {len(ictal_runs)}")
    log(f"Number of skipped seizures: {skipped_seizures}")
    log("Variant cache stats: " + json.dumps(version_stats, indent=2, ensure_ascii=False))
    return caches
