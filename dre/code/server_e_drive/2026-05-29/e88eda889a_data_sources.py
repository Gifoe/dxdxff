from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from utils import log, norm_text


@dataclass(frozen=True)
class BidsRun:
    subject_id: str
    session: str | None
    stem: str
    run_id: str
    task: str
    run_num: str
    phase_group: str
    edf_path: Path
    channels_path: Path | None
    events_path: Path | None
    json_path: Path | None


@dataclass
class QualityDecision:
    status: str
    kept: bool
    drop_reason: str
    matched_by: str


def _parse_bids_entities(stem: str) -> tuple[str, str | None, str, str]:
    subject_match = re.search(r"(sub-[A-Za-z0-9]+)", stem)
    session_match = re.search(r"ses-([A-Za-z0-9]+)", stem)
    task_match = re.search(r"task-([A-Za-z0-9]+)", stem)
    run_match = re.search(r"run-([0-9]+)", stem)
    subject_id = subject_match.group(1) if subject_match else "unknown"
    session = session_match.group(1) if session_match else None
    task = task_match.group(1).lower() if task_match else "unknown"
    run_num = run_match.group(1) if run_match else "01"
    return subject_id, session, task, run_num


def _phase_from_task(task: str) -> str:
    task_lower = task.lower()
    if "interictal" in task_lower:
        return "interictal"
    if "ictal" in task_lower:
        return "ictal"
    return "unknown"


def discover_hup_bids_runs(dataset_dir: Path) -> list[BidsRun]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    edfs = [path for path in dataset_dir.rglob("*") if path.is_file() and path.name.lower().endswith("_ieeg.edf")]
    runs: list[BidsRun] = []
    for edf_path in sorted(edfs):
        stem = re.sub(r"_ieeg\.edf$", "", edf_path.name, flags=re.IGNORECASE)
        subject_id, session, task, run_num = _parse_bids_entities(stem)
        if not subject_id.upper().startswith("SUB-HUP"):
            continue
        channels_path = edf_path.parent / f"{stem}_channels.tsv"
        events_path = edf_path.parent / f"{stem}_events.tsv"
        json_path = edf_path.parent / f"{stem}_ieeg.json"
        run_id = f"{subject_id}__ses-{session}__{task}__run-{run_num}" if session else f"{subject_id}__{task}__run-{run_num}"
        runs.append(
            BidsRun(
                subject_id=subject_id,
                session=session,
                stem=stem,
                run_id=run_id,
                task=task,
                run_num=run_num,
                phase_group=_phase_from_task(task),
                edf_path=edf_path,
                channels_path=channels_path if channels_path.exists() else None,
                events_path=events_path if events_path.exists() else None,
                json_path=json_path if json_path.exists() else None,
            )
        )
    if not runs:
        raise RuntimeError(f"No HUP BIDS EDFs found under {dataset_dir}. Expected sub-HUP*/**/*_ieeg.edf files.")
    return runs


def _pick_column(columns: Sequence[str], required_tokens: Sequence[str], fallback_tokens: Sequence[str]) -> str | None:
    normalized = [(col, str(col).strip().lower()) for col in columns]
    for col, norm in normalized:
        if all(token.lower() in norm for token in required_tokens):
            return col
    for col, norm in normalized:
        if any(token.lower() in norm for token in fallback_tokens):
            return col
    return None


def _quality_run_key(subject_id: str, run_text: str) -> tuple[str, str, str]:
    text = str(run_text)
    task_match = re.search(r"task-([A-Za-z0-9]+)", text)
    run_match = re.search(r"run-([0-9]+)", text)
    task = task_match.group(1).lower() if task_match else "unknown"
    run_num = run_match.group(1) if run_match else "01"
    return subject_id.upper().replace("SUB-", "HUP"), task, run_num


def load_quality_table(
    quality_report: Path,
    *,
    allowed_quality: set[str],
    preview_aliases: set[str],
) -> tuple[dict[tuple[str, str, str], str], dict[str, str]]:
    if not quality_report.exists():
        raise FileNotFoundError(f"Quality report does not exist: {quality_report}")
    xls = pd.ExcelFile(quality_report)
    selected_df: pd.DataFrame | None = None
    selected_cols: dict[str, str] | None = None
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df.empty:
            continue
        subject_col = _pick_column(df.columns, ("患者",), ("subject", "patient", "id"))
        run_col = _pick_column(df.columns, ("发作",), ("run", "seizure", "edf", "文件"))
        quality_col = _pick_column(df.columns, ("质量",), ("quality", "status", "category", "评级"))
        if subject_col and run_col and quality_col:
            selected_df = df.copy()
            selected_cols = {"sheet": sheet, "subject_col": subject_col, "run_col": run_col, "quality_col": quality_col}
            break
    if selected_df is None or selected_cols is None:
        raise RuntimeError(f"Could not infer subject/run/quality columns from {quality_report}. Sheets={xls.sheet_names}")

    detected = sorted({norm_text(v) for v in selected_df[selected_cols["quality_col"]].dropna().unique()})
    effective_allowed = set(allowed_quality)
    if "preview" in allowed_quality and "preview" not in detected:
        alias_hits = preview_aliases.intersection(detected)
        if alias_hits:
            effective_allowed.update(alias_hits)
            log(f"Quality report has no literal PREVIEW; treating {sorted(alias_hits)} as preview aliases.")
    log(f"Detected quality columns: {selected_cols}; statuses={detected}; effective_allowed={sorted(effective_allowed)}")

    quality_by_run: dict[tuple[str, str, str], str] = {}
    for _, row in selected_df.iterrows():
        subject = str(row[selected_cols["subject_col"]]).strip().upper().replace("SUB-", "")
        if not subject.startswith("HUP"):
            continue
        key = _quality_run_key(subject, str(row[selected_cols["run_col"]]))
        quality_by_run[key] = norm_text(row[selected_cols["quality_col"]])
    selected_cols["effective_allowed"] = ",".join(sorted(effective_allowed))
    return quality_by_run, selected_cols


def quality_decision_for_run(
    run: BidsRun,
    quality_by_run: dict[tuple[str, str, str], str],
    *,
    allowed_quality: set[str],
    interictal_quality_policy: str,
) -> QualityDecision:
    subject_key = run.subject_id.upper().replace("SUB-", "")
    exact_key = (subject_key, run.task.lower(), run.run_num)
    status = quality_by_run.get(exact_key)
    if status is not None:
        kept = status in allowed_quality
        return QualityDecision(status, kept, "" if kept else f"quality_not_allowed:{status}", "exact_quality_row")
    if run.phase_group == "interictal" and interictal_quality_policy == "subject_if_any_kept":
        has_allowed_subject_run = any(subj == subject_key and value in allowed_quality for (subj, _, _), value in quality_by_run.items())
        if has_allowed_subject_run:
            return QualityDecision("subject_has_allowed_ictal_quality", True, "", "subject_if_any_kept")
    return QualityDecision("missing", False, "missing_quality_row", "none")
