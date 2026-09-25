from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_patient_reports(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    report_dir = output_dir / "patient_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        subject_id = str(record["subject_id"])
        csv_path = report_dir / f"{subject_id}_channel_scores.csv"
        rows = list(record.get("channel_scores", []))
        with open(csv_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(
                fout,
                fieldnames=[
                    "subject_id",
                    "channel_name",
                    "rank",
                    "ez_score",
                    "predicted_ez",
                    "true_ez",
                    "predicted_count",
                    "is_tp",
                    "is_fp",
                    "is_fn",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        clean_record = {
            key: value
            for key, value in record.items()
            if key not in {"channel_scores", "temporal_attention", "seizure_attention"}
        }
        with open(report_dir / f"{subject_id}_report.json", "w", encoding="utf-8") as fout:
            json.dump(_jsonable(clean_record), fout, ensure_ascii=False, indent=2)

        if "temporal_attention" in record:
            np.save(report_dir / f"{subject_id}_temporal_attention.npy", np.asarray(record["temporal_attention"]))
        if "seizure_attention" in record:
            np.save(report_dir / f"{subject_id}_seizure_attention.npy", np.asarray(record["seizure_attention"]))


def write_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(_jsonable(summary), fout, ensure_ascii=False, indent=2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


__all__ = ["write_patient_reports", "write_summary"]

