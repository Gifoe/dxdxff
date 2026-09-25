from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List


KEY_METRICS = [
    "patient_macro_accuracy",
    "patient_macro_balanced_accuracy",
    "patient_macro_f1",
    "patient_macro_ez_precision",
    "patient_macro_ez_recall",
    "patient_macro_ez_f1",
    "patient_macro_auroc_ez",
    "patient_macro_auprc_ez",
    "patient_macro_ez_recall_at_true_count",
    "patient_macro_ez_mrr",
    "pooled_accuracy",
    "pooled_balanced_accuracy",
    "pooled_macro_f1",
    "pooled_auroc_ez",
    "pooled_auprc_ez",
    "ez_auprc",
    "ez_auc_roc",
    "ez_mrr",
    "ez_recall_at_true_count",
]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as fin:
        return list(csv.DictReader(fin))


def _write_csv(path: Path, rows: List[Dict[str, object]], preferred_fields: Iterable[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_order: List[str] = []
    for field in preferred_fields:
        if field not in field_order:
            field_order.append(field)
    for row in rows:
        for field in row.keys():
            if field not in field_order:
                field_order.append(field)
    with open(path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / float(len(values) - 1))


def aggregate(output_root: Path) -> None:
    manifest_path = output_root / "experiment_manifest.csv"
    manifest_rows = _read_csv_rows(manifest_path)
    manifest = {row["experiment"]: row for row in manifest_rows if row.get("experiment")}

    combined_rows: List[Dict[str, object]] = []
    for summary_path in sorted(output_root.glob("*/heldout_summary_neuroez_v3.csv")):
        experiment = summary_path.parent.name
        summary_rows = _read_csv_rows(summary_path)
        if not summary_rows:
            continue
        row: Dict[str, object] = {}
        row.update(manifest.get(experiment, {}))
        row.update(summary_rows[0])
        row["experiment"] = experiment
        row["summary_path"] = str(summary_path)
        combined_rows.append(row)

    preferred = [
        "priority",
        "experiment",
        "family",
        "variant",
        "seed",
        "threshold_tuning_metric",
        "early_stop_metric",
        "notes",
    ] + KEY_METRICS + ["summary_path"]
    _write_csv(output_root / "combined_heldout_summary.csv", combined_rows, preferred_fields=preferred)

    key_rows: List[Dict[str, object]] = []
    for row in combined_rows:
        key_row = {
            "priority": row.get("priority", ""),
            "experiment": row.get("experiment", ""),
            "family": row.get("family", ""),
            "variant": row.get("variant", ""),
            "seed": row.get("seed", ""),
            "threshold_tuning_metric": row.get("threshold_tuning_metric", ""),
            "early_stop_metric": row.get("early_stop_metric", ""),
            "notes": row.get("notes", ""),
        }
        for metric in KEY_METRICS:
            key_row[metric] = row.get(metric, "")
        key_rows.append(key_row)
    _write_csv(output_root / "combined_key_metrics.csv", key_rows)

    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in combined_rows:
        key = (str(row.get("family", "")), str(row.get("threshold_tuning_metric", "")))
        grouped.setdefault(key, []).append(row)

    group_rows: List[Dict[str, object]] = []
    for (family, threshold_metric), rows in sorted(grouped.items()):
        group_row: Dict[str, object] = {
            "family": family,
            "threshold_tuning_metric": threshold_metric,
            "n": len(rows),
            "experiments": ";".join(str(row.get("experiment", "")) for row in rows),
        }
        for metric in KEY_METRICS:
            values = [value for value in (_to_float(row.get(metric)) for row in rows) if value is not None]
            group_row[f"{metric}_mean"] = _mean(values) if values else ""
            group_row[f"{metric}_std"] = _std(values) if values else ""
        group_rows.append(group_row)
    _write_csv(output_root / "group_mean_std_key_metrics.csv", group_rows)

    print(f"Aggregated {len(combined_rows)} completed experiment(s) under {output_root}")
    print(f"Wrote {output_root / 'combined_heldout_summary.csv'}")
    print(f"Wrote {output_root / 'combined_key_metrics.csv'}")
    print(f"Wrote {output_root / 'group_mean_std_key_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate NeuroEZ heldout summaries under one output root.")
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.output_root)


if __name__ == "__main__":
    main()
