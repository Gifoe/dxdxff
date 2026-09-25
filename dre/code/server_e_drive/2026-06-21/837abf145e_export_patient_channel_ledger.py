"""Export a standardized patient-channel ledger from NeuroEZ-C prediction CSVs.

Filters by split_role (default ``test``) to avoid mixing val/test predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_patient_channel_ledger import normalize_ledger_dataframe


def _load_run_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_args_b0_pruned.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _coalesce(row: pd.Series, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def export_channel_ledger_from_run(
    run_dir: str | Path,
    *,
    split_role: str = "test",
    channel_glob: str = "*_channel_predictions_neuroez_v2_fold_*.csv",
) -> pd.DataFrame:
    """Export a patient-channel ledger from a run directory.

    Parameters
    ----------
    split_role:
        ``"test"`` (default), ``"val"``, or ``"all"``.
        ``"all"`` must only be used for debug; total diagnostic scripts
        must pass ``"test"`` explicitly.
    """
    run_dir = Path(run_dir)
    all_paths = sorted(run_dir.glob(channel_glob))
    if not all_paths:
        all_paths = sorted(run_dir.rglob(channel_glob))
    if not all_paths:
        raise FileNotFoundError(
            f"No channel prediction CSVs found under {run_dir} with glob {channel_glob!r}"
        )

    # filter by split_role
    role = str(split_role).strip().lower()
    if role != "all":
        filtered = [p for p in all_paths if p.name.startswith(f"{role}_")]
        if not filtered:
            raise FileNotFoundError(
                f"No '{role}_' channel CSVs found under {run_dir} (glob={channel_glob!r}). "
                f"Available: {[p.name for p in all_paths]}"
            )
    else:
        filtered = all_paths

    run_args = _load_run_args(run_dir)
    rows: list[dict[str, Any]] = []
    for path in filtered:
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            subject_id = str(_coalesce(row, "subject_id", "patient_id", default="unknown"))
            fold_id = int(_coalesce(row, "fold_idx", "fold_id", default=-1))
            csv_split = path.name.split("_")[0]
            split_role_val = str(_coalesce(row, "split", "split_role", default=csv_split))
            channel_name = str(_coalesce(row, "channel_name", default=f"ch{idx}"))
            rows.append(
                {
                    "config": run_dir.name,
                    "fold_id": fold_id,
                    "split_role": split_role_val,
                    "subject_id": subject_id,
                    "patient_id": str(_coalesce(row, "patient_id", default=subject_id)),
                    "record_id": str(
                        _coalesce(row, "record_id", "run_id", default=f"{subject_id}:fold{fold_id}:{split_role_val}")
                    ),
                    "center": str(_coalesce(row, "center", default="unknown")),
                    "center_id": int(_coalesce(row, "center_id", default=4)),
                    "channel_id": int(_coalesce(row, "channel_id", default=idx)),
                    "channel_name": channel_name,
                    "label_ez": float(_coalesce(row, "label_ez", "true_ez", default=0.0)),
                    "patient_ez_count": float(_coalesce(row, "patient_ez_count", "n_ez", default=0.0)),
                    "score_raw": float(
                        _coalesce(row, "score_raw", "score_ez_probability", "score_ez_final", default=0.0)
                    ),
                    "score_patient": float(
                        _coalesce(row, "score_patient", "score_ez_probability", "score_ez_final", default=0.0)
                    ),
                    "score_eval": float(
                        _coalesce(row, "score_eval", "score_ez_probability", "score_ez_final", default=0.0)
                    ),
                    "rank_eval": int(_coalesce(row, "rank_eval", "rank_ez_desc", default=-1)),
                    "pred_topk": int(_coalesce(row, "pred_topk", "predicted_ez", default=0)),
                    "is_top1": int(_coalesce(row, "is_top1", default=0)),
                    "temporal_pooling": run_args.get("temporal_pooling", ""),
                    "temporal_pooling_top_p": run_args.get("temporal_pooling_top_p", ""),
                    "temporal_pooling_tau": run_args.get("temporal_pooling_tau", ""),
                    "record_pooling": run_args.get("record_pooling", ""),
                    "record_pooling_top_p": run_args.get("record_pooling_top_p", ""),
                    "record_pooling_alpha": run_args.get("record_pooling_alpha", ""),
                }
            )
    ledger = pd.DataFrame(rows)
    if ledger["patient_ez_count"].fillna(0).astype(float).sum() <= 0:
        ledger["patient_ez_count"] = ledger.groupby("patient_id")["label_ez"].transform("sum")
    return normalize_ledger_dataframe(ledger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a standardized patient-channel ledger from NeuroEZ-C prediction CSVs."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--split_role", type=str, default="test", choices=["test", "val", "all"])
    parser.add_argument("--channel_glob", type=str, default="*_channel_predictions_neuroez_v2_fold_*.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_csv = Path(args.output_csv) if args.output_csv else run_dir / "patient_channel_ledger.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ledger = export_channel_ledger_from_run(
        run_dir, split_role=args.split_role, channel_glob=args.channel_glob,
    )
    ledger.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({len(ledger)} rows, split_role={args.split_role})")


if __name__ == "__main__":
    main()
