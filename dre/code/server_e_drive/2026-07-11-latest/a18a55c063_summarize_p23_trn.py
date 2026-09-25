"""Aggregate completed P23 profile outputs without selecting on held-out metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in root.glob(f"*/seed_*/{name}"):
        frame = pd.read_csv(path)
        frame["profile"] = path.parents[1].name
        frame["seed"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root); output = root / "profile_comparison"; output.mkdir(parents=True, exist_ok=True)
    overall, fold, center = (_read(root, name) for name in ("p23_overall_summary.csv", "p23_fold_summary.csv", "p23_center_summary.csv"))
    overall.to_csv(output / "p23_P0_P6_overall.csv", index=False)
    fold.to_csv(output / "p23_P0_P6_by_fold.csv", index=False)
    center.to_csv(output / "p23_P0_P6_by_center.csv", index=False)
    if not overall.empty:
        oracle = overall[[column for column in overall.columns if "oracle" in column or column in {"profile", "seed"}]]
    else:
        oracle = overall
    oracle.to_csv(output / "p23_P0_P6_oracle.csv", index=False)
    overall[[column for column in overall.columns if "mrr" in column or "auprc" in column or column in {"profile", "seed"}]].to_csv(output / "p23_P0_P6_ranking.csv", index=False)
    (output / "P23_ABLATION_REPORT.md").write_text(
        "# P23 Profile Comparison\n\nOnly predeclared profiles and seeds are listed. Held-out values are reporting metrics, not a selection mechanism.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

