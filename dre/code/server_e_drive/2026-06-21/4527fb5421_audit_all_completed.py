from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def center_of(s: str) -> str:
    return str(s).split(":", 1)[0].lower() if ":" in str(s) else "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("test_patient_predictions_neuroez_v2_fold_*.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        df["center"] = df["subject_id"].map(center_of)
        for center, g in df.groupby("center"):
            rows.append({
                "config": d.name,
                "center": center,
                "n_patients": int(g["subject_id"].nunique()),
                "patient_macro_f1": float(g["patient_macro_f1"].mean()),
                "patient_ez_f1": float(g["patient_ez_f1"].mean()),
                "ez_mrr": float(g["ez_mrr"].mean()),
                "ez_recall_at_true_count": float(g["ez_recall_at_true_count"].mean()),
            })
    if rows:
        out = pd.DataFrame(rows).sort_values(["config", "center"])
        out.to_csv(root / "a0_to_a6_by_center_patient_summary.csv", index=False)
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
