import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def get_first(d, names, default=None):
    if not isinstance(d, dict):
        return default
    for n in names:
        if n in d:
            return d[n]
    return default


def as_1d_array(x):
    if x is None:
        return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    return arr.reshape(-1)


def infer_label_array(rec):
    # 常见命名兜底
    candidates = [
        "channel_labels",
        "labels",
        "y",
        "channel_y",
        "ez_labels",
        "is_ez",
        "target",
        "targets",
    ]
    for k in candidates:
        if isinstance(rec, dict) and k in rec:
            arr = as_1d_array(rec[k])
            if arr is not None and arr.size > 0:
                return k, arr

    # 自动扫描：找长度像 channel 数、且值像 0/1 的数组
    if isinstance(rec, dict):
        for k, v in rec.items():
            arr = as_1d_array(v)
            if arr is None or arr.size < 2:
                continue
            vals = pd.Series(arr).dropna().unique()
            if len(vals) <= 3:
                try:
                    vals_f = set(float(z) for z in vals)
                    if vals_f.issubset({0.0, 1.0}):
                        return k, arr
                except Exception:
                    pass

    return None, None


def infer_records(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["records", "run_records", "all_records", "data", "samples"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise TypeError(f"Cannot infer records from cache object type: {type(obj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    cache_path = Path(args.cache_path)
    with cache_path.open("rb") as f:
        obj = pickle.load(f)

    records = infer_records(obj)

    rows = []
    debug_keys = None

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        if debug_keys is None:
            debug_keys = list(rec.keys())

        label_key, y = infer_label_array(rec)
        if y is None:
            rows.append({
                "record_index": i,
                "status": "NO_LABEL_FOUND",
                "label_key": None,
                "patient_id": get_first(rec, ["patient_id", "patient", "subject_id", "case_id"], f"record_{i}"),
                "run_id": get_first(rec, ["run_id", "seizure_id", "record_id", "file_id"], ""),
                "n_channels": np.nan,
                "n_ez": np.nan,
                "ez_ratio": np.nan,
            })
            continue

        y = np.asarray(y).astype(float).reshape(-1)
        finite = np.isfinite(y)
        y = y[finite]

        # 兼容 label 不是严格 0/1，而是 bool/int
        n_channels = int(y.size)
        n_ez = int(np.sum(y > 0.5))
        ratio = n_ez / n_channels if n_channels > 0 else np.nan

        rows.append({
            "record_index": i,
            "status": "OK",
            "label_key": label_key,
            "patient_id": get_first(rec, ["patient_id", "patient", "subject_id", "case_id"], f"record_{i}"),
            "run_id": get_first(rec, ["run_id", "seizure_id", "record_id", "file_id"], ""),
            "n_channels": n_channels,
            "n_ez": n_ez,
            "ez_ratio": ratio,
        })

    df = pd.DataFrame(rows)

    print("cache_path:", cache_path)
    print("num_records:", len(records))
    print("sample_keys:", debug_keys)
    print()
    print("status_counts:")
    print(df["status"].value_counts(dropna=False).to_string())
    print()

    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        print("No usable label arrays found. Need manual key inspection.")
        return

    print("EZ ratio summary by run record:")
    print(ok["ez_ratio"].describe().to_string())
    print()

    high = ok[ok["ez_ratio"] >= args.threshold].sort_values("ez_ratio", ascending=False)
    print(f"records with ez_ratio >= {args.threshold}: {len(high)}")
    if not high.empty:
        print(high[["patient_id", "run_id", "n_channels", "n_ez", "ez_ratio", "label_key"]].head(50).to_string(index=False))
    else:
        print("None found.")

    # patient-level 聚合：同一 patient 多 seizure/run 时看最大值和平均值
    by_patient = ok.groupby("patient_id").agg(
        n_records=("record_index", "count"),
        max_ez_ratio=("ez_ratio", "max"),
        mean_ez_ratio=("ez_ratio", "mean"),
        min_ez_ratio=("ez_ratio", "min"),
        total_channels=("n_channels", "sum"),
        total_ez=("n_ez", "sum"),
    ).reset_index()
    by_patient["pooled_ez_ratio"] = by_patient["total_ez"] / by_patient["total_channels"]

    high_p = by_patient[
        (by_patient["max_ez_ratio"] >= args.threshold) |
        (by_patient["pooled_ez_ratio"] >= args.threshold)
    ].sort_values("max_ez_ratio", ascending=False)

    print()
    print(f"patients with max_ez_ratio >= {args.threshold} or pooled_ez_ratio >= {args.threshold}: {len(high_p)}")
    if not high_p.empty:
        print(high_p.head(80).to_string(index=False))

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        by_patient.to_csv(out.with_name(out.stem + "_by_patient.csv"), index=False, encoding="utf-8-sig")
        print()
        print("saved:", out)
        print("saved:", out.with_name(out.stem + "_by_patient.csv"))


if __name__ == "__main__":
    main()
