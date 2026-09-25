from __future__ import annotations

import numpy as np
import pandas as pd
from ..functional_graph import normalize_channel_name

FEATURES=("outside_nez_q10","outside_nez_q25","outside_non_nez_q90","outside_non_nez_top10_mean","non_nez_capture","true_ez_nez_median","outside_nez_median","true_ez_vs_outside_nez_gap","outside_high_non_nez_fraction")


def load_p2_oof_channel_ledger(path, original_folds: pd.DataFrame, strict=True):
    frame=pd.read_csv(path); aliases={"subject_id":"patient_key","channel_name":"channel","q_nez":"probability_nez","final_nez_probability":"probability_nez","fold":"outer_fold"}; frame=frame.rename(columns={k:v for k,v in aliases.items() if k in frame})
    required={"patient_key","channel","probability_nez","outer_fold"}; missing=required-set(frame)
    if missing: raise ValueError(f"P2 OOF ledger missing columns: {sorted(missing)}")
    frame["patient_key"]=frame.patient_key.astype(str); frame["channel"]=frame.channel.map(normalize_channel_name)
    expected=original_folds[["patient_key","outer_fold"]].copy(); expected.patient_key=expected.patient_key.astype(str)
    check=frame[["patient_key","outer_fold"]].drop_duplicates().merge(expected,on="patient_key",suffixes=("_p2","_original"),validate="many_to_one")
    if strict and (check.outer_fold_p2.astype(int)!=check.outer_fold_original.astype(int)).any(): raise ValueError("P2 ledger is not original-fold OOF")
    if frame.duplicated(["patient_key","channel"]).any(): raise ValueError("P2 OOF ledger has duplicate patient-channel rows")
    return frame


def patient_nez_features(channel_frame: pd.DataFrame) -> dict[str,float]:
    q=channel_frame.probability_nez.to_numpy(float); target=channel_frame.true_ez.astype(bool).to_numpy(); valid=np.isfinite(q)&channel_frame.get("valid",pd.Series(True,index=channel_frame.index)).astype(bool).to_numpy(); outside=~target&valid; inside=target&valid; non=1-q
    def quant(x,m,z): return float(np.quantile(x[m],z)) if m.any() else np.nan
    top=np.sort(non[outside]); top=top[-max(1,int(np.ceil(.1*len(top)))):] if len(top) else np.asarray([np.nan])
    denominator=np.sum(non[valid])
    return {"outside_nez_q10":quant(q,outside,.1),"outside_nez_q25":quant(q,outside,.25),"outside_non_nez_q90":quant(non,outside,.9),"outside_non_nez_top10_mean":float(np.nanmean(top)),"non_nez_capture":float(np.sum(non[inside])/denominator) if denominator>0 else np.nan,"true_ez_nez_median":quant(q,inside,.5),"outside_nez_median":quant(q,outside,.5),"true_ez_vs_outside_nez_gap":quant(q,inside,.5)-quant(q,outside,.5),"outside_high_non_nez_fraction":float(np.mean(non[outside]>.8)) if outside.any() else np.nan}


__all__=["FEATURES","load_p2_oof_channel_ledger","patient_nez_features"]
