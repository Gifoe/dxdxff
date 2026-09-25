from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..clean_nez_prototype import CleanNEZPrototype
from ..seizure_nez_prototype import seizure_prototype_probability


P2_AUX_FEATURES=("p2__outside_final_nez_q10","p2__outside_final_residual_top10","p2__outside_persistent_abnormality_top10","p2__outside_inside_nez_gap")


def _top10(value:torch.Tensor)->float:
    k=min(value.numel(),max(1,math.ceil(value.numel()*.10)));return float(torch.topk(value,k).values.mean())


def fit_seizure_clean_nez_prototype(evidence:Sequence[dict[str,Any]],train_subjects:set[str])->CleanNEZPrototype:
    patients=[]
    for record in evidence:
        if record["patient_key"] not in train_subjects:continue
        outside=record["channel_mask"][0].bool()&~record["clinical_target_mask"][0].bool();valid=record["seizure_mask"][0].bool().unsqueeze(-1)&record["seizure_channel_mask"][0].bool()&outside.unsqueeze(0)
        local=record["seizure_channel_embedding"][0][valid]
        if local.numel():patients.append(local)
    return CleanNEZPrototype("cop_seizure_channel_embedding").fit(patients)


def build_p2_auxiliary(evidence:Sequence[dict[str,Any]],prototype:CleanNEZPrototype,*,target_permutation:bool=False,seed:int=42)->pd.DataFrame:
    rows=[];rng=np.random.default_rng(seed)
    for record in evidence:
        valid=record["channel_mask"][0].bool();target=record["clinical_target_mask"][0].bool().clone()
        if target_permutation:
            idx=torch.where(valid)[0].numpy();values=target[idx].numpy();target[idx]=torch.from_numpy(rng.permutation(values))
        outside=valid&~target;inside=valid&target
        if not outside.any() or not inside.any():raise ValueError(f"COP P2 target/outside empty: {record['patient_key']}")
        q=torch.sigmoid(record["final_nez_logit"][0].float());residual=1-q
        seizure_valid=record["seizure_mask"][0].bool().unsqueeze(-1)&record["seizure_channel_mask"][0].bool()&valid.unsqueeze(0)
        q_seizure=seizure_prototype_probability(record["seizure_channel_embedding"][0].float(),seizure_valid,prototype);persistent=torch.zeros_like(q)
        for channel in torch.where(valid)[0]:
            local=(1-q_seizure[:,channel])[seizure_valid[:,channel]]
            if local.numel():persistent[channel]=torch.quantile(local,.10)
        rows.append({"patient_key":record["patient_key"],"center":record["center"],"p2__outside_final_nez_q10":float(torch.quantile(q[outside],.10)),
            "p2__outside_final_residual_top10":_top10(residual[outside]),"p2__outside_persistent_abnormality_top10":_top10(persistent[outside]),
            "p2__outside_inside_nez_gap":float(q[outside].mean()-q[inside].mean())})
    return pd.DataFrame(rows)


__all__=["P2_AUX_FEATURES","build_p2_auxiliary","fit_seizure_clean_nez_prototype"]
