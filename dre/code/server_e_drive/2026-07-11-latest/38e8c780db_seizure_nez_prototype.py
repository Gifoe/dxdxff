from __future__ import annotations

import math
import torch

from .clean_nez_prototype import CleanNEZPrototype


SEIZURE_STABILITY_FEATURES = (
    "outside_persistent_mean", "outside_persistent_max", "outside_persistent_top10",
    "outside_worstcase_mean", "outside_worstcase_max", "outside_worstcase_top10",
    "outside_abnormal_seizure_fraction", "outside_temporal_std",
)


def seizure_prototype_probability(embedding: torch.Tensor, valid: torch.Tensor, prototype: CleanNEZPrototype) -> torch.Tensor:
    output = torch.zeros(valid.shape, dtype=torch.float32, device=embedding.device)
    if valid.any(): output[valid] = prototype.probability_nez(embedding[valid].detach().cpu()).to(embedding.device)
    return output


def _top10(values: torch.Tensor) -> torch.Tensor:
    return torch.topk(values, min(values.numel(), max(1, math.ceil(values.numel()*.10)))).values.mean()


def compute_seizure_nez_stability(q_seizure_nez: torch.Tensor, valid: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    bsz, _, channels = q_seizure_nez.shape
    stable = torch.zeros((bsz,channels),device=q_seizure_nez.device); persistent=torch.zeros_like(stable); worstcase=torch.zeros_like(stable); temporal_std=torch.zeros_like(stable)
    rows={name:[] for name in SEIZURE_STABILITY_FEATURES}; audits=[]
    for p in range(bsz):
        for c in range(channels):
            local=q_seizure_nez[p,:,c][valid[p,:,c].bool()]
            if local.numel():
                stable[p,c]=torch.quantile(local,.10); abnormal=1-local; persistent[p,c]=torch.quantile(abnormal,.10); worstcase[p,c]=1-torch.quantile(local,.10); temporal_std[p,c]=abnormal.std(unbiased=False)
        outside=channel_mask[p].bool()&~target[p].bool(); op=persistent[p][outside]; ow=worstcase[p][outside]; ot=temporal_std[p][outside]
        seizure_abnormal=(q_seizure_nez[p] < .5)&valid[p]
        fraction=seizure_abnormal[:,outside].float().sum()/valid[p,:,outside].float().sum().clamp_min(1)
        values={"outside_persistent_mean":op.mean(),"outside_persistent_max":op.max(),"outside_persistent_top10":_top10(op),"outside_worstcase_mean":ow.mean(),"outside_worstcase_max":ow.max(),"outside_worstcase_top10":_top10(ow),"outside_abnormal_seizure_fraction":fraction,"outside_temporal_std":ot.mean()}
        for name in SEIZURE_STABILITY_FEATURES: rows[name].append(values[name])
        local_all=q_seizure_nez[p][valid[p]]; audits.append({"n_valid_seizures":int(valid[p].any(-1).sum()),"q_seizure_nez_mean":float(local_all.mean()),"q_seizure_nez_std":float(local_all.std(unbiased=False)),"persistent_abnormality_mean":float(persistent[p][channel_mask[p]].mean()),"worstcase_abnormality_mean":float(worstcase[p][channel_mask[p]].mean()),"outside_persistent_top10":float(values["outside_persistent_top10"]),"outside_worstcase_top10":float(values["outside_worstcase_top10"])})
    output={name:torch.stack(value) for name,value in rows.items()}; output.update({"stable_nez_q10":stable,"persistent_abnormality":persistent,"worstcase_abnormality":worstcase,"seizure_nez_stability_audit":audits}); return output


__all__=["SEIZURE_STABILITY_FEATURES","compute_seizure_nez_stability","seizure_prototype_probability"]
