from __future__ import annotations
import math
import torch
from torch import nn

class CrossSeizureRecurrenceAggregator(nn.Module):
    def __init__(self,model_dim=32,dropout=.4):
        super().__init__(); self.proj=nn.Sequential(nn.Linear(4*model_dim+1,2*model_dim),nn.GELU(),nn.Dropout(dropout),nn.LayerNorm(2*model_dim))
    def forward(self,emb,risk,seizure_mask,seizure_channel_mask):
        valid=seizure_mask[:,:,None]&seizure_channel_mask; w=valid.float(); raw_count=w.sum(1); safe_count=raw_count.clamp_min(1); observed_channel_mask=raw_count>0
        mean=(emb*w[...,None]).sum(1)/safe_count[...,None]
        variance=((((emb-mean[:,None])**2)*w[...,None]).sum(1)/safe_count[...,None])
        std=torch.where(variance>0,torch.sqrt(variance.clamp_min(1e-8)),torch.zeros_like(variance))
        median=torch.zeros_like(mean); maximum=torch.zeros_like(mean)
        risk_median=torch.zeros_like(risk[:,0]); risk_max=torch.zeros_like(risk[:,0])
        for bi in range(emb.shape[0]):
            for ci in range(emb.shape[2]):
                idx=torch.where(valid[bi,:,ci])[0]
                if idx.numel():
                    vals=emb[bi,idx,ci]; sorted_vals=torch.sort(vals,dim=0).values; count=idx.numel()
                    median[bi,ci]=sorted_vals[count//2] if count%2 else .5*(sorted_vals[count//2-1]+sorted_vals[count//2])
                    maximum[bi,ci]=vals.max(0).values
                    rv=torch.sort(risk[bi,idx,ci]).values; risk_median[bi,ci]=rv[count//2] if count%2 else .5*(rv[count//2-1]+rv[count//2]); risk_max[bi,ci]=rv[-1]
        selected=torch.zeros_like(valid)
        for bi in range(risk.shape[0]):
            for si in range(risk.shape[1]):
                idx=torch.where(valid[bi,si])[0]
                if idx.numel():
                    k=max(1,math.ceil(.2*idx.numel())); chosen=idx[torch.topk(risk[bi,si,idx],k).indices]; selected[bi,si,chosen]=True
        recurrence=(selected&valid).float().sum(1)/safe_count
        out=self.proj(torch.cat([mean,std,median,maximum,recurrence[...,None]],-1))*observed_channel_mask[...,None]
        risk_mean=(risk*w).sum(1)/safe_count
        risk_var=(((risk-risk_mean[:,None])**2)*w).sum(1)/safe_count
        risk_std=torch.where(risk_var>0,torch.sqrt(risk_var.clamp_min(1e-8)),torch.zeros_like(risk_var))
        diagnostics={"risk_mean_across_seizures":risk_mean,"risk_std_across_seizures":risk_std,"risk_median_across_seizures":risk_median,"risk_max_across_seizures":risk_max,"top20_recurrence":recurrence,"valid_seizure_count_per_channel":raw_count,"observed_channel_mask":observed_channel_mask}
        return out,diagnostics
__all__=["CrossSeizureRecurrenceAggregator"]
