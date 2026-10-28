"""Reusable PR²/expanded-ABS adapter. Public API never receives clinical labels."""
import torch
from torch import nn
from reference import views

class PR2(nn.Module):
    def __init__(self,dimension,global_statistics,mode='pr2',calibration=True,seed=42):
        super().__init__();self.dimension=dimension;self.mode=mode
        for name in ['mean','sd','unit','epsilon']:
            value=global_statistics[name].detach().float().clone()
            assert value.shape==(dimension,)
            self.register_buffer('global_'+name,value)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.anchor=nn.Linear(dimension,1)
            self.ranking=nn.Sequential(nn.Linear(2*dimension,8),nn.GELU(),nn.Linear(8,1))
            self.calibration=nn.Linear(3,1)
        nn.init.zeros_(self.anchor.weight);nn.init.zeros_(self.anchor.bias)
        nn.init.zeros_(self.ranking[-1].weight);nn.init.zeros_(self.ranking[-1].bias)
        nn.init.zeros_(self.calibration.weight);nn.init.zeros_(self.calibration.bias)
        self.calibration_enabled=bool(calibration)
        self.calibration.requires_grad_(bool(calibration))
        self.ranking_parameters=16*dimension+17
        assert sum(p.numel() for p in self.ranking.parameters())==self.ranking_parameters
        assert sum(p.numel() for p in self.parameters())==17*dimension+22

    def forward(self,h,base_ez_logit,groups=None,valid=None,ranking=True,calibration=True):
        assert h.ndim==2 and h.shape[1]==self.dimension and base_ez_logit.shape==(len(h),)
        groups=torch.zeros(len(h),device=h.device,dtype=torch.long) if groups is None else groups
        valid=torch.ones(len(h),device=h.device,dtype=torch.bool) if valid is None else valid.bool()
        assert groups.shape==valid.shape==(len(h),)
        out=dict(anchor=base_ez_logit.clone(),delta_rank=torch.zeros_like(base_ez_logit),
                 delta_cal=torch.zeros_like(base_ez_logit),rank=base_ez_logit.clone(),
                 final=base_ez_logit.clone(),diagnostics=[])
        for p in torch.unique(groups[valid]):
            ix=torch.nonzero((groups==p)&valid).flatten();x=h[ix];base=base_ez_logit[ix]
            absolute,v,diagnostic=views(x,self.global_mean,self.global_sd,self.global_unit,self.global_epsilon,self.mode)
            anchor=base+self.anchor(absolute).flatten()
            u=self.ranking(v).flatten();dr=diagnostic['alpha']*(u-u.mean()) if ranking else torch.zeros_like(u)
            rank=anchor+dr
            statistics=torch.stack([rank.mean(),rank.std(unbiased=False),rank.new_tensor(len(ix)).log1p()])
            dc=self.calibration(statistics).squeeze() if calibration and self.calibration_enabled else rank.new_zeros(())
            out['anchor'][ix]=anchor;out['delta_rank'][ix]=dr;out['delta_cal'][ix]=dc
            out['rank'][ix]=rank;out['final'][ix]=rank+dc
            out['diagnostics'].append(diagnostic)
        assert all(torch.isfinite(out[k][valid]).all() for k in ['anchor','delta_rank','delta_cal','rank','final'])
        return out
