from __future__ import annotations

import math
import numpy as np
import torch

from .functional_graph import NETWORK_PHASES


BASE_VIRTUAL_FEATURES=("residual_edge_mass","persistent_residual_edge_mass","residual_bridge_mass","residual_hub_mass","original_spectral_radius","residual_spectral_radius","residual_spectral_radius_ratio","original_global_efficiency","residual_global_efficiency","residual_global_efficiency_ratio","residual_largest_component_fraction","residual_density","residual_strength_mean","residual_strength_top10","virtual_disruption","hub_miss_ratio")
VIRTUAL_NETWORK_FEATURES=tuple(f"{name}_{phase}" for phase in NETWORK_PHASES for name in BASE_VIRTUAL_FEATURES)+tuple(f"{name}_{delta}" for delta in ("onset_minus_preictal","spread_minus_onset","spread_minus_preictal") for name in BASE_VIRTUAL_FEATURES)


def virtual_residual_graph(adjacency: torch.Tensor,target:torch.Tensor)->torch.Tensor:
    outside=(1-target.to(adjacency.dtype)); return adjacency*outside.unsqueeze(-1)*outside.unsqueeze(-2)


def _spectral_radius(matrix:torch.Tensor)->torch.Tensor:
    if matrix.numel()==0:return matrix.new_tensor(0.)
    return torch.linalg.eigvalsh((matrix+matrix.T)/2).abs().max()


def _efficiency(matrix:torch.Tensor)->torch.Tensor:
    n=matrix.shape[0]
    if n<2:return matrix.new_tensor(0.)
    dist=torch.full_like(matrix,float("inf")); positive=matrix>0; dist[positive]=1/matrix[positive].clamp_min(1e-8); dist.fill_diagonal_(0)
    for k in range(n): dist=torch.minimum(dist,dist[:,k:k+1]+dist[k:k+1,:])
    off=~torch.eye(n,dtype=torch.bool,device=matrix.device); finite=off&torch.isfinite(dist)&(dist>0); return torch.where(finite,1/dist.clamp_min(1e-8),torch.zeros_like(dist))[off].mean()


def _largest_component_fraction(matrix:torch.Tensor)->torch.Tensor:
    n=matrix.shape[0]
    if n==0:return matrix.new_tensor(0.)
    seen=set(); largest=0; binary=(matrix>0).cpu().numpy()
    for start in range(n):
        if start in seen:continue
        stack=[start]; seen.add(start); size=0
        while stack:
            node=stack.pop(); size+=1
            for nxt in np.where(binary[node])[0].tolist():
                if nxt not in seen:seen.add(nxt);stack.append(nxt)
        largest=max(largest,size)
    return matrix.new_tensor(largest/max(n,1))


def _single_graph(g:torch.Tensor,t:torch.Tensor,r:torch.Tensor,persistent:torch.Tensor,eps:float)->dict[str,torch.Tensor]:
    t=t.to(g.dtype); n=g.shape[0]; outside=1-t; residual=r*outside; pres=persistent*outside; tri=torch.triu(torch.ones_like(g,dtype=torch.bool),1); edge=g[tri]; denom=edge.sum().clamp_min(eps)
    ri=residual[:,None].expand_as(g)[tri]; rj=residual[None,:].expand_as(g)[tri]; pi=pres[:,None].expand_as(g)[tri]; pj=pres[None,:].expand_as(g)[tri]; ti=t[:,None].expand_as(g)[tri]; tj=t[None,:].expand_as(g)[tri]
    minus=virtual_residual_graph(g,t); outside_idx=torch.where(~t.bool())[0]; minus_out=minus[outside_idx][:,outside_idx]; strength=g.sum(-1); residual_strength=minus.sum(-1); original_radius=_spectral_radius(g); residual_radius=_spectral_radius(minus_out); original_eff=_efficiency(g); residual_eff=_efficiency(minus_out)
    k=max(1,math.ceil(n*.10)); hubs=torch.topk(strength,k).indices; high=residual>=torch.quantile(residual,.90)
    ai=r[:,None].expand_as(g)[tri]; aj=r[None,:].expand_as(g)[tri]; bridge_weight=ri*aj+ai*rj
    out_strength=residual_strength[outside_idx] if outside_idx.numel() else residual_strength.new_zeros(1); k_out=min(out_strength.numel(),max(1,math.ceil(out_strength.numel()*.10)))
    return {"residual_edge_mass":(edge*ri*rj).sum()/denom,"persistent_residual_edge_mass":(edge*pi*pj).sum()/denom,"residual_bridge_mass":(edge*bridge_weight*(ti*(1-tj)+(1-ti)*tj)).sum()/denom,"residual_hub_mass":(strength*residual).sum()/strength.sum().clamp_min(eps),"original_spectral_radius":original_radius,"residual_spectral_radius":residual_radius,"residual_spectral_radius_ratio":residual_radius/original_radius.clamp_min(eps),"original_global_efficiency":original_eff,"residual_global_efficiency":residual_eff,"residual_global_efficiency_ratio":residual_eff/original_eff.clamp_min(eps),"residual_largest_component_fraction":_largest_component_fraction(minus_out),"residual_density":((minus_out>0)&~torch.eye(minus_out.shape[0],dtype=torch.bool,device=g.device)).float().sum()/max(minus_out.shape[0]*(minus_out.shape[0]-1),1),"residual_strength_mean":out_strength.mean(),"residual_strength_top10":torch.topk(out_strength,k_out).values.mean(),"virtual_disruption":1-residual_radius/original_radius.clamp_min(eps),"hub_miss_ratio":((~t[hubs].bool())&high[hubs]).float().mean()}


def compute_virtual_target_network(adjacency:torch.Tensor,target:torch.Tensor,reliable_abnormality:torch.Tensor,persistent_abnormality:torch.Tensor,phase_channel_mask:torch.Tensor,graph_valid:torch.Tensor,eps:float=1e-8)->dict[str,torch.Tensor]:
    rows={name:[] for name in VIRTUAL_NETWORK_FEATURES}; audits=[]
    for p in range(adjacency.shape[0]):
        phase_values={name:[] for name in BASE_VIRTUAL_FEATURES}; valid_count=0
        for phase_idx,phase in enumerate(NETWORK_PHASES):
            local={name:[] for name in BASE_VIRTUAL_FEATURES}
            for s in range(adjacency.shape[1]):
                mask=phase_channel_mask[p,s,phase_idx].bool()
                if not bool(graph_valid[p,s,phase_idx]) or mask.sum()<2:continue
                idx=torch.where(mask)[0]; values=_single_graph(adjacency[p,s,phase_idx][idx][:,idx],target[p,idx],reliable_abnormality[p,idx],persistent_abnormality[p,idx],eps);valid_count+=1
                for name in BASE_VIRTUAL_FEATURES:local[name].append(values[name])
            for name in BASE_VIRTUAL_FEATURES:phase_values[name].append(torch.stack(local[name]).mean() if local[name] else adjacency.new_tensor(0.))
        patient={}
        for i,phase in enumerate(NETWORK_PHASES):
            for name in BASE_VIRTUAL_FEATURES:patient[f"{name}_{phase}"]=phase_values[name][i]
        for delta,left,right in (("onset_minus_preictal",1,0),("spread_minus_onset",2,1),("spread_minus_preictal",2,0)):
            for name in BASE_VIRTUAL_FEATURES:patient[f"{name}_{delta}"]=phase_values[name][left]-phase_values[name][right]
        for name in VIRTUAL_NETWORK_FEATURES:rows[name].append(patient[name])
        audits.append({"n_valid_graphs":valid_count,**{name:float(value) for name,value in patient.items()}})
    output={name:torch.stack(value) for name,value in rows.items()};output["virtual_target_network_audit"]=audits;return output


__all__=["BASE_VIRTUAL_FEATURES","VIRTUAL_NETWORK_FEATURES","compute_virtual_target_network","virtual_residual_graph"]
