from __future__ import annotations

import torch

from .functional_graph import NETWORK_PHASES


BASE_NETWORK_FEATURES=("abnormal_network","covered_abnormal_network","residual_abnormal_network","bridge_abnormal_network","abnormal_hub_coverage","residual_hub_burden","network_coverage_ratio","network_residual_ratio")
NETWORK_FEATURES=tuple(f"{name}_{phase}" for phase in NETWORK_PHASES for name in BASE_NETWORK_FEATURES)+tuple(f"{name}_{delta}" for delta in ("onset_minus_preictal","spread_minus_onset","spread_minus_preictal") for name in BASE_NETWORK_FEATURES)


def compute_target_network(adjacency: torch.Tensor, abnormality: torch.Tensor, target: torch.Tensor, phase_channel_mask: torch.Tensor, graph_valid: torch.Tensor, eps: float=1e-8)->dict[str,torch.Tensor]:
    values={name:[] for name in NETWORK_FEATURES}; coverage=[]
    for p in range(adjacency.shape[0]):
        phase_values={name:[] for name in BASE_NETWORK_FEATURES}; valid_graphs=0; total_graphs=0
        for phase_idx,phase in enumerate(NETWORK_PHASES):
            per={name:[] for name in BASE_NETWORK_FEATURES}
            for s in range(adjacency.shape[1]):
                total_graphs+=1
                if not bool(graph_valid[p,s,phase_idx]): continue
                mask=phase_channel_mask[p,s,phase_idx].bool(); idx=torch.where(mask)[0]
                if idx.numel()<2: continue
                valid_graphs+=1; g=adjacency[p,s,phase_idx][idx][:,idx]; a=abnormality[p,idx]; t=target[p,idx].to(a.dtype)
                tri=torch.triu(torch.ones_like(g,dtype=torch.bool),diagonal=1); w=g[tri]; ai=a[:,None].expand_as(g)[tri]; aj=a[None,:].expand_as(g)[tri]; ti=t[:,None].expand_as(g)[tri]; tj=t[None,:].expand_as(g)[tri]
                denom=w.sum().clamp_min(eps); base=w*ai*aj; abnormal=base.sum()/denom
                covered=(base*ti*tj).sum()/denom; residual=(base*(1-ti)*(1-tj)).sum()/denom; bridge=(base*(ti*(1-tj)+(1-ti)*tj)).sum()/denom
                strength=g.sum(dim=-1); hubden=(strength*a).sum().clamp_min(eps)
                local=(abnormal,covered,residual,bridge,(strength*a*t).sum()/hubden,(strength*a*(1-t)).sum()/hubden,covered/abnormal.clamp_min(eps),residual/abnormal.clamp_min(eps))
                for name,value in zip(BASE_NETWORK_FEATURES,local): per[name].append(value)
            for name in BASE_NETWORK_FEATURES: phase_values[name].append(torch.stack(per[name]).mean() if per[name] else adjacency.new_tensor(0.0))
        for phase_idx,phase in enumerate(NETWORK_PHASES):
            for name in BASE_NETWORK_FEATURES: values[f"{name}_{phase}"].append(phase_values[name][phase_idx])
        for delta,left,right in (("onset_minus_preictal",1,0),("spread_minus_onset",2,1),("spread_minus_preictal",2,0)):
            for name in BASE_NETWORK_FEATURES: values[f"{name}_{delta}"].append(phase_values[name][left]-phase_values[name][right])
        coverage.append(adjacency.new_tensor(valid_graphs/max(total_graphs,1)))
    output={name:torch.stack(items) for name,items in values.items()}; output["graph_coverage"]=torch.stack(coverage); return output


__all__=["BASE_NETWORK_FEATURES","NETWORK_FEATURES","compute_target_network"]
