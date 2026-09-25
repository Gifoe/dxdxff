from __future__ import annotations

import math
import numpy as np
import torch


DISPERSION_FEATURES=("global_abnormality_entropy_normalized","global_abnormality_gini","global_top10_mass_fraction","global_top20_mass_fraction","outside_abnormality_entropy_normalized","outside_abnormal_cluster_count","outside_largest_cluster_fraction")


def _entropy(value:torch.Tensor,eps:float)->torch.Tensor:
    if value.numel()<=1:return value.new_tensor(0.)
    probability=value.clamp_min(0)/value.clamp_min(0).sum().clamp_min(eps);return -(probability*probability.clamp_min(eps).log()).sum()/math.log(value.numel())


def _gini(value:torch.Tensor,eps:float)->torch.Tensor:
    value=value.clamp_min(0).sort().values;n=value.numel()
    if n<=1 or value.sum()<=eps:return value.new_tensor(0.)
    index=torch.arange(1,n+1,device=value.device,dtype=value.dtype);return ((2*index-n-1)*value).sum()/(n*value.sum().clamp_min(eps))


def _top_mass(value:torch.Tensor,fraction:float,eps:float)->torch.Tensor:
    k=min(value.numel(),max(1,math.ceil(value.numel()*fraction)));return torch.topk(value,k).values.sum()/value.sum().clamp_min(eps)


def _clusters(matrix:torch.Tensor,active:torch.Tensor)->tuple[int,float]:
    indices=torch.where(active)[0]
    if not indices.numel():return 0,0.
    binary=(matrix[indices][:,indices]>0).cpu().numpy();seen=set();sizes=[]
    for start in range(len(indices)):
        if start in seen:continue
        stack=[start];seen.add(start);size=0
        while stack:
            node=stack.pop();size+=1
            for nxt in np.where(binary[node])[0].tolist():
                if nxt not in seen:seen.add(nxt);stack.append(nxt)
        sizes.append(size)
    return len(sizes),max(sizes)/len(indices)


def compute_abnormality_dispersion(reliable_abnormality:torch.Tensor,target:torch.Tensor,channel_mask:torch.Tensor,adjacency:torch.Tensor|None=None,graph_valid:torch.Tensor|None=None,eps:float=1e-8)->dict[str,torch.Tensor]:
    rows={name:[] for name in DISPERSION_FEATURES}
    for p in range(reliable_abnormality.shape[0]):
        valid=channel_mask[p].bool();outside=valid&~target[p].bool();global_value=reliable_abnormality[p][valid];outside_value=reliable_abnormality[p][outside]
        cluster_count=0.;largest=0.
        if adjacency is not None:
            graphs=[]
            for s in range(adjacency.shape[1]):
                for phase in range(adjacency.shape[2]):
                    if graph_valid is None or bool(graph_valid[p,s,phase]):graphs.append(adjacency[p,s,phase])
            if graphs:
                mean_graph=torch.stack(graphs).mean(0);threshold=torch.quantile(outside_value,.90);active=outside&(reliable_abnormality[p]>=threshold);cluster_count,largest=_clusters(mean_graph,active)
        values={"global_abnormality_entropy_normalized":_entropy(global_value,eps),"global_abnormality_gini":_gini(global_value,eps),"global_top10_mass_fraction":_top_mass(global_value,.10,eps),"global_top20_mass_fraction":_top_mass(global_value,.20,eps),"outside_abnormality_entropy_normalized":_entropy(outside_value,eps),"outside_abnormal_cluster_count":global_value.new_tensor(cluster_count),"outside_largest_cluster_fraction":global_value.new_tensor(largest)}
        for name in DISPERSION_FEATURES:rows[name].append(values[name])
    return {name:torch.stack(value) for name,value in rows.items()}


__all__=["DISPERSION_FEATURES","compute_abnormality_dispersion"]
