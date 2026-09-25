from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RESIDUAL_RISK=("outside_residual_top10_mean","outside_residual_max","outside_persistent_top10","outside_worstcase_top10")
NETWORK_RISK=("residual_edge_mass_spread","persistent_residual_edge_mass_spread","residual_spectral_radius_ratio_spread","hub_miss_ratio_spread")
DIFFUSE_RISK=("global_abnormality_entropy_normalized","outside_abnormal_cluster_count","outside_largest_cluster_fraction")
SUPPORT=("inside_abnormality_mean","inside_abnormality_top10","inside_outside_reliable_gap","virtual_disruption_spread")
MONOTONE_FEATURES=RESIDUAL_RISK+NETWORK_RISK+DIFFUSE_RISK+SUPPORT


class MonotoneBasisFeature(nn.Module):
    def __init__(self,knots:tuple[float,...]=(-1.,0.,1.))->None:
        super().__init__();self.register_buffer("knots",torch.tensor(knots));self.raw_weights=nn.Parameter(torch.zeros(1+len(knots)))
    def forward(self,value:torch.Tensor)->torch.Tensor:
        weight=F.softplus(self.raw_weights);return weight[0]*value+(F.relu(value.unsqueeze(-1)-self.knots)*weight[1:]).sum(-1)


class MonotoneOutcomeHead(nn.Module):
    def __init__(self,feature_names:tuple[str,...]=MONOTONE_FEATURES,set_embedding_dim:int=0)->None:
        super().__init__();self.feature_names=tuple(feature_names);self.register_buffer("normalizer_mean",torch.zeros(len(feature_names)));self.register_buffer("normalizer_std",torch.ones(len(feature_names)));self.register_buffer("normalizer_fitted",torch.tensor(False));self.basis=nn.ModuleDict({name:MonotoneBasisFeature() for name in feature_names});self.bias=nn.Parameter(torch.zeros(()));self.set_head=nn.Sequential(nn.Linear(set_embedding_dim,16),nn.GELU(),nn.Linear(16,1)) if set_embedding_dim else None
    def fit_normalizer(self,values:torch.Tensor)->None:
        with torch.no_grad():self.normalizer_mean.copy_(values.mean(0));self.normalizer_std.copy_(values.std(0,unbiased=False).clamp_min(1e-6));self.normalizer_fitted.fill_(True)
    def forward(self,values:torch.Tensor,set_embedding:torch.Tensor|None=None)->dict[str,torch.Tensor]:
        x=(values-self.normalizer_mean)/self.normalizer_std;terms={name:self.basis[name](x[:,i]) for i,name in enumerate(self.feature_names)}
        residual=sum((terms[name] for name in RESIDUAL_RISK),start=x.new_zeros(x.shape[0]));network=sum((terms[name] for name in NETWORK_RISK),start=x.new_zeros(x.shape[0]));diffuse=sum((terms[name] for name in DIFFUSE_RISK),start=x.new_zeros(x.shape[0]));support=sum((terms[name] for name in SUPPORT),start=x.new_zeros(x.shape[0]));structured=self.bias+support-residual-network-diffuse
        delta=.149999*torch.tanh(self.set_head(set_embedding).squeeze(-1)) if self.set_head is not None else torch.zeros_like(structured);logit=structured+delta;probability=torch.sigmoid(logit)
        return {"outcome_logit_success":logit,"outcome_probability_success":probability,"outcome_probability_failure":1-probability,"structured_success_logit":structured,"set_delta":delta,"residual_risk_score":residual,"network_risk_score":network,"diffuse_risk_score":diffuse,"support_score":support}


__all__=["DIFFUSE_RISK","MONOTONE_FEATURES","MonotoneBasisFeature","MonotoneOutcomeHead","NETWORK_RISK","RESIDUAL_RISK","SUPPORT"]
