from __future__ import annotations

import math
import torch
from torch import nn


class ResidualSetEncoder(nn.Module):
    """Small shared-token, permutation-invariant residual set encoder."""
    def __init__(self, patient_embedding_dim:int, scalar_token_dim:int=13)->None:
        super().__init__();self.patient_projection=nn.Sequential(nn.LayerNorm(patient_embedding_dim),nn.Linear(patient_embedding_dim,8),nn.GELU());token_dim=8+int(scalar_token_dim)
        self.token_encoder=nn.Sequential(nn.LayerNorm(token_dim),nn.Linear(token_dim,16),nn.GELU(),nn.Dropout(.20));self.output_dim=16*3*4

    @staticmethod
    def _pool(tokens:torch.Tensor,mask:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
        rows=[]
        for p in range(tokens.shape[0]):
            local=tokens[p][mask[p]];local_weight=weight[p][mask[p]].clamp_min(0)
            if not local.numel():rows.append(tokens.new_zeros(48));continue
            weighted=(local*local_weight.unsqueeze(-1)).sum(0)/local_weight.sum().clamp_min(1e-8);maximum=local.max(0).values;k=min(local.shape[0],max(1,math.ceil(local.shape[0]*.10)));rank=local_weight if local_weight.numel() else local.norm(dim=-1);top=local[torch.topk(rank,k).indices].mean(0);rows.append(torch.cat((weighted,maximum,top)))
        return torch.stack(rows)

    def forward(self,patient_embedding:torch.Tensor,channel_scalars:torch.Tensor,target:torch.Tensor,channel_mask:torch.Tensor,reliable_abnormality:torch.Tensor)->dict[str,torch.Tensor]:
        projected=self.patient_projection(patient_embedding);tokens=self.token_encoder(torch.cat((projected,channel_scalars),dim=-1));valid=channel_mask.bool();target_mask=valid&target.bool();outside=valid&~target.bool()
        residual_mask=torch.zeros_like(valid)
        for p in range(tokens.shape[0]):
            idx=torch.where(outside[p])[0];k=min(idx.numel(),max(1,math.ceil(idx.numel()*.10)));chosen=idx[torch.topk(reliable_abnormality[p,idx],k).indices];residual_mask[p,chosen]=True
        # Risk weights also define the top-k ordering. Using constant weights
        # would make tied top-k selection depend on input channel order.
        h_target=self._pool(tokens,target_mask,reliable_abnormality);h_outside=self._pool(tokens,outside,reliable_abnormality);h_residual=self._pool(tokens,residual_mask,reliable_abnormality)
        contrast=h_target-h_residual;embedding=torch.cat((h_target,h_outside,h_residual,contrast),dim=-1)
        return {"H_target":h_target,"H_outside":h_outside,"H_residual":h_residual,"H_target_residual_contrast":contrast,"set_embedding":embedding,"residual_token_mask":residual_mask}


__all__=["ResidualSetEncoder"]
