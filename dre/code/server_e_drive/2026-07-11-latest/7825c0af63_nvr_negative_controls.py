from __future__ import annotations

import torch


def _permute_valid(values:torch.Tensor,mask:torch.Tensor,generator:torch.Generator)->torch.Tensor:
    output=values.clone()
    for p in range(values.shape[0]):
        idx=torch.where(mask[p].bool())[0];order=idx[torch.randperm(idx.numel(),generator=generator)];output[p,idx]=values[p,order]
    return output


def target_permutation(target:torch.Tensor,channel_mask:torch.Tensor,seed:int=42)->torch.Tensor:return _permute_valid(target,channel_mask,torch.Generator().manual_seed(seed))


def p2_channel_permutation(mapping:dict[str,torch.Tensor],channel_mask:torch.Tensor,seed:int=42)->dict[str,torch.Tensor]:
    generator=torch.Generator().manual_seed(seed);output=dict(mapping)
    for key in ("q_final_nez","q_direct_nez","q_proto_nez","patient_channel_embedding","reliable_abnormality","persistent_abnormality","worstcase_abnormality"):
        if key in output:output[key]=_permute_valid(output[key],channel_mask,generator)
    return output


def graph_permutation(adjacency:torch.Tensor,channel_mask:torch.Tensor,seed:int=42)->torch.Tensor:
    generator=torch.Generator().manual_seed(seed);output=adjacency.clone()
    for p in range(adjacency.shape[0]):
        idx=torch.where(channel_mask[p].bool())[0];order=idx[torch.randperm(idx.numel(),generator=generator)];output[p,...,idx,:]=adjacency[p,...,order,:];output[p,...,:,idx]=output[p,...,:,order]
    return output


__all__=["graph_permutation","p2_channel_permutation","target_permutation"]
