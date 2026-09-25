from __future__ import annotations

import torch
from torch.nn import functional as F


def center_balanced_outcome_loss(logit:torch.Tensor,target_success:torch.Tensor,centers:list[str]|tuple[str,...])->dict[str,torch.Tensor]:
    per_patient=F.binary_cross_entropy_with_logits(logit,target_success,reduction="none");patient_loss=per_patient.mean();unique=sorted(set(map(str,centers)));center_loss=torch.stack([per_patient[torch.tensor([str(value)==center for value in centers],device=logit.device)].mean() for center in unique]).mean();outcome=.5*patient_loss+.5*center_loss;return {"patient_loss":patient_loss,"center_loss":center_loss,"outcome_loss":outcome}


def nvr_loss(logit:torch.Tensor,target_success:torch.Tensor,centers:list[str]|tuple[str,...],parameters,*,dropped_logit:torch.Tensor|None=None,plus_logit:torch.Tensor|None=None,minus_logit:torch.Tensor|None=None,counterfactual_valid:torch.Tensor|None=None,robust:bool=False)->dict[str,torch.Tensor]:
    parts=center_balanced_outcome_loss(logit,target_success,centers);l2=sum((value.square().mean() for value in parameters if value.requires_grad),start=logit.new_zeros(()));drop=F.mse_loss(dropped_logit,logit.detach()) if robust and dropped_logit is not None else logit.new_zeros(());cf=logit.new_zeros(())
    if robust and plus_logit is not None and minus_logit is not None:
        valid=counterfactual_valid.bool() if counterfactual_valid is not None else torch.ones_like(logit,dtype=torch.bool);terms=F.relu(logit-plus_logit+.01)+F.relu(minus_logit-logit+.01);cf=terms[valid].mean() if valid.any() else logit.new_zeros(())
    total=parts["outcome_loss"]+1e-4*l2+(.05*drop+.03*cf if robust else 0);return {"loss":total,"l2":l2,"channel_drop_consistency":drop,"counterfactual_consistency":cf,**parts}


__all__=["center_balanced_outcome_loss","nvr_loss"]
