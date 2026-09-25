"""SCOPE-v2 additions layered on the unchanged P2-Q10 objective."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

from .p2_scope_v2_decoder import beta_binomial_log_prob

def _zero(x): return x.sum() * 0.0
def _ramp(epoch: int) -> float: return min(max((int(epoch)-3)/5.0, 0.0), 1.0)

def patient_wise_smooth_ap_loss(logits, labels_ez, mask, tau=.10):
    rows=[]
    for i in range(logits.shape[0]):
        valid=mask[i] & (labels_ez[i]>=0); pos=valid & (labels_ez[i]>.5); neg=valid & ~pos
        if not torch.any(pos) or not torch.any(neg): continue
        s=-logits[i,valid]; p=labels_ez[i,valid]>.5; diff=((s[None,:]-s[:,None])/tau).clamp(-20,20); eye=torch.eye(s.numel(),device=s.device,dtype=torch.bool)
        ranks=1+torch.sigmoid(diff).masked_fill(eye,0).sum(1); posr=1+(torch.sigmoid(diff)*p[None,:]).masked_fill(eye,0).sum(1)
        rows.append(1-(posr[p]/ranks[p]).mean())
    return torch.stack(rows).mean() if rows else _zero(logits)

def scope_losses(outputs, batch, epoch, args):
    logits=outputs['scope_nez_logit']; labels_ez=batch['labels_ez']; mask=batch['channel_mask'].bool(); zero=_zero(logits); boundary=[]; total_pairs=0; selected_pairs=0; selected_mean=[]; selected_max=[]
    for i in range(logits.shape[0]):
        valid=mask[i] & (labels_ez[i]>=0); n=int(valid.sum()); k=int((labels_ez[i,valid]>.5).sum())
        if n < 2 or not (0 < k < n): continue
        order=torch.argsort(outputs['direct_nez_logit'][i,valid].detach()); indices=torch.nonzero(valid,as_tuple=False).squeeze(1)[order]; band=max(2, math.ceil(float(args.scope_boundary_band_fraction)*n)); chosen=indices[max(0,k-band):min(n,k+band)]
        pos=chosen[labels_ez[i,chosen]>.5]; neg=chosen[labels_ez[i,chosen]<=.5]
        if pos.numel() and neg.numel():
            # The boundary band is selected by detached direct logits, but the
            # ranking gradient is applied only to the SCOPE score.
            pair_loss=F.softplus(float(args.scope_boundary_margin)+(-logits[i,neg])[:,None]-(-logits[i,pos])[None,:]).flatten()
            total_pairs += int(pair_loss.numel())
            chosen = torch.topk(pair_loss, k=min(int(args.scope_max_pairs_per_patient), int(pair_loss.numel())), largest=True).values
            boundary.append(chosen.mean()); selected_pairs += int(chosen.numel()); selected_mean.append(float(chosen.detach().mean())); selected_max.append(float(chosen.detach().max()))
    b_loss=torch.stack(boundary).mean() if boundary else zero
    count_nll=[]; frac=[]
    for i in range(logits.shape[0]):
        valid=mask[i] & (labels_ez[i]>=0); n=valid.sum().to(logits.dtype); k=(labels_ez[i,valid]>.5).sum().to(logits.dtype)
        if n < 1: continue
        a=outputs['scope_count_alpha'][i]; b=outputs['scope_count_beta'][i]
        logp=beta_binomial_log_prob(k, n, a, b)
        count_nll.append(-logp/(torch.log(n+1)+1)); frac.append(F.smooth_l1_loss(outputs['scope_count_predicted_mu'][i], k/n))
    progress=_ramp(epoch); smooth=patient_wise_smooth_ap_loss(logits,labels_ez,mask,float(args.scope_smoothap_tau)); nll=torch.stack(count_nll).mean() if count_nll else zero; fraction=torch.stack(frac).mean() if frac else zero
    total=progress*(float(args.scope_smoothap_weight)*smooth+float(args.scope_boundary_weight)*b_loss+float(args.scope_count_nll_weight)*nll+float(args.scope_count_fraction_weight)*fraction)+.005*(outputs['scope_boundary_delta'][mask].square().mean() if torch.any(mask) else zero)
    return total, {'scope_smoothap_loss':float(smooth.detach()),'scope_smoothap_active_weight':float(progress*args.scope_smoothap_weight),'scope_boundary_loss':float(b_loss.detach()),'scope_boundary_active_weight':float(progress*args.scope_boundary_weight),'scope_boundary_valid_patient_count':float(len(boundary)),'scope_boundary_total_pair_count':float(total_pairs),'scope_boundary_selected_pair_count':float(selected_pairs),'scope_boundary_mean_selected_loss':float(np.mean(selected_mean)) if selected_mean else 0.,'scope_boundary_max_selected_loss':float(np.max(selected_max)) if selected_max else 0.,'scope_count_nll':float(nll.detach()),'scope_count_nll_active_weight':float(progress*args.scope_count_nll_weight),'scope_fraction_loss':float(fraction.detach()),'scope_fraction_active_weight':float(progress*args.scope_count_fraction_weight),'scope_boundary_delta_mean_abs':float(outputs['scope_boundary_delta'][mask].abs().mean().detach()) if torch.any(mask) else 0.,'scope_boundary_delta_max_abs':float(outputs['scope_boundary_delta'][mask].abs().max().detach()) if torch.any(mask) else 0.}
