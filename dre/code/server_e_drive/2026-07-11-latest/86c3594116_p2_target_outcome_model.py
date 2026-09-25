from __future__ import annotations

import torch
from torch import nn

from .profiles import get_profile
from .seizure_target_probe import CROSS_SEIZURE_FEATURES, SeizureTargetProbe, compute_cross_seizure_features
from .target_concordance import ABNORMALITY_FEATURES, CONCORDANCE_FEATURES, TARGET_FEATURES, compute_target_concordance, stack_features
from .target_network import NETWORK_FEATURES, compute_target_network
from .target_pooling import compute_target_pooling


FINAL_LOGIT_FEATURES=("final_nez_logit_mean","final_nez_logit_std","final_nez_logit_min","final_nez_logit_max","temporal_delta_norm_mean","delta_onset_norm_mean","delta_spread_norm_mean")


class P2TargetOutcomeModel(nn.Module):
    """Clinical-target-conditioned P(success) head. Center/outcome metadata are never inputs."""
    def __init__(self, embedding_dim:int, profile:str, *, seizure_embedding_dim:int|None=None)->None:
        super().__init__(); self.profile=get_profile(profile); self.embedding_dim=int(embedding_dim); seizure_dim=int(seizure_embedding_dim or embedding_dim)
        self.channel_projection=nn.Sequential(nn.LayerNorm(embedding_dim),nn.Linear(embedding_dim,16),nn.GELU(),nn.Dropout(.10))
        self.seizure_target_probe=SeizureTargetProbe(seizure_dim) if self.profile.cross_seizure else None
        scalar_names=[]
        if self.profile.p2_signal: scalar_names.extend(ABNORMALITY_FEATURES+FINAL_LOGIT_FEATURES)
        if self.profile.clinical_target: scalar_names.extend(TARGET_FEATURES)
        if self.profile.clinical_target and self.profile.p2_signal: scalar_names.extend(CONCORDANCE_FEATURES)
        if self.profile.cross_seizure: scalar_names.extend(CROSS_SEIZURE_FEATURES)
        if self.profile.target_network: scalar_names.extend(NETWORK_FEATURES+("graph_coverage",))
        self.scalar_names=tuple(scalar_names); n=max(1,len(self.scalar_names))
        self.register_buffer("scalar_mean",torch.zeros(n)); self.register_buffer("scalar_std",torch.ones(n)); self.register_buffer("scalar_normalizer_fitted",torch.tensor(False))
        self.scalar_encoder=nn.Sequential(nn.LayerNorm(n),nn.Linear(n,32),nn.GELU(),nn.Dropout(.20))
        if self.profile.p2_signal:
            pool_count=2 if not self.profile.clinical_target else 8
            self.embedding_encoder=nn.Sequential(nn.LayerNorm(pool_count*16),nn.Linear(pool_count*16,64),nn.GELU(),nn.Dropout(.20),nn.Linear(64,32),nn.GELU())
        else: self.embedding_encoder=None
        self.outcome_head=nn.Sequential(nn.LayerNorm(64 if self.profile.p2_signal else 32),nn.Linear(64 if self.profile.p2_signal else 32,32),nn.GELU(),nn.Dropout(.20),nn.Linear(32,8),nn.GELU(),nn.Linear(8,1))

    def fit_scalar_normalizer(self, values:torch.Tensor)->None:
        if values.ndim!=2 or values.shape[1]!=len(self.scalar_mean): raise ValueError("scalar normalizer shape mismatch")
        with torch.no_grad():
            self.scalar_mean.copy_(values.mean(0)); self.scalar_std.copy_(values.std(0,unbiased=False).clamp_min(1e-6)); self.scalar_normalizer_fitted.fill_(True)

    @staticmethod
    def _masked_stats(values:torch.Tensor,mask:torch.Tensor,prefix:str)->dict[str,torch.Tensor]:
        output={f"{prefix}_{name}":[] for name in ("mean","std","min","max")}
        for p in range(values.shape[0]):
            v=values[p][mask[p].bool()];
            for name,x in (("mean",v.mean()),("std",v.std(unbiased=False)),("min",v.min()),("max",v.max())): output[f"{prefix}_{name}"].append(x)
        return {name:torch.stack(rows) for name,rows in output.items()}

    def forward(self,p2:dict[str,torch.Tensor],graphs:dict[str,torch.Tensor]|None=None,clinical_target_mask:torch.Tensor|None=None)->dict[str,torch.Tensor]:
        mask=p2["channel_mask"].bool(); target=clinical_target_mask
        if self.profile.clinical_target and target is None: raise KeyError(f"{self.profile.name} requires clinical_target_mask")
        abnormality=p2["final_score_ez"] if self.profile.p2_signal else torch.zeros_like(target,dtype=p2["patient_channel_embedding"].dtype)
        features:dict[str,torch.Tensor]={}
        if self.profile.p2_signal:
            dummy=target if target is not None else torch.zeros_like(abnormality)
            concord=compute_target_concordance(abnormality,dummy,mask); features.update({name:concord[name] for name in ABNORMALITY_FEATURES})
            features.update(self._masked_stats(p2["final_nez_logit"],mask,"final_nez_logit"))
            for name in ("temporal_delta_norm","delta_onset_norm","delta_spread_norm"):
                features[f"{name}_mean"]=self._masked_stats(p2[name],mask,name)[f"{name}_mean"]
        if self.profile.clinical_target:
            concord=compute_target_concordance(abnormality,target,mask)
            features.update({name:concord[name] for name in TARGET_FEATURES})
            if self.profile.p2_signal: features.update({name:concord[name] for name in CONCORDANCE_FEATURES})
        elif self.profile.p2_signal and target is not None:
            # Audit-only concordance for C1; these fields are deliberately absent
            # from scalar_names and therefore cannot affect the C1 prediction.
            audit_concord=compute_target_concordance(abnormality,target,mask)
            features.update({name:audit_concord[name] for name in TARGET_FEATURES+CONCORDANCE_FEATURES})
        cross={}; probe={}
        if self.seizure_target_probe is not None:
            valid=p2["seizure_channel_mask"].bool()&p2["seizure_mask"].bool().unsqueeze(-1)
            probe=self.seizure_target_probe(p2["seizure_channel_embedding"],valid)
            cross=compute_cross_seizure_features(probe["seizure_target_probability"],target,valid); features.update({name:cross[name] for name in CROSS_SEIZURE_FEATURES})
            probe["seizure_target_valid"]=valid
        network={}
        if self.profile.target_network:
            if graphs is None: raise KeyError(f"{self.profile.name} requires three-phase raw graphs")
            network=compute_target_network(graphs["adjacency"],abnormality,target,graphs["phase_channel_mask"],graphs["graph_valid"]); features.update(network)
        scalar=stack_features(features,self.scalar_names) if self.scalar_names else abnormality.new_zeros((abnormality.shape[0],1))
        if self.training and self.profile.clinical_target and self.profile.p2_signal:
            drop=(torch.rand((scalar.shape[0],1),device=scalar.device)<.20)
            for name in TARGET_FEATURES:
                if name in self.scalar_names: scalar[:,self.scalar_names.index(name)]=torch.where(drop[:,0],torch.zeros_like(scalar[:,self.scalar_names.index(name)]),scalar[:,self.scalar_names.index(name)])
        normalized=(scalar-self.scalar_mean)/self.scalar_std if bool(self.scalar_normalizer_fitted) else scalar
        pieces=[self.scalar_encoder(normalized)]
        pools={}
        if self.profile.p2_signal:
            projected=self.channel_projection(p2["patient_channel_embedding"])
            dummy=target if target is not None else torch.zeros_like(abnormality); pools=compute_target_pooling(projected,abnormality,dummy,mask)
            names=("global","global_std") if not self.profile.clinical_target else ("global","target","non_target","target_contrast","covered","residual","covered_residual_contrast","top10")
            pieces.append(self.embedding_encoder(torch.cat([pools[name] for name in names],dim=-1)))
        patient_embedding=torch.cat(pieces,dim=-1); logit=self.outcome_head(patient_embedding).squeeze(-1); probability=torch.sigmoid(logit)
        old_values=p2["seizure_nez_probability"]; old_valid=p2["seizure_channel_mask"].bool()&p2["seizure_mask"].bool().unsqueeze(-1)
        old_std=[]; old_min=[]; old_max=[]
        for patient_idx in range(old_values.shape[0]):
            local=old_values[patient_idx][old_valid[patient_idx]]; old_std.append(local.std(unbiased=False)); old_min.append(local.min()); old_max.append(local.max())
        return {"outcome_logit_success":logit,"outcome_probability_success":probability,"outcome_probability_failure":1-probability,"patient_outcome_embedding":patient_embedding,
                "patient_scalar_features":scalar,"patient_scalar_feature_names":self.scalar_names,"predicted_abnormality_score":abnormality,
                "old_q10_std":torch.stack(old_std),"old_q10_min":torch.stack(old_min),"old_q10_max":torch.stack(old_max),**features,**pools,**probe,**cross,**network}


__all__=["FINAL_LOGIT_FEATURES","P2TargetOutcomeModel"]
