from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .anchor import CleanNEZAnchor,masked_patient_channel_zscore,masked_seizure_channel_zscore
from .encoders import GatedEvidenceEncoder
from .recurrence import CrossSeizureRecurrenceAggregator

@dataclass
class CleanNEZConfig:
    model_dim:int=32; num_heads:int=2; dropout:float=.4; anchor_dim:int=16; early_pool_frac:float=.25; lse_pool_tau:float=1.

class CleanNEZModel(nn.Module):
    def __init__(self,config=CleanNEZConfig()):
        super().__init__(); self.config=config; d=config.model_dim
        self.encoder=GatedEvidenceEncoder(d,config.num_heads,config.dropout,config.early_pool_frac,config.lse_pool_tau)
        self.seizure_head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(d,1)); self.raw_seizure_early=nn.Parameter(torch.tensor(-3.))
        self.recurrence=CrossSeizureRecurrenceAggregator(d,config.dropout); pd=2*d
        self.attn=nn.MultiheadAttention(pd,config.num_heads,dropout=config.dropout,batch_first=True); self.norm=nn.LayerNorm(pd); self.classifier=nn.Sequential(nn.Linear(pd,pd),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(pd,1))
        self.anchor=CleanNEZAnchor(pd,config.anchor_dim,config.dropout); self.raw_anchor_weight=nn.Parameter(torch.tensor(-3.)); self.raw_early_weight=nn.Parameter(torch.tensor(-3.)); self.raw_recurrence_weight=nn.Parameter(torch.tensor(-3.))
    def forward(self,b):
        emb,early,gate,pool_diag=self.encoder(b["b0_features"],b["physics_features"],b["window_mask"],b["seizure_channel_mask"],b["window_centers"])
        base=masked_seizure_channel_zscore(self.seizure_head(emb).squeeze(-1),b["seizure_channel_mask"])
        szlogit=base-F.softplus(self.raw_seizure_early)*early; szp=torch.sigmoid(szlogit).masked_fill(~b["seizure_channel_mask"],0.); szrisk=(1-szp).masked_fill(~b["seizure_channel_mask"],0.)
        patient,diag=self.recurrence(emb,szrisk,b["seizure_mask"],b["seizure_channel_mask"]); effective=b["channel_mask"]&diag["observed_channel_mask"]
        h=masked_patient_channel_zscore(patient,effective); invalid=~effective; invalid=invalid.clone(); invalid[invalid.all(1)]=False; ctx,_=self.attn(h,h,h,key_padding_mask=invalid); h=self.norm(h+ctx)
        supervised=self.classifier(h).squeeze(-1); anc=self.anchor(patient,effective)
        n=diag["valid_seizure_count_per_channel"].clamp_min(1); patient_early=(early*(b["seizure_mask"][:,:,None]&b["seizure_channel_mask"])).sum(1)/n
        aa,ae,ar=map(F.softplus,(self.raw_anchor_weight,self.raw_early_weight,self.raw_recurrence_weight)); final=supervised-aa*anc["anchor_distance_z"]-ae*patient_early-ar*diag["top20_recurrence"]
        final=torch.nan_to_num(final,nan=0.,posinf=30.,neginf=-30.).masked_fill(~effective,-1e9); p=torch.sigmoid(final).masked_fill(~effective,0.)
        candidate=(1-p).masked_fill(~effective,0.); masked_diag={k:(v if k=="observed_channel_mask" else v.masked_fill(~effective,0.)) for k,v in diag.items()}
        fallback_count=pool_diag["post_onset_fallback_mask"].sum(1).masked_fill(~effective,0.); valid_count=diag["valid_seizure_count_per_channel"].clamp_min(1); fallback_rate=(fallback_count/valid_count).masked_fill(~effective,0.)
        return {"final_nez_logit":final,"score_nez_probability":p,"candidate_risk":candidate,"supervised_nez_logit":supervised.masked_fill(~effective,0.),"seizure_p_nez":szp,"seizure_candidate_risk":szrisk,"early_fast_score":patient_early.masked_fill(~effective,0.),"early_fast_gate_mean":gate.masked_fill(~effective,0.),"post_onset_fallback_count_per_channel":fallback_count,"post_onset_fallback_rate_per_channel":fallback_rate,"effective_channel_mask":effective,"observed_channel_mask":diag["observed_channel_mask"],**anc,**masked_diag,**pool_diag,"a_seizure_early":F.softplus(self.raw_seizure_early),"a_anchor":aa,"a_early":ae,"a_recurrence":ar}
__all__=["CleanNEZConfig","CleanNEZModel"]
