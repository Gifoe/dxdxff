from __future__ import annotations

import torch
from torch import nn

from graph_spectral_encoder import WindowGraphSpectralEncoder
from .temporal import PostOnsetTemporalEncoder


class EarlyFastDynamicsEncoder(nn.Module):
    """Small early-fast state residual; this is not a neural-fragility model."""
    def __init__(self, model_dim=32, num_heads=2, dropout=.4):
        super().__init__()
        self.state = nn.Sequential(nn.LazyLinear(model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.source = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.damping = nn.Parameter(torch.zeros(model_dim))
        self.out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim))

    def forward(self, x, window_mask, seizure_channel_mask):
        h = self.state(x); b,s,t,c,d = h.shape
        flat = h.reshape(b*s*t,c,d)
        invalid = (~seizure_channel_mask[:,:,None,:].expand(b,s,t,c)).reshape(b*s*t,c)
        invalid = invalid.clone(); invalid[invalid.all(1)] = False
        context,_ = self.attn(flat,flat,flat,key_padding_mask=invalid)
        context = context.reshape_as(h)
        valid = (window_mask[:,:,:,None] & seizure_channel_mask[:,:,None,:]).unsqueeze(-1)
        velocity = self.source(h) + context - torch.nn.functional.softplus(self.damping)*h
        return self.out(velocity) * valid


class GatedEvidenceEncoder(nn.Module):
    def __init__(self, model_dim=32, num_heads=2, dropout=.4, early_pool_frac=.25, lse_pool_tau=1.):
        super().__init__()
        self.b0 = WindowGraphSpectralEncoder(model_dim=model_dim,num_heads=num_heads,dropout=dropout)
        self.dyn = EarlyFastDynamicsEncoder(model_dim,num_heads,dropout)
        self.gate = nn.Linear(2*model_dim, model_dim)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -3.)
        self.norm = nn.LayerNorm(model_dim)
        self.pool = PostOnsetTemporalEncoder(model_dim=model_dim,early_pool_frac=early_pool_frac,lse_pool_tau=lse_pool_tau)
        self.early_head = nn.Sequential(nn.Linear(model_dim,model_dim//2),nn.GELU(),nn.Linear(model_dim//2,1))

    def forward(self,b0,physics,window_mask,seizure_channel_mask,window_centers):
        b0=torch.nan_to_num(b0); physics=torch.nan_to_num(physics)
        hb = self.b0(b0,seizure_channel_mask=seizure_channel_mask)
        hd = self.dyn(physics,window_mask,seizure_channel_mask)
        gate = torch.sigmoid(self.gate(torch.cat([hb,hd],-1)))
        fused = torch.nan_to_num(self.norm(hb + gate*hd))
        emb,_,pool_diag = self.pool(fused,seizure_channel_mask,window_mask,window_centers)
        score = torch.sigmoid(self.early_head(emb).squeeze(-1)).masked_fill(~seizure_channel_mask,0.)
        return emb,score,gate.mean(dim=(1,2,4)),pool_diag

__all__=["EarlyFastDynamicsEncoder","GatedEvidenceEncoder"]
