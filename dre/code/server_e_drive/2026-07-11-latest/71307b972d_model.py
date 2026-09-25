from __future__ import annotations
import torch
from torch import nn
from neuroez_c.task2.tech_outcome_c1.model import TeChOutcomeC1
class TeChOutcomeC1Aligned(TeChOutcomeC1):
    def __init__(self,d_model=64,d_core=16,cotar_layers=2,cotar_dropout=.05,channel_dropout=.05,seizure_dropout=.05,patient_dropout=.10):
        super().__init__(d_model,d_core,cotar_layers,cotar_dropout);self.channel=nn.Sequential(nn.Linear(128,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(channel_dropout),nn.Linear(96,64),nn.LayerNorm(64),nn.GELU());self.seizure=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,96),nn.GELU(),nn.Dropout(seizure_dropout),nn.Linear(96,96),nn.LayerNorm(96),nn.GELU());self.head=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,64),nn.GELU(),nn.Dropout(patient_dropout),nn.Linear(64,16),nn.GELU(),nn.Dropout(.05),nn.Linear(16,1));self.augmentations_enabled=False
    def forward_patient_views(self,grouped_views,encoder_window_batch_size=1024):
        sizes=[len(v) for v in grouped_views];flat=[view for patient in grouped_views for view in patient];view_outputs=super().forward(flat,encoder_window_batch_size);result=[];offset=0
        for size in sizes:
            outputs=view_outputs[offset:offset+size];offset+=size;logits=torch.stack([x['logit'].float() for x in outputs]);result.append({'logit':logits.mean(),'view_logits':logits,'view_outputs':outputs})
        return result
