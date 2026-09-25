from __future__ import annotations
import torch
from torch import nn
from .raw_stem import RawWindowStem
from .masked_ops import masked_mean,masked_max
from .cotar import MaskedCoTAREncoderLayer
from .augment import augment_seizure

class TeChOutcomeC1(nn.Module):
    def __init__(self,d_model=64,d_core=16,cotar_layers=2,dropout=.1):
        super().__init__();self.raw=RawWindowStem();self.phase=nn.Embedding(3,d_model);self.time=nn.Sequential(nn.Linear(1,16),nn.GELU(),nn.Linear(16,d_model));self.channel=nn.Sequential(nn.Linear(128,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,64),nn.LayerNorm(64),nn.GELU());self.layers=nn.ModuleList([MaskedCoTAREncoderLayer(d_model,d_core,dropout) for _ in range(cotar_layers)]);self.seizure=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,96),nn.GELU(),nn.Dropout(.15),nn.Linear(96,96),nn.LayerNorm(96),nn.GELU());self.head=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1));self.augmentations_enabled=True
    @staticmethod
    def normalize(w):
        median=w.median(-1,keepdim=True).values;mad=(w-median).abs().median(-1,keepdim=True).values;return ((w-median)/(1.4826*mad).clamp_min(1e-6)).clamp(-8,8)
    def forward(self,views,encoder_window_batch_size=1024):
        prepared=[];flat=[]
        for view in views:
            patient=[]
            for seizure in view['seizures']:
                windows,mask=seizure['windows'],seizure['window_mask'].bool()
                if self.training and self.augmentations_enabled:windows,mask,_=augment_seizure(windows,mask)
                active=torch.where(mask.any(1))[0]
                if len(active)==0:raise ValueError('all-padding seizure')
                windows,mask=windows[active],mask[active];phase=seizure['phase_ids'];times=seizure['relative_times_sec'];phase=phase[None].expand(seizure['windows'].shape[:2]) if phase.ndim==1 else phase;times=times[None].expand(seizure['windows'].shape[:2]) if times.ndim==1 else times;phase,times=phase[active],times[active]
                indexes=mask.nonzero();start=sum(len(x) for x in flat);values=self.normalize(windows[mask]).unsqueeze(1);flat.append(values);patient.append((seizure,mask,phase,times,indexes,start,start+len(values)))
            prepared.append(patient)
        device=next(self.parameters()).device;all_windows=torch.cat(flat);emb=self.raw.stream(all_windows,encoder_window_batch_size,device=device);outputs=[]
        for view,patient in zip(views,prepared):
            seizure_embeddings=[];core_audits=[]
            for seizure,mask,phase,times,indexes,start,end in patient:
                c,t=mask.shape;h=torch.zeros(c,t,64,device=device,dtype=emb.dtype);h[indexes[:,0],indexes[:,1]]=emb[start:end];phase=phase.to(device);times=times.to(device).clamp(-10,15)/15;z=h+self.phase(phase)+self.time(times[...,None]);token=self.channel(torch.cat([masked_mean(z,mask.to(device),1),masked_max(z,mask.to(device),1)],-1));valid=mask.any(1).to(device);x=token[None]
                for layer_index,layer in enumerate(self.layers,1):
                    x=layer(x,valid[None]);weights=layer.cotar.last_weights.float();core=layer.cotar.last_core.float()
                    for core_dimension in range(weights.shape[-1]):
                        dimension_weights=weights[0,valid,core_dimension];entropy=-(dimension_weights.clamp_min(1e-12)*dimension_weights.clamp_min(1e-12).log()).sum();core_audits.append({'seizure_id':seizure['seizure_id'],'layer_index':layer_index,'n_valid_channels':int(valid.sum()),'core_dimension':core_dimension,'core_norm':float(core[0,0,core_dimension].abs()),'mean_channel_weight_entropy':float(entropy),'minimum_channel_weight':float(dimension_weights.min()),'maximum_channel_weight':float(dimension_weights.max()),'effective_channel_count':float(torch.exp(entropy)),'finite':bool(torch.isfinite(dimension_weights).all())})
                x=x[0];seizure_embeddings.append(self.seizure(torch.cat([masked_mean(x,valid,0),masked_max(x,valid,0)]).float()))
            u=torch.stack(seizure_embeddings);std=u.std(0,unbiased=False) if len(u)>1 else torch.zeros_like(u[0]);patient_embedding=torch.cat([u.mean(0),std,u.amax(0)]);logit=self.head(patient_embedding.float()).squeeze(-1);outputs.append({'logit':logit,'patient_embedding':patient_embedding,'seizure_embeddings':u,'core_audits':core_audits,'n_seizures':len(u),'mean_valid_channels':sum(x['n_valid_channels'] for x in core_audits if x['layer_index']==1 and x['core_dimension']==0)/len(u)})
        return outputs
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
