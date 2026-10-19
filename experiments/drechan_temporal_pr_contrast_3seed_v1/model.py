"""Fixed six-feature no-interaction backbone plus1776-parameter temporal fusion."""
import importlib.util
from pathlib import Path
import sys
import torch
from contrast import Fusion,loo_reference,shifted_reference,relative,MIN_OTHERS,MAD_FLOOR,EPSILON,CLIP

PARENT=Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1')
spec=importlib.util.spec_from_file_location('tprc_parent_model',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['B0','B1','B2','B3']

class Temporal(BASE.Ablation):
    def __init__(self,source,arm,fold,seed):
        super().__init__(source,'FULL_F0');assert arm in ARMS
        self.variant=arm;self.fold=fold;self.reference=None;self.collect=False;self.last_audit={}
        if arm!='B0':self.temporal_fusion=Fusion(seed+270019)

    def input_tensors(self,raw,mask,times,delta,scale):
        S,C,W,L=raw.shape;assert L==500 and mask.shape==(S,C,W) and raw.dtype==torch.float32
        flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        masked=torch.where(self.keep7.bool(),delta,torch.zeros_like(delta))
        transform=self.raw_feature_function.__globals__['transform']
        u=(transform(masked,mask,times,'full',None)/scale).clamp(-8,8)
        assert torch.isfinite(masked[mask]).all()
        return dict(v=v,u=u,mask=mask,times=times,shape=(S,C,W))

    def encode_prepared(self,p):
        S,C,W=p['shape'];mask=p['mask'];v=p['v'];u=p['u']
        emb=v.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None],w).flatten(1) for q,w in zip(v.split(4096),u[mask].split(4096))])
        return emb

    @staticmethod
    def residuals(emb,mask,times):
        result=[]
        for s in range(emb.shape[0]):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
            assert (count[present]>0).all()
            center=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
            result.append(emb[s]-center[:,None])
        return torch.stack(result)

    def set_reference(self,ref):
        self.reference={k:v.to(next(self.parameters()).device) if isinstance(v,torch.Tensor) else v for k,v in ref.items()} if ref is not None else None

    def contrast(self,r,mask,times,seizure):
        C,W,D=r.shape
        if self.variant=='B2':
            assert self.reference is not None
            ref=self.reference;bins=ref['times'];idx=torch.searchsorted(bins,times).clamp_max(len(bins)-1)
            matched=bins[idx]==times
            median=ref['median'][idx][None].expand_as(r);mad=ref['mad'][idx][None].expand_as(r)
            ok=mask[:,:,None]&matched[None,:,None]&(ref['count'][idx][None,:,None]>=MIN_OTHERS)&(mad>MAD_FLOOR)
            extra={}
        else:
            median,mad,ok=loo_reference(r,mask);extra={}
            if self.variant=='B3':median,mad,ok,extra=shifted_reference(median,mad,ok,mask,times,self.fold,seizure)
        q=relative(r,median,mad,ok)
        if self.collect:
            valid=mask[:,:,None].expand_as(r)
            extra.update(reference_valid_coordinates=int((ok&valid).sum()),reference_fallback_coordinates=int((~ok&valid).sum()),
                q_clipped_coordinates=int(((q.abs()>=CLIP)&valid).sum()),post_valid_windows=int(mask.sum()))
        return self.temporal_fusion(r,q),extra

    def features(self,raw,mask,times,delta,scale,permutation=None):
        if self.variant=='B0':return super().features(raw,mask,times,delta,scale,permutation)
        p=self.input_tensors(raw,mask,times,delta,scale);emb=self.encode_prepared(p);res=self.residuals(emb,mask,times)
        summaries=[];available=[];activity=[];audits={}
        for s in range(raw.shape[0]):
            posttime=(times[s]>=0)&(times[s]<20);post=mask[s]&posttime[None]
            fused,stats=self.contrast(res[s][:,posttime],mask[s][:,posttime],times[s][posttime],s)
            latent=res[s].clone();latent[:,posttime]=fused
            g=self.gate(p['u'][s]);activity.append(g[post]);guided=g*latent;st=[];av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=select.sum(1)
                st.append((guided*select[...,None]).sum(1)/n.clamp_min(1)[:,None]);av.append(n>0)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
            for k,v in stats.items():audits[k]=audits.get(k,0)+v
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0);assert (n>0).all()
        mu=(summary*valid).sum(0)/n;var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var));vector=torch.stack([mu,sd],2).reshape(raw.shape[1],96).double()
        # Original FINAL patient Z remains fully differentiable, unlike reference backgrounds.
        z=(vector-vector.mean(0))/torch.where(vector.std(0,unbiased=False)>1e-8,vector.std(0,unbiased=False),torch.ones_like(vector.std(0,unbiased=False)))
        self.last_audit=audits
        return z.to(raw.dtype),torch.cat(activity)

objective=BASE.objective

class GlobalReference:
    """All current TRAIN waveform windows, current CNN, every optimizer update.

    Input normalization/conditioning tensors are weight-independent and cached
    on GPU without writing a new Raw cache. Encoding retains original per-patient
    4096-window chunk boundaries. No sampling, stale encoder or validation input.
    """
    def __init__(self,m,d,inputs,scale):
        self.packs=[];self.refreshes=0;self.valid_windows=0;self.fit_ids=sorted(set(d['patient'][d['train']]))
        with torch.no_grad():
            for pid in self.fit_ids:
                ii=d['train'][d['patient'][d['train']]==pid]
                raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
                p=m.input_tensors(raw,mask,times,delta,scale)
                self.valid_windows+=int(mask.sum());self.packs.append({k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in p.items()})
                del raw,delta

    @torch.no_grad()
    def fit(self,m):
        arrays=[];clock=[]
        for p in self.packs:
            emb=m.encode_prepared(p);r=m.residuals(emb,p['mask'],p['times'])
            S,C,W=p['shape'];times=p['times'][:,None,:].expand(S,C,W);post=p['mask']&(times>=0)&(times<20)
            arrays.append(r[post]);clock.append(times[post])
        x=torch.cat(arrays);t=torch.cat(clock);bins=torch.unique(t,sorted=True);med=[];mad=[];counts=[]
        for value in bins:
            a=x[t==value];mu=torch.quantile(a,.5,dim=0);md=torch.quantile((a-mu).abs(),.5,dim=0)
            med.append(mu);mad.append(md);counts.append(len(a))
        self.refreshes+=1
        return dict(times=bins.detach(),median=torch.stack(med).detach(),mad=torch.stack(mad).detach(),count=torch.tensor(counts,device=x.device))
