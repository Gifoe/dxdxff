import torch
from torch import nn
from .omni_archived import OmniTimeConvCNNEncoder,OmniSEEGNetEncoder
from .seegformer.model import SEEGformerTask1
from patient_relative import patient_relative


class EEGNetAdapted(nn.Module):
    """EEGNet-8,2 principles; spatial depthwise kernel degenerates to one channel.

    Temporal kernels scaled from 128 to250Hz; no invented electrode count.
    Flattened final temporal map remains the window-level embedding.
    """
    output_dim=16*(500//4//8)
    def __init__(self):
        super().__init__()
        self.temporal=nn.Sequential(nn.Conv1d(1,8,125,padding=62,bias=False),nn.BatchNorm1d(8,eps=.001,momentum=.01))
        self.spatial=nn.Conv1d(8,16,1,groups=8,bias=False)
        self.first=nn.Sequential(nn.BatchNorm1d(16,eps=.001,momentum=.01),nn.ELU(),nn.AvgPool1d(4),nn.Dropout(.5))
        self.second=nn.Sequential(nn.Conv1d(16,16,31,padding=15,groups=16,bias=False),nn.Conv1d(16,16,1,bias=False),nn.BatchNorm1d(16,eps=.001,momentum=.01),nn.ELU(),nn.AvgPool1d(8),nn.Dropout(.5))
    def forward(self,x):return self.second(self.first(self.spatial(self.temporal(x)))).flatten(1)


def normalized(x):
    return ((x-x.mean(-1,keepdim=True))/x.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)


class RawBackbone(nn.Module):
    def __init__(self,name,pr,config):
        super().__init__();self.name=name;self.pr=pr;self.multichannel=name=='SEEGformer'
        if name=='TimeConvCNN':self.encoder=OmniTimeConvCNNEncoder(n_times=500,sfreq=250,timeconv_pretrained_resnet=False)
        elif name=='SEEGNet':self.encoder=OmniSEEGNetEncoder(n_times=500,sfreq=250)
        elif name=='EEGNet':self.encoder=EEGNetAdapted()
        elif name=='SEEGformer':
            self.encoder=SEEGformerTask1(**config['SEEGformer'])
            # Same branch head shape except for required mean/SD aggregation width.
            for branch in [self.encoder.real_branch,self.encoder.imag_branch,self.encoder.amplitude_branch]:
                branch.head[0]=nn.LayerNorm(128);branch.head[1]=nn.Linear(128,32)
            self.window_dim=192
        else:raise ValueError(name)
        if not self.multichannel:
            self.window_dim=self.encoder.output_dim
            self.head=nn.Linear(self.window_dim*2,1)

    def encode(self,raw,mask=None):
        raw=normalized(raw)
        if not self.multichannel:return self.encoder(raw[:,None])
        assert mask is not None and mask.shape==raw.shape[:2]
        raw=raw.masked_fill(~mask[...,None],0)
        bands=self.encoder.frontend(raw,mask);emb=[]
        for features,branch in zip(bands,[self.encoder.real_branch,self.encoder.imag_branch,self.encoder.amplitude_branch]):
            values=branch.projection(features);B=values.shape[0]
            values=torch.cat([branch.cls.expand(B,-1,-1),values],1)
            pad=torch.cat([torch.zeros(B,1,dtype=torch.bool,device=raw.device),~mask],1)
            for block in branch.blocks:values,_=block(values,pad)
            emb.append(values[:,1:].masked_fill(~mask[...,None],0))
        return torch.cat(emb,-1)

    def classify(self,embedding):
        h=patient_relative(embedding) if self.pr else embedding
        if not self.multichannel:return self.head(h).squeeze(-1)
        # feature-major mean/SD interleaving retains original branch separation.
        branch_h=h.reshape(len(h),3,128)
        heads=[self.encoder.real_branch.head,self.encoder.imag_branch.head,self.encoder.amplitude_branch.head]
        logits=torch.stack([head(branch_h[:,j]).squeeze(-1) for j,head in enumerate(heads)],1)
        return (logits*torch.softmax(self.encoder.branch_weight_logits,0)).sum(1)

    @torch.no_grad()
    def constraints(self):
        if self.name=='EEGNet':
            w=self.encoder.spatial.weight;w.mul_((1/w.flatten(1).norm(dim=1).clamp_min(1)).view(-1,1,1))
            w=self.head.weight;w.mul_((.25/w.norm(dim=1).clamp_min(.25))[:,None])


def aggregate(window_embeddings,mask):
    """Mean genuine windows per seizure/channel; mean/population SD over seizures."""
    assert window_embeddings.shape[:3]==mask.shape
    v=window_embeddings.float();m=mask[...,None].to(v.dtype)
    n=m.sum(2);has=n>0
    seizure=(v*m).sum(2)/n.clamp_min(1)
    count=has.sum(0);assert (count>0).all()
    mean=(seizure*has).sum(0)/count
    var=((seizure-mean[None]).square()*has).sum(0)/count
    sd=torch.where(var>0,var.clamp_min(1e-30).sqrt(),torch.zeros_like(var))
    return torch.stack([mean,sd],-1).flatten(1)
