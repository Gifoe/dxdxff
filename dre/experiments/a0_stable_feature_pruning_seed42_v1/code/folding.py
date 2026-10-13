"""Exact LN88 reparameterization on manifest-derived unique coordinates.

Let A_i = W_i * gamma_i and Z=sum A_i over zero input positions.
Their normalized value u_0=-mu/sigma obeys
u_0=-sum_j(m_j*u_j)/n0. Thus folded W_j=sum_{copies} A_i-Z*m_j/n0.
The affine bias is b+W@beta. No learned normalizer is added.
"""
import copy
import numpy as np
import torch
from torch import nn


def mapping(columns):
    assert [q['index'] for q in columns] == list(range(88))
    retained = [q for q in columns if q['view'] != 'zdelta']
    zero = [q['index'] for q in retained if q['constant_zero_in_frozen_export']]
    dup = [(q['index'], q['duplicate_of']) for q in retained if 'duplicate_of' in q]
    unique = [q for q in retained if not q['constant_zero_in_frozen_export'] and 'duplicate_of' not in q]
    assert len(retained)==62 and len(zero)==12 and len(dup)==8 and len(unique)==42
    assert all(q['block']=='b0' for q in unique)
    records=[]
    for q in unique:
        copies=[q['index']]+[i for i,j in dup if j==q['index']]
        records.append(dict(q, copies=copies, multiplicity=len(copies)))
    keys=sorted({(q['physiological_feature'],q['view']) for q in records})
    groups=[]
    for feature,view in keys:
        members=[q['index'] for q in records if q['physiological_feature']==feature and q['view']==view]
        assert len(members)==2
        assert {q['aggregation'] for q in records if q['index'] in members}=={'mean','std'}
        groups.append(dict(group=len(groups),feature=feature,view=view,members=members))
    assert len(groups)==21 and sum(q['multiplicity'] for q in records)==50
    return dict(unique=records,groups=groups,zero=zero,duplicates=dup,masked=[q['index'] for q in columns if q['view']=='zdelta'])


def fold_weights(network, records):
    ln,linear=network[0],network[1]
    w=linear.weight.detach().double(); gamma=ln.weight.detach().double(); beta=ln.bias.detach().double()
    active=[i for q in records for i in q['copies']]
    zeros=[i for i in range(88) if i not in active]; n0=len(zeros)
    assert n0>0 and linear.in_features==88 and ln.eps==1e-5
    a=w*gamma; z=a[:,zeros].sum(1)
    folded=torch.stack([a[:,q['copies']].sum(1)-z*q['multiplicity']/n0 for q in records],1)
    bias=linear.bias.detach().double()+w@beta
    return folded,bias


class Compact(nn.Module):
    def __init__(self, network, records):
        super().__init__()
        assert records
        self.register_buffer('indices',torch.tensor([q['index'] for q in records]))
        self.register_buffer('multiplicities',torch.tensor([q['multiplicity'] for q in records],dtype=torch.float64))
        self.n0=88-sum(q['multiplicity'] for q in records)
        w,b=fold_weights(network,records)
        self.first=nn.Linear(len(records),96,dtype=network[1].weight.dtype)
        with torch.no_grad():self.first.weight.copy_(w);self.first.bias.copy_(b)
        self.last=copy.deepcopy(network[4]);self.dropout=nn.Dropout(.15)
        assert sum(p.numel() for p in self.parameters())==96*len(records)+193

    def normalized(self,x):
        m=self.multiplicities.to(x.dtype)
        mu=(x*m).sum(-1,keepdim=True)/88
        var=(((x-mu).square()*m).sum(-1,keepdim=True)+self.n0*mu.square())/88
        return (x-mu)/torch.sqrt(var+1e-5)

    def forward(self,x,g=None):
        # Same original preprocessing; no removed coordinate enters any pathway.
        if x.shape[-1]==88:x=x.index_select(-1,self.indices)
        assert x.shape[-1]==len(self.indices)
        return self.last(self.dropout(torch.nn.functional.gelu(self.first(self.normalized(x))))).squeeze(-1)

    def regularizer(self,counts):return self.first.weight.new_zeros(())


def factory(original,records):
    def construct(seed=42):return Compact(original(seed).network,records)
    return construct


def retained(mp,remove_groups):
    deleted={i for q in mp['groups'] if q['group'] in remove_groups for i in q['members']}
    return [q for q in mp['unique'] if q['index'] not in deleted]


def inflate(x,records):
    out=x.new_zeros((len(x),88))
    for j,q in enumerate(records):out[:,q['copies']]=x[:,j,None]
    return out
