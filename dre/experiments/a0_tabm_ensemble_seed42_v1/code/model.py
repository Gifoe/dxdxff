"""Exactly one width control and one four-member A0-specific TabM model."""
import torch
from torch import nn
from tabm_layers import EnsembleView,LinearBatchEnsemble,LinearEnsemble


class WidenedPRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.LayerNorm(88),nn.Linear(88,111),nn.GELU(),nn.Dropout(.15),nn.Linear(111,1))

    def forward(self,x):
        return self.network(x).squeeze(-1)


class TabMPRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(88)
        self.view=EnsembleView(k=4)
        self.hidden=LinearBatchEnsemble(88,96,k=4,scaling_init=('random-signs','ones'))
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(.15)
        self.head=LinearEnsemble(96,1,k=4)

    def forward(self,x):
        return self.head(self.dropout(self.activation(self.hidden(self.view(self.norm(x)))))).squeeze(-1)


def make(arm):
    if arm=='W1':return WidenedPRMLP()
    if arm=='W2':return TabMPRMLP()
    raise ValueError('No additional model arms')


def objective(logits,y,pos_weight):
    if logits.ndim==2:
        assert logits.shape[1]==4
        target=y[:,None].expand_as(logits)
        each=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=pos_weight,reduction='none')
        return each.mean(dim=1).mean()
    assert logits.ndim==1
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight)


def probability(logits):
    p=torch.sigmoid(logits)
    return p.mean(dim=1) if logits.ndim==2 else p
