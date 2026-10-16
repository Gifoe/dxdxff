"""A single Raw classifier, with matched global-versus-local hidden modulation."""
import torch
from torch import nn
import torch.nn.functional as F

ARMS = ['NATIVE','REFIT','PATIENT','LOCAL','NO_REFERENCE','NO_PHASES','NO_SD','NO_Z']
RECIPES = dict.fromkeys(ARMS, '1111')
RECIPES.update(NO_REFERENCE='0111', NO_PHASES='1011', NO_SD='1101', NO_Z='1110')


def groups(patient):
    import numpy as np
    _, inverse = np.unique(patient, return_inverse=True)
    return torch.as_tensor(inverse, dtype=torch.long)


class Head(nn.Module):
    def __init__(self, arm, reference_state, fold):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42 + 1009 * fold)
            self.head = nn.Sequential(nn.Linear(160,64), nn.GELU(), nn.Linear(64,1))
            self.head.load_state_dict({k.removeprefix('head.'):v for k,v in reference_state.items() if k.startswith('head.')})
            if arm in ['PATIENT','LOCAL']:
                torch.manual_seed(42 + 1009 * fold + 370003)
                self.phi = nn.Sequential(nn.Linear(96,16), nn.GELU())
                self.conditioner = nn.Linear(16,128)
                nn.init.zeros_(self.conditioner.weight); nn.init.zeros_(self.conditioner.bias)

    def forward(self, z, group):
        assert z.shape[1] == 96 and group.shape == (len(z),)
        h = self.head[1](self.head[0](torch.cat([z,z.new_zeros(len(z),64)],-1)))
        if self.arm in ['PATIENT','LOCAL']:
            c = self.phi(z)
            if self.arm == 'PATIENT':
                total = c.new_zeros(int(group.max())+1,16).index_add(0,group,c)
                count = torch.bincount(group,minlength=len(total)).to(c.dtype)
                c = (total/count[:,None])[group]
            gamma,beta = self.conditioner(c).chunk(2,-1)
            h = h*(1+.25*gamma.tanh())+.25*beta.tanh()
        return self.head[2](h).squeeze(-1)


def objective(logit, y, group, pos):
    n = int(group.max())+1
    count = torch.bincount(group,minlength=n).to(logit.dtype)
    bce = F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos,reduction='none')
    bce = (logit.new_zeros(n).index_add(0,group,bce)/count).mean()
    p = logit.sigmoid()
    def summed(x): return logit.new_zeros(n).index_add(0,group,x)
    nez = 2*summed(p*y)/(summed(p)+summed(y)).clamp_min(1e-8)
    ez = 2*summed((1-p)*(1-y))/(summed(1-p)+summed(1-y)).clamp_min(1e-8)
    return .75*bce+.25*(1-.5*(ez+nez)).mean()
