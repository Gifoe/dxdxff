"""Train-time old7 masking in BOTH D1 and M8's physiology conditioner."""
import torch
from direct_correction import MorphCorrection


class MaskedM8(MorphCorrection):
    def __init__(self,base,keep):
        super().__init__(base)
        self.register_buffer('keep7',torch.tensor(keep,dtype=torch.float32))
        assert self.keep7.shape==(7,)

    def features(self,raw,mask,times,delta,scale,mode='full',perms=None):
        return super().features(raw,mask,times,delta*self.keep7,scale,mode,perms)

    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        return super().forward(x*self.keep7.repeat_interleave(2),raw,mask,times,delta,scale,mode,perms)
