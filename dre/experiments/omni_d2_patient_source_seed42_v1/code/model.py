"""Frozen-logit additive linear correction; exactly35/105 trainable parameters."""
import torch
class Head(torch.nn.Module):
    def __init__(self,arm):
        super().__init__();assert arm in ['C1','C2','C3'];self.arm=arm
        self.beta=torch.nn.Parameter(torch.zeros(32));self.b=torch.nn.Parameter(torch.zeros(3))
        if arm!='C1':
            gen=torch.Generator(device='cpu').manual_seed(42)
            self.V=torch.nn.Parameter(.01*torch.randn(2,32,generator=gen));self.U=torch.nn.Parameter(torch.zeros(3,2))
        assert sum(p.numel() for p in self.parameters())==(35 if arm=='C1' else 105)
    def components(self,z,g):
        shared=z@self.beta;index=g.clamp(0,2)
        direction=torch.zeros_like(shared) if self.arm=='C1' else ((z@self.V.T)*self.U[index]).sum(-1)
        valid=(g>=0)&(g<3);direction=torch.where(valid,direction,torch.zeros_like(direction));bias=torch.where(valid,self.b[index],torch.zeros_like(shared))
        return shared,direction,bias
    def forward(self,z,g,base):
        shared,direction,bias=self.components(z,g);return base+(shared+direction+bias)
    def regularizer(self,counts):
        weights=(counts.sum()/3)/counts.clamp_min(1);source=(weights*self.b.square()).mean()
        if self.arm!='C1':source=source+self.V.square().mean()+(weights*self.U.square().mean(1)).mean()
        return .001*(self.beta.square().mean()+source)
