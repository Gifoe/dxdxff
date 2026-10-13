"""Only input scaling, 2-wide residual, or registered monotonic reparameterization."""
import torch
from torch import nn
from torch.nn import functional as F


class Curves(nn.Module):
 def __init__(self,stats,linear=False):
  super().__init__();self.linear=linear
  self.theta=nn.Parameter(torch.zeros(56) if linear else torch.zeros(56,8))
  self.register_buffer('column_to_curve',torch.as_tensor(stats['column_to_curve'],dtype=torch.long))
  if not linear:
   self.register_buffer('knots',torch.as_tensor(stats['knots'],dtype=torch.float64))
   self.register_buffer('slope_mask',torch.as_tensor(stats['slope_mask'],dtype=torch.bool))
   self.register_buffer('boundary_mask',torch.as_tensor(stats['boundary_mask'],dtype=torch.bool))

 def slopes(self):
  s=1+.75*torch.tanh(self.theta.to(torch.float64))
  return s if self.linear else torch.where(self.slope_mask,s,torch.ones_like(s))

 def forward(self,x,permutation=None):
  cmap=self.column_to_curve.clamp_min(0)
  if permutation is not None:cmap=permutation[cmap]
  slopes=self.slopes()[cmap];u=x.to(torch.float64)
  if self.linear:v=u*slopes
  else:
   t=self.knots[cmap];bm=self.boundary_mask[cmap]
   diff=(slopes[:,1:]-slopes[:,:-1])*bm
   v=slopes[:,0]*u+(diff*(F.relu(u.unsqueeze(-1)-t)-F.relu(-t))).sum(-1)
  out=torch.where(self.column_to_curve>=0,v,u).to(x.dtype)
  return out


def constructor(original,arm,stats):
 assert arm in ['B0','B1','B2','B3']
 class Model(nn.Module):
  def __init__(self,seed=42):
   super().__init__();self.arm=arm
   # First original constructor: preserve shared weights and original RNG path.
   self.network=original(seed).network
   if arm in ['B1','B3']:self.transform=Curves(stats,linear=arm=='B1')
   if arm=='B2':
    self.input_residual=nn.Sequential(nn.Linear(88,2),nn.GELU(),nn.Linear(2,88))
    nn.init.zeros_(self.input_residual[-1].weight);nn.init.zeros_(self.input_residual[-1].bias)
   assert sum(p.numel() for p in self.parameters())=={'B0':8817,'B1':8873,'B2':9259,'B3':9265}[arm]

  def transformed(self,x,identity=False,permutation=None):
   if identity or self.arm=='B0':return x
   if self.arm=='B2':return x+self.input_residual(x)
   return self.transform(x,permutation)

  def forward(self,x,g=None,identity=False,permutation=None):
   return self.network(self.transformed(x,identity,permutation)).squeeze(-1)

  def regularizer(self,counts):return next(self.parameters()).new_zeros(())
 return Model
