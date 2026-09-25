from __future__ import annotations
import contextlib
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class RawEncoder(nn.Module):
 def __init__(self):
  super().__init__(); self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,k,4,k//2),nn.GroupNorm(2,8),nn.GELU()) for k in (15,31,63)])
  self.b1=nn.Sequential(nn.Conv1d(24,24,15,2,7,groups=24),nn.Conv1d(24,32,1),nn.GroupNorm(4,32),nn.GELU(),nn.Dropout(.1)); self.b2=nn.Sequential(nn.Conv1d(32,32,9,2,4,groups=32),nn.Conv1d(32,48,1),nn.GroupNorm(6,48),nn.GELU(),nn.Dropout(.1)); self.project=nn.Sequential(nn.Linear(96,32),nn.LayerNorm(32),nn.GELU())
  self.last_n_flat_windows=0; self.last_encoder_chunks=0
 def _forward_one(self,x,checkpoint_blocks=False):
  x=torch.cat([b(x) for b in self.branches],1)
  if checkpoint_blocks and self.training:
   x=checkpoint(self.b1,x,use_reentrant=False);x=checkpoint(self.b2,x,use_reentrant=False)
  else: x=self.b2(self.b1(x))
  return self.project(torch.cat([x.mean(-1),x.amax(-1)],1))
 def forward(self,x): return self._forward_one(x)
 def stream(self,windows,device,window_batch_size,checkpoint_blocks=False,autocast_enabled=False):
  """Raw windows remain on CPU; only a bounded convolution chunk enters CUDA."""
  device=torch.device(device);n=len(windows); self.last_n_flat_windows=n; self.last_encoder_chunks=0
  if not n: return torch.empty((0,32),device=device)
  output=torch.empty((n,32),device=device,dtype=torch.float32)
  for start in range(0,n,window_batch_size):
   end=min(n,start+window_batch_size); chunk=windows[start:end]
   if device.type=='cuda' and chunk.device.type=='cpu' and not chunk.is_pinned(): chunk=chunk.pin_memory()
   chunk=chunk.to(device,non_blocking=device.type=='cuda')
   amp=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if autocast_enabled and device.type=='cuda' else contextlib.nullcontext()
   with amp: encoded=self._forward_one(chunk,checkpoint_blocks)
   output[start:end].copy_(encoded.float()); self.last_encoder_chunks+=1
  return output
