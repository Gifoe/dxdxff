import torch
from neuroez_c.task2.trace_rawmil.encoder import RawEncoder

def test_encoder_window_chunking_preserves_eval_embeddings():
 torch.manual_seed(3); encoder=RawEncoder().eval();x=torch.randn(7,1,500)
 a=encoder.stream(x,'cpu',2,False,False);b=encoder.stream(x,'cpu',32,False,False)
 assert torch.allclose(a,b,atol=1e-6) and encoder.last_encoder_chunks==1

def test_checkpointing_is_not_used_in_eval(monkeypatch):
 encoder=RawEncoder().eval();calls=[]
 import neuroez_c.task2.trace_rawmil.encoder as module
 monkeypatch.setattr(module,'checkpoint',lambda *a,**k:calls.append(1))
 encoder.stream(torch.randn(2,1,500),'cpu',1,True,False)
 assert not calls
