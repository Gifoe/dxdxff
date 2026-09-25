import torch
from neuroez_c.task2.trace_rawmil.encoder import RawEncoder
def test_encoder_shape():assert RawEncoder()(torch.randn(3,1,500)).shape==(3,32)
