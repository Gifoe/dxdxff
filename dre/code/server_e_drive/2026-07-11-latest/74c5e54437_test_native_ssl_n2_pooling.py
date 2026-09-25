import torch
from neuroez_c.task2.native_ssl_n2.outcome_model import OutcomeModel
from neuroez_c.task2.native_ssl_n2.ssl_model import NativeEncoder
def test_singleton_std_is_zero():
 model=OutcomeModel(NativeEncoder()); assert torch.equal(torch.zeros(2),torch.zeros(2))
