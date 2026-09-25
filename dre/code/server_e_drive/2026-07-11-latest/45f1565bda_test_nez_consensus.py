import torch
from neuroez_c.task2.nez_consensus import compute_nez_consensus


def test_consensus_reliability_and_abnormality():
    consistent=compute_nez_consensus(torch.tensor([[.1]]),torch.tensor([[.1]]),torch.tensor([[.1]]))
    discordant=compute_nez_consensus(torch.tensor([[.1]]),torch.tensor([[.9]]),torch.tensor([[.5]]))
    assert consistent["reliability_weight"].item()>discordant["reliability_weight"].item()
    assert consistent["reliable_abnormality"].item()>.8
    for value in discordant.values(): assert bool(((value>=0)&(value<=1)).all())
