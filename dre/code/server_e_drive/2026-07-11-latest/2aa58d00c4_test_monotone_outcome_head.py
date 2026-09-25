import torch
from neuroez_c.task2.monotone_outcome_head import MONOTONE_FEATURES,MonotoneOutcomeHead


def test_monotone_directions_and_bounded_set_delta():
    head=MonotoneOutcomeHead(MONOTONE_FEATURES,4); head.fit_normalizer(torch.stack((torch.zeros(len(MONOTONE_FEATURES)),torch.ones(len(MONOTONE_FEATURES)))))
    base=torch.zeros(1,len(MONOTONE_FEATURES)); risk=base.clone(); risk[0,0]=1; support=base.clone(); support[0,-1]=1
    assert head(risk,torch.zeros(1,4))["outcome_logit_success"]<=head(base,torch.zeros(1,4))["outcome_logit_success"]
    assert head(support,torch.zeros(1,4))["outcome_logit_success"]>=head(base,torch.zeros(1,4))["outcome_logit_success"]
    assert head(base,torch.full((1,4),100.))["set_delta"].abs().item()<=.15
