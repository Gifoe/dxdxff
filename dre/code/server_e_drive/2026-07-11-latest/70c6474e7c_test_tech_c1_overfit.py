import torch
from neuroez_c.task2.tech_outcome_c1.losses import patient_bce
def test_bce_backward():
 logit=torch.tensor(0.,requires_grad=True);loss=patient_bce([{'logit':logit}],[1]);loss.backward();assert torch.isfinite(loss) and torch.isfinite(logit.grad)
