import torch
from neuroez_c.task2.clean_nez_prototype import CleanNEZPrototype


def test_patient_balanced_clean_nez_prototype_and_probability_range():
    small=torch.tensor([[0.,0.]])
    large=torch.tensor([[10.,10.]]).repeat(100,1)
    prototype=CleanNEZPrototype("patient").fit([small,large])
    assert torch.allclose(prototype.prototype_mu,torch.tensor([5.,5.]))
    probability=prototype.probability_nez(torch.tensor([[0.,0.],[100.,100.]]))
    assert bool(((probability>=0)&(probability<=1)).all())
    state=prototype.state_dict(); assert state["n_train_patients"]==2


def test_transform_does_not_refit_from_outer_test():
    prototype=CleanNEZPrototype("patient").fit([torch.zeros(2,3)])
    before=prototype.prototype_mu.clone(); prototype.probability_nez(torch.full((4,3),100.))
    assert torch.equal(before,prototype.prototype_mu)
