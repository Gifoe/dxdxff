import torch
from neuroez_c.task2.cop.p2_nez_auxiliary import P2_AUX_FEATURES,build_p2_auxiliary,fit_seizure_clean_nez_prototype


def _record(patient,shift=0.):
    return {"patient_key":patient,"center":"c","channel_mask":torch.ones(1,4,dtype=torch.bool),"clinical_target_mask":torch.tensor([[1,0,0,0]],dtype=torch.bool),"seizure_mask":torch.ones(1,2,dtype=torch.bool),"seizure_channel_mask":torch.ones(1,2,4,dtype=torch.bool),"final_nez_logit":torch.tensor([[0.,1.,2.,3.]])+shift,"seizure_channel_embedding":torch.randn(1,2,4,5)+shift}


def test_p2_auxiliary_has_exactly_four_fold_safe_features():
    torch.manual_seed(2);records=[_record("train"),_record("test",1.)];prototype=fit_seizure_clean_nez_prototype(records,{"train"});table=build_p2_auxiliary(records,prototype)
    assert all(name in table for name in P2_AUX_FEATURES) and len(P2_AUX_FEATURES)==4 and prototype.n_train_patients==1
