import torch
from task1_clean_nez.model import CleanNEZModel
from task1_clean_nez.losses import CleanNEZLoss
from tests.test_task1_clean_nez_model import batch
def test_unobserved_labeled_channel_is_not_scored_or_lost():
    b=batch(b=1,s=2,c=7); b["seizure_channel_mask"][:,:,0]=False; o=CleanNEZModel()(b)
    assert not o["effective_channel_mask"][0,0] and o["score_nez_probability"][0,0]==0 and o["candidate_risk"][0,0]==0
    x=CleanNEZLoss()(o,b)["loss"]; b["labels_nez"][0,0]=0;b["labels_ez"][0,0]=1; assert torch.allclose(x,CleanNEZLoss()(o,b)["loss"])
