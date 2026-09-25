import torch
from task1_clean_nez.recurrence import CrossSeizureRecurrenceAggregator

def test_single_seizure_has_zero_std_and_finite_recurrence():
    emb=torch.randn(1,1,5,4); risk=torch.arange(5.).reshape(1,1,5); sm=torch.ones(1,1,dtype=torch.bool); cm=torch.ones(1,1,5,dtype=torch.bool)
    _,d=CrossSeizureRecurrenceAggregator(4)(emb,risk,sm,cm)
    assert torch.equal(d["risk_std_across_seizures"],torch.zeros_like(risk[:,0]))
    assert torch.isfinite(d["top20_recurrence"]).all() and d["top20_recurrence"].sum()==1

def test_unobserved_channel_is_masked_from_projection():
    emb=torch.randn(1,3,2,4); risk=torch.rand(1,3,2); sm=torch.ones(1,3,dtype=torch.bool); cm=torch.tensor([[[True,False]]*3])
    out,d=CrossSeizureRecurrenceAggregator(4)(emb,risk,sm,cm)
    assert d["valid_seizure_count_per_channel"].tolist()==[[3.,0.]] and not d["observed_channel_mask"][0,1] and torch.equal(out[0,1],torch.zeros(8))
