import numpy as np
import pytest
import torch

from neuroez_c.task2.clinical_target import build_clinical_target_lookup, center_clinical_target, collate_clinical_targets, task1_nez_to_clinical_target
from neuroez_c.task2.p2_target_outcome_model import P2TargetOutcomeModel
from neuroez_c.task2.seizure_target_probe import SeizureTargetProbe, compute_cross_seizure_features, patient_balanced_probe_loss
from neuroez_c.task2.target_concordance import compute_target_concordance
from neuroez_c.task2.target_network import NETWORK_FEATURES, compute_target_network
from neuroez_c.task2.target_pooling import compute_target_pooling


def _cache(names=("A1","A2","B1"), labels=(0,1,1)):
    meta=[{"channel_name_norm":name,"label_source":"composite_ez"} for name in names]
    return {"run_records":[],"patient_index":{"lzu:p1":{"canonical_channels":list(reversed(names)),"channel_meta":meta,"labels":list(labels),"label_mask":[1]*len(names),"source_center":"lzu","surgery_success":True}}}


def test_label_semantics_conversion_and_missing_rejected():
    np.testing.assert_array_equal(task1_nez_to_clinical_target([0,1]),[1,0])
    with pytest.raises(ValueError): task1_nez_to_clinical_target([0,np.nan])
    target,source=center_clinical_target([{"soz":1,"resection":0},{"soz":0,"resection":1}],"hup",[1,1])
    assert target.tolist()==[1,1] and source==["HUP:SOZ_OR_RESECTION"]*2


def test_name_alignment_reorders_and_strict_rules():
    lookup,audit,_=build_clinical_target_lookup(_cache(),strict=True)
    mask=collate_clinical_targets(["lzu:p1"],[["B1","A1","A2"]],lookup,3)
    assert mask.tolist()==[[0,1,0]] and audit.iloc[0].match_fraction==1
    bad=_cache(("A1","A1","B1"),(0,1,1))
    with pytest.raises(ValueError): build_clinical_target_lookup(bad,strict=True)


def test_concordance_coverage_top_tail_and_count_metrics():
    a=torch.tensor([[.9,.8,.2,.1,0.]])
    t=torch.tensor([[1.,0.,1.,0.,0.]])
    m=torch.tensor([[1,1,1,1,0]],dtype=torch.bool)
    out=compute_target_concordance(a,t,m)
    assert torch.allclose(out["coverage_mass"]+out["residual_mass"],torch.ones(1))
    assert out["top05_target_coverage"].item()==1 and out["top10_target_miss"].item()==0
    assert out["target_count_precision"].item()==.5 and out["target_count_recall"].item()==.5
    assert out["target_count_jaccard"].item()==pytest.approx(1/3)


def test_target_pooling_masks_padding_and_has_safe_fallback():
    h=torch.tensor([[[1.,0.],[3.,0.],[100.,100.]]]); a=torch.tensor([[1.,.5,9.]])
    t=torch.tensor([[1.,0.,0.]]); m=torch.tensor([[1,1,0]],dtype=torch.bool)
    out=compute_target_pooling(h,a,t,m)
    assert out["target"].tolist()==[[1,0]] and out["non_target"].tolist()==[[3,0]]
    empty=compute_target_pooling(h,a,torch.ones_like(t),m)
    assert not empty["non_target_valid"].item() and torch.isfinite(empty["non_target"]).all()


def test_probe_is_patient_balanced_nonconstant_and_q10_shape():
    torch.manual_seed(3); emb=torch.randn(2,3,4,5); valid=torch.ones(2,3,4,dtype=torch.bool); target=torch.tensor([[1,0,0,1],[0,1,1,0]],dtype=torch.float32)
    probe=SeizureTargetProbe(5); out=probe(emb,valid); loss=patient_balanced_probe_loss(out["seizure_target_logit"],target,valid)
    assert loss.ndim==0 and out["seizure_target_probability"].std()>0
    features=compute_cross_seizure_features(out["seizure_target_probability"],target,valid)
    assert features["stable_target_coverage"].shape==(2,)


def test_target_network_three_phases_and_model_success_probability():
    b,s,c,d=2,2,4,8; adjacency=torch.ones(b,s,3,c,c)-torch.eye(c); phase_mask=torch.ones(b,s,3,c,dtype=torch.bool); graph_valid=torch.ones(b,s,3,dtype=torch.bool)
    a=torch.tensor([[.9,.7,.2,.1],[.8,.6,.3,.2]]); t=torch.tensor([[1,1,0,0],[1,0,1,0]],dtype=torch.float32)
    network=compute_target_network(adjacency,a,t,phase_mask,graph_valid)
    assert all(name in network for name in NETWORK_FEATURES)
    p2={"channel_mask":torch.ones(b,c,dtype=torch.bool),"final_score_ez":a,"final_nez_logit":torch.logit(1-a),"patient_channel_embedding":torch.randn(b,c,d),"seizure_channel_embedding":torch.randn(b,s,c,d),"seizure_nez_probability":torch.rand(b,s,c),"seizure_mask":torch.ones(b,s,dtype=torch.bool),"seizure_channel_mask":torch.ones(b,s,c,dtype=torch.bool),"temporal_delta_norm":torch.rand(b,c),"delta_onset_norm":torch.rand(b,c),"delta_spread_norm":torch.rand(b,c)}
    model=P2TargetOutcomeModel(d,"C2_P2_TARGET_CONCORDANCE").eval(); output=model(p2,clinical_target_mask=t)
    assert output["outcome_logit_success"].shape==(b,) and torch.allclose(output["outcome_probability_failure"],1-output["outcome_probability_success"])
