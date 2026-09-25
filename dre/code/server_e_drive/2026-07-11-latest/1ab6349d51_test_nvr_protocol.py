import pytest
import torch
from neuroez_c.task2.nvr_negative_controls import graph_permutation,p2_channel_permutation,target_permutation
from neuroez_c.task2.nvr_outcome_model import NVROutcomeModel
from neuroez_c.task2.profiles import nvr_profile_names,profile_names
from neuroez_c.task2.protocol import ProtocolError,assert_checkpoint_safe


def test_r_profiles_are_separate_and_model_has_no_center_input():
    assert len(nvr_profile_names())==7 and not set(nvr_profile_names())&set(profile_names())
    model=NVROutcomeModel(8,"R6_ROBUST_FULL"); assert "center" not in model.forward.__annotations__


def test_negative_target_control_preserves_counts():
    target=torch.tensor([[1,1,0,0,0],[1,0,0,1,0]],dtype=torch.bool); valid=torch.ones_like(target)
    shuffled=target_permutation(target,valid,42); assert torch.equal(target.sum(-1),shuffled.sum(-1)) and not torch.equal(target,shuffled)


def test_p2_and_graph_controls_preserve_values_but_break_mapping():
    valid=torch.ones(1,5,dtype=torch.bool); score=torch.arange(5.).reshape(1,5)
    shuffled=p2_channel_permutation({"q_final_nez":score},valid,42)["q_final_nez"]
    assert torch.equal(torch.sort(score).values,torch.sort(shuffled).values) and not torch.equal(score,shuffled)
    graph=torch.arange(25.).reshape(1,1,1,5,5); changed=graph_permutation(graph,valid,42)
    assert torch.equal(torch.sort(graph.flatten()).values,torch.sort(changed.flatten()).values) and not torch.equal(graph,changed)


def test_p2_outer_test_overlap_is_rejected():
    with pytest.raises(ProtocolError): assert_checkpoint_safe(["p1","p2"],["p2","p3"])
