from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from neuroez_c.cane_path_cp_heads import CausalPropagationResidual, CleanNEZPrototypeAnchor, MultiSeizureNEZEvidenceResidual
from neuroez_c.cane_path_cp_loss import CenterBalancedPatientBatchSampler, compute_cane_path_cp_ranking_loss
from neuroez_c.cane_path_cp_trainer import (
    apply_fixed_nez_probability_threshold,
    ranking_validation_summary,
    _write_ledgers,
)
from neuroez_c.cane_set_loss import clean_nez_prototype_loss
from neuroez_c.model import NeuroEZCModel
from patient_channel_ranker import PatientChannelClassifier
from run_neuroez_c import validate_cane_path_cp_args
from exp_ez_hybrid import _summarize_prediction_records
from tests.cane_path_test_helpers import cane_args, cane_model, synthetic_batch


def _identity_anchor() -> CleanNEZPrototypeAnchor:
    anchor = CleanNEZPrototypeAnchor(2, projection_dim=2, num_prototypes=2, temperature=.01)
    anchor.projector = nn.Identity()
    with torch.no_grad():
        anchor.prototypes.copy_(torch.tensor([[1., 0.], [0., 1.]]))
    return anchor


def test_nez_is_one_ez_is_zero() -> None:
    labels_ez = torch.tensor([1., 0.])
    assert torch.equal(1-labels_ez, torch.tensor([0., 1.]))


def test_prediction_summary_ignores_batch_padding_beyond_canonical_channels() -> None:
    record = {
        "subject_id": "hup:padding", "center": "hup", "center_id": 0,
        "canonical_channels": ["A1", "A2"],
        "channel_mask": np.array([True, True, False]),
        # The final padded element deliberately looks like NEZ.  It must
        # never be exported because no canonical channel exists for it.
        "labels_nez": np.array([1.0, 0.0, 1.0]),
        "labels_ez": np.array([0.0, 1.0, 0.0]),
        "score_nez": np.array([0.8, 0.2, 0.9]),
        "score_ez": np.array([0.2, 0.8, 0.1]),
        "positive_label": "nez",
        "classification_threshold": 0.5,
    }
    _, enriched = _summarize_prediction_records([record])
    assert enriched[0]["true_nez_channels"] == ["A1"]
    assert enriched[0]["channel_name_alignment_ok"] is False


def test_ledger_writer_accepts_summary_list_masks(tmp_path) -> None:
    record = {
        "subject_id": "pediatric:S002", "center": "pediatric",
        "canonical_channels": ["A1", "A2"], "channel_mask": [True, True],
        "labels_nez": [1.0, 0.0], "labels_ez": [0.0, 1.0],
        "score_nez": [0.8, 0.2], "score_ez": [0.2, 0.8],
        "positive_label": "nez", "decision_rule": "patient_adaptive_standardized_nez_threshold",
        "predicted_nez_mask": [1, 0], "predicted_ez_mask": [0, 1],
        "predicted_patient_threshold": 0.1, "n_seizures": 2,
        "inner_fold_source": "outer1_test",
        "causal_propagation_features": [[0.0] * 6, [0.0] * 6],
        "causal_feature_valid": [False, False],
    }
    for name in (
        "direct_nez_logit", "final_nez_logit", "standardized_nez_logit",
        "direct_score_nez", "anchor_residual", "anchor_distance",
        "anchor_nez_evidence", "seizure_residual", "seizure_nez_probability_mean",
        "seizure_nez_probability_std", "seizure_nez_agreement", "causal_residual",
    ):
        record[name] = [0.0, 0.0]

    _, enriched = _summarize_prediction_records([record])
    assert isinstance(enriched[0]["predicted_nez_mask"], list)
    _write_ledgers(enriched, tmp_path, fold=1, seed=42)

    patient = np.genfromtxt(
        tmp_path / "test_patient_predictions_neuroez_v2_fold_1.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8",
    )
    channel = np.genfromtxt(
        tmp_path / "test_channel_predictions_neuroez_v2_fold_1.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8",
    )
    assert int(patient["predicted_nez_count"]) == 1
    assert int(patient["predicted_ez_count"]) == 1
    assert list(channel["predicted_nez"]) == [1, 0]


def test_ledger_writer_rejects_misaligned_channel_fields_before_writing(tmp_path) -> None:
    record = {
        "subject_id": "hup:bad", "center": "hup", "canonical_channels": ["A1", "A2"],
        "channel_mask": [True], "labels_nez": [1.0], "labels_ez": [0.0],
        "predicted_nez_mask": [True], "predicted_ez_mask": [False],
    }
    with pytest.raises(ValueError, match="channel_mask.*hup:bad.*shape"):
        _write_ledgers([record], tmp_path, fold=1, seed=42)
    assert not list(tmp_path.iterdir())


def test_outer_only_fixed_threshold_uses_no_true_count() -> None:
    records = [{
        "subject_id": "lzu:test", "channel_mask": [True, True, False],
        "score_nez": [0.7, 0.3, 0.99], "true_nez_count": 0,
    }]
    predicted = apply_fixed_nez_probability_threshold(records, 0.5)[0]
    assert predicted["predicted_nez_mask"].tolist() == [True, False, False]
    assert predicted["predicted_ez_mask"].tolist() == [False, True, False]
    assert predicted["true_count_used_for_prediction"] is False
    assert predicted["oracle_threshold_used_for_prediction"] is False
    assert predicted["decision_rule"] == "fixed_nez_probability_threshold"


def test_validation_threshold_selection_maximizes_patient_macro_f1() -> None:
    records = [{
        "subject_id": "hup:validation", "center": "hup",
        "channel_mask": np.ones(4, dtype=bool),
        "labels_nez": np.array([0, 0, 1, 1], dtype=np.float32),
        "score_nez": np.array([0.10, 0.20, 0.60, 0.70], dtype=np.float32),
        "standardized_nez_logit": np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32),
    }]
    summary = ranking_validation_summary(records)
    assert summary["validation_patient_macro_f1"] == pytest.approx(1.0)
    assert 0.20 < summary["validation_selected_threshold"] <= 0.60


def test_center_balanced_sampler_honors_requested_batch_size() -> None:
    centers = [center for center in ("hup", "lzu", "multicenter", "pediatric") for _ in range(8)]
    sampler = CenterBalancedPatientBatchSampler(centers, seed=42, batch_size=16)
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 16
        assert len(set(batch)) == 16
        assert {center: sum(centers[index] == center for index in batch) for center in set(centers)} == {
            "hup": 4, "lzu": 4, "multicenter": 4, "pediatric": 4,
        }


def test_center_balanced_sampler_rejects_impossible_distinct_quota() -> None:
    centers = ["hup"] * 8 + ["lzu"] * 8 + ["multicenter"] * 8 + ["pediatric"] * 3
    with pytest.raises(ValueError, match="more distinct patients per center"):
        CenterBalancedPatientBatchSampler(centers, batch_size=16)


def test_sigmoid_logit_is_p_nez() -> None:
    batch = synthetic_batch(); out = cane_model()(batch)
    assert torch.allclose(out["score_nez"][batch["channel_mask"]], torch.sigmoid(out["final_nez_logit"])[batch["channel_mask"]])


def test_score_ez_is_one_minus_score_nez() -> None:
    out = cane_model()(synthetic_batch())
    assert torch.equal(out["score_ez"], 1-out["score_nez"])


def test_forward_does_not_read_label() -> None:
    model, batch = cane_model(), synthetic_batch()
    first = model(batch)["final_nez_logit"]
    changed = dict(batch); changed["labels_ez"] = torch.rand(4, 6); changed["true_nez_count"] = torch.arange(4)
    assert torch.equal(first, model(changed)["final_nez_logit"])


def test_contextual_embedding_returned() -> None:
    out = PatientChannelClassifier(8, num_heads=2, dropout=0)(torch.randn(2,3,8), torch.ones(2,3,dtype=torch.bool))
    assert out["contextual_channel_embedding"].shape == (2,3,8)


def test_invalid_contextual_embedding_zero() -> None:
    out = PatientChannelClassifier(4, num_heads=2, dropout=0)(torch.randn(1,2,4), torch.tensor([[1,0]],dtype=torch.bool))
    assert not out["contextual_channel_embedding"][0,1].any()


def test_prototype_normalize() -> None:
    anchor = _identity_anchor()
    with torch.no_grad(): anchor.prototypes.mul_(9)
    assert torch.allclose(anchor.normalized_prototypes.norm(dim=-1), torch.ones(2))


def test_near_prototype_distance_smaller() -> None:
    out = _identity_anchor()(torch.tensor([[[1.,0.],[-1.,0.]]]), torch.ones(1,2,dtype=torch.bool))
    assert out["anchor_distance"][0,0] < out["anchor_distance"][0,1]


def test_anchor_direction_monotonic() -> None:
    out = _identity_anchor()(torch.tensor([[[1.,0.],[.7,.7],[-1.,0.]]]), torch.ones(1,3,dtype=torch.bool))
    order = torch.argsort(out["anchor_nez_evidence"][0])
    assert torch.all(torch.diff(out["anchor_residual"][0,order]) >= -1e-7)


def test_anchor_residual_bounded() -> None:
    out = _identity_anchor()(torch.randn(2,10,2), torch.ones(2,10,dtype=torch.bool))
    assert out["anchor_residual"].abs().max().item() <= .200001


def test_anchor_initial_near_zero() -> None:
    out = _identity_anchor()(torch.randn(1,5,2), torch.ones(1,5,dtype=torch.bool))
    assert out["anchor_residual"].abs().max().item() < 1e-3


def test_compactness_clean_nez_only() -> None:
    d = torch.tensor([[.1,.9]], requires_grad=True); p=torch.eye(2)
    _, compact, _ = clean_nez_prototype_loss(d, torch.tensor([[1.,0.]]), torch.ones(1,2,dtype=torch.bool), p)
    assert compact.item() == pytest.approx(.1)


def test_observed_ez_excluded_from_compactness() -> None:
    d=torch.tensor([[.2,99.]]); _,compact,_=clean_nez_prototype_loss(d,torch.tensor([[1.,0.]]),torch.ones(1,2,dtype=torch.bool),torch.eye(2))
    assert compact.item() < 1


def test_prototype_diversity_penalizes_collapse() -> None:
    d=torch.ones(1,1); labels=torch.ones(1,1); mask=torch.ones(1,1,dtype=torch.bool)
    *_, collapsed=clean_nez_prototype_loss(d,labels,mask,torch.tensor([[1.,0.],[1.,0.]]))
    *_, diverse=clean_nez_prototype_loss(d,labels,mask,torch.tensor([[1.,0.],[0.,1.]]))
    assert collapsed > diverse


def test_seizure_output_shape() -> None:
    head=MultiSeizureNEZEvidenceResidual(8); out=head(torch.randn(2,3,4,8),torch.ones(2,3,dtype=torch.bool),torch.ones(2,3,4,dtype=torch.bool))
    assert out["seizure_residual"].shape == (2,4)


def test_seizure_invalid_mask() -> None:
    head=MultiSeizureNEZEvidenceResidual(8); mask=torch.zeros(1,2,3,dtype=torch.bool); out=head(torch.randn(1,2,3,8),torch.ones(1,2,dtype=torch.bool),mask)
    assert not out["seizure_residual"].any() and not out["seizure_nez_agreement"].any()


def test_seizure_residual_initial_zero() -> None:
    head=MultiSeizureNEZEvidenceResidual(8); out=head(torch.randn(1,2,3,8),torch.ones(1,2,dtype=torch.bool),torch.ones(1,2,3,dtype=torch.bool))
    assert not out["seizure_residual"].any()


def test_seizure_residual_bounded() -> None:
    head=MultiSeizureNEZEvidenceResidual(8,.2)
    with torch.no_grad(): head.evidence_residual_mlp[-1].bias.fill_(100)
    out=head(torch.randn(1,2,3,8),torch.ones(1,2,dtype=torch.bool),torch.ones(1,2,3,dtype=torch.bool))
    assert out["seizure_residual"].abs().max() <= .2


def test_seizure_agreement_range() -> None:
    head=MultiSeizureNEZEvidenceResidual(8); out=head(torch.randn(2,3,4,8),torch.ones(2,3,dtype=torch.bool),torch.ones(2,3,4,dtype=torch.bool))
    assert torch.all((out["seizure_nez_agreement"]>=0)&(out["seizure_nez_agreement"]<=1))


def test_no_valid_seizure_safe() -> None:
    head=MultiSeizureNEZEvidenceResidual(8); out=head(torch.randn(1,2,2,8),torch.zeros(1,2,dtype=torch.bool),torch.ones(1,2,2,dtype=torch.bool))
    assert all(torch.isfinite(v).all() for v in out.values())


def test_causal_residual_zero_initialization() -> None:
    head=CausalPropagationResidual(); out=head(torch.rand(2,3,6),torch.ones(2,3,dtype=torch.bool),torch.ones(2,3,dtype=torch.bool))
    assert not out["causal_residual"].any()


def test_causal_residual_bounded_and_invalid_zero() -> None:
    head=CausalPropagationResidual(.15)
    with torch.no_grad(): head.network[-1].bias.fill_(100)
    valid=torch.tensor([[1,0]],dtype=torch.bool); out=head(torch.rand(1,2,6),valid,torch.ones(1,2,dtype=torch.bool))
    assert out["causal_residual"][0,0] <= .15 and out["causal_residual"][0,1] == 0


def test_center_not_model_forward_argument() -> None:
    assert "center" not in inspect.signature(NeuroEZCModel._forward_cane_path_cp_nez).parameters


def test_cane_path_model_has_no_cardinality_head() -> None:
    assert not hasattr(cane_model(), "cardinality_head")


def test_p3_ablation_removes_anchor_from_logits_and_path_summary() -> None:
    model = NeuroEZCModel(cane_args("--cane-profile", "P3")).eval()
    out = model(synthetic_batch())
    assert not out["anchor_residual"].any() and not out["anchor_distance"].any() and not out["anchor_nez_evidence"].any()


def test_p4_ablation_removes_multiseizure_evidence_and_path_summary() -> None:
    model = NeuroEZCModel(cane_args("--cane-profile", "P4")).eval()
    out = model(synthetic_batch())
    assert not out["seizure_residual"].any() and not out["seizure_nez_probability_mean"].any() and not out["seizure_nez_agreement"].any()


def test_synthetic_forward_loss_backward_finite() -> None:
    model=NeuroEZCModel(cane_args()); batch=synthetic_batch(labels=True); out=model(batch)
    loss,parts=compute_cane_path_cp_ranking_loss(out,batch,cane_args(),epoch=16); loss.backward()
    assert torch.isfinite(loss) and parts["cane_path_stage"] == "stage3"
    groups=(model.b0_encoder,model.channel_classifier,model.clean_nez_anchor.projector,model.multi_seizure_evidence.seizure_nez_head,model.causal_propagation_residual.network)
    assert all(any(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters()) for group in groups)
