from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import exp_ez_hybrid
from data_factory import build_outer_splits
from exp_ez_hybrid import _clone_ema_teacher, _select_n6_validation_threshold, _update_ema_teacher
from neuroez_c.dual_view_data import RawAlignmentStore
from neuroez_c.model import NeuroEZCModel
from neuroez_c.n6_dualview_ema_loss import (
    compute_n6_total_loss,
    compute_patient_clean_nez_robust_loss,
    compute_patient_ema_rank_loss,
    ema_observed_ez_reliability,
)
from neuroez_c.protocol import STEP4B_N6_DUALVIEW_EMA_PROTOCOL, STEP4B_STATIC_TOP20_FEATURES, assert_fixed_all90_protocol
from neuroez_c.raw_window_encoder import RawWindowEncoder
from run_neuroez_c import build_parser, validate_n6_args


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_step4b_n6_dualview_ema_all90.ps1"


def _model_args(**updates: object) -> SimpleNamespace:
    values = dict(model_dim=8, num_heads=2, dropout=0.0, positive_label="nez", use_channel_attention=True, use_patient_relative_z=True,
                  use_physics_dynamics=False, use_diffusion_residual=False, use_negative_anchor_head=False, use_view_gated_fusion=False,
                  use_two_expert_router=False, use_feature_separated_two_expert=False, use_a9v8_lcbo=False, use_broad_ez_mil_loss=False,
                  group_robust_mode="none", use_n6_dual_view_ema=True, temporal_pooling="mean", channel_pooling_mode="mean", record_pooling="mean",
                  raw_encoder_dropout=0.2, n6_initial_feature_gate=0.75, n6_warmup_epochs=0, n6_noise_discount=0.5, n6_reliability_min=0.5,
                  n6_feature_aux_weight=0.15, n6_raw_aux_weight=0.15, n6_rank_loss_weight=0.05, n6_rank_margin=0.1, n6_gate_anchor=0.7, n6_gate_loss_weight=.005)
    values.update(updates)
    return SimpleNamespace(**values)


def _batch() -> dict[str, torch.Tensor]:
    b, s, w, c, f, t = 2, 2, 3, 4, 6, 64
    batch = {"b0_features": torch.randn(b, s, w, c, f), "physics_features": torch.randn(b, s, w, c, 2),
             "window_centers": torch.zeros(b, s, w), "window_mask": torch.ones(b, s, w, dtype=torch.bool),
             "channel_mask": torch.ones(b, c, dtype=torch.bool), "seizure_mask": torch.ones(b, s, dtype=torch.bool),
             "seizure_channel_mask": torch.ones(b, s, c, dtype=torch.bool), "raw_windows": torch.randn(b, s, c, w, t),
             "raw_window_mask": torch.ones(b, s, c, w, dtype=torch.bool), "raw_seizure_channel_mask": torch.ones(b, s, c, dtype=torch.bool),
             "labels_ez": torch.tensor([[0., 1., 0., 1.], [0., 1., 1., 0.]])}
    return batch


def _record(subject: str, run: str, onset: float, raw: bool = False) -> dict:
    sample = {"sample_id": f"{run}-sample", "seizure_onset_sec": onset, "start_sec": onset - 30, "end_sec": onset + 30,
              "window_relative_centers_sec": [-1.0, 1.0], "feature_scale_used_secs": [2.0, 2.0]}
    if raw:
        sample.update(raw_waveform=torch.ones((2, 15000), dtype=torch.float32).numpy(), raw_temporal_sfreq=250.0, raw_temporal_duration_sec=60.0)
    return {"subject_id": subject, "run_id": run, "channel_names_norm": ["A1", "A2"], "sample": sample}


def _store(tmp_path: Path, feature: list[dict], raw: list[dict]) -> RawAlignmentStore:
    raw_path = tmp_path / "raw.pkl"
    with raw_path.open("wb") as handle:
        pickle.dump({"run_records": raw}, handle)
    return RawAlignmentStore(feature, feature_cache_path="feature.pkl", raw_cache_path=raw_path, raw_target_samples=500, raw_target_sampling_rate=250.0)


def test_raw_feature_alignment_uses_compound_key(tmp_path: Path) -> None:
    feature, raw = _record("p1", "r1", 100), _record("p1", "r1", 100, raw=True)
    values, mask, available = _store(tmp_path, [feature], [raw]).aligned_windows(feature)
    assert values.shape == (2, 2, 500) and mask.all() and available.all()


def test_raw_alignment_rejects_duplicate_keys(tmp_path: Path) -> None:
    record = _record("p1", "r1", 100, raw=True)
    with pytest.raises(ValueError, match="Duplicate raw"):
        _store(tmp_path, [_record("p1", "r1", 100)], [record, record])


def test_raw_alignment_rejects_cross_patient_match(tmp_path: Path) -> None:
    store = _store(tmp_path, [_record("p1", "r1", 100)], [_record("p2", "r1", 100, raw=True)])
    with pytest.raises(ValueError, match="matched patients"):
        store.assert_formal_coverage(min_channel_match_rate=.95, min_window_match_rate=.9, expected_patients=1)


def test_raw_alignment_rejects_cross_seizure_match(tmp_path: Path) -> None:
    store = _store(tmp_path, [_record("p1", "r1", 100)], [_record("p1", "r2", 100, raw=True)])
    assert store.aligned_windows(_record("p1", "r1", 100))[1].sum() == 0


def test_raw_encoder_output_shape() -> None:
    encoder = RawWindowEncoder(model_dim=8)
    output = encoder(torch.randn(2, 2, 3, 4, 64), torch.ones(2, 2, 3, 4, dtype=torch.bool))
    assert output.shape == (2, 2, 3, 4, 8)


def test_raw_encoder_masks_invalid_windows() -> None:
    encoder = RawWindowEncoder(model_dim=8)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool); mask[..., 1, 1] = False
    assert torch.equal(encoder(torch.randn(1, 1, 2, 2, 64), mask)[..., 1, 1, :], torch.zeros(1, 1, 8))


def test_missing_raw_channel_forces_feature_gate() -> None:
    batch = _batch(); batch["raw_window_mask"][0, :, 3] = False; batch["raw_seizure_channel_mask"][0, :, 3] = False
    output = NeuroEZCModel(_model_args())(batch)
    assert output["fusion_gate_feature"][0, 3] == 1 and torch.allclose(output["logits"][0, 3], output["logits_feature"][0, 3])


def test_fusion_gate_is_bounded() -> None:
    output = NeuroEZCModel(_model_args())(_batch())
    gate = output["fusion_gate_feature"].detach()
    assert float(gate.min()) >= .05 and float(gate.max()) <= .95


def test_fusion_gate_initializes_to_feature_preference() -> None:
    output = NeuroEZCModel(_model_args())(_batch())
    assert torch.allclose(output["fusion_gate_feature"], torch.full_like(output["fusion_gate_feature"], .75), atol=1e-5)


def test_fused_logit_matches_formula() -> None:
    output = NeuroEZCModel(_model_args())(_batch())
    gate = output["fusion_gate_feature"]
    assert torch.allclose(output["logits"], gate * output["logits_feature"] + (1 - gate) * output["logits_raw"])


def test_clean_nez_has_full_positive_weight() -> None:
    logits, labels, mask = torch.tensor([[0.2, -1.0]]), torch.tensor([[0., 1.]]), torch.ones(1, 2, dtype=torch.bool)
    first, _ = compute_patient_clean_nez_robust_loss(logits, labels, mask, torch.tensor([[.1, .5]]))
    second, _ = compute_patient_clean_nez_robust_loss(logits, labels, mask, torch.tensor([[1., .5]]))
    assert torch.allclose(first, second)


def test_observed_ez_reliability_is_bounded() -> None:
    rel = ema_observed_ez_reliability(torch.tensor([[-20., 20.]]), torch.tensor([[1., 1.]]), torch.ones(1, 2, dtype=torch.bool), epoch=6, warmup_epochs=5, noise_discount=.5, reliability_min=.5)
    assert float(rel.min()) >= .5 and float(rel.max()) <= 1


def test_warmup_reliability_equals_one() -> None:
    rel = ema_observed_ez_reliability(torch.randn(1, 2), torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), epoch=5, warmup_epochs=5, noise_discount=.5, reliability_min=.5)
    assert torch.equal(rel, torch.ones_like(rel))


def test_observed_ez_uses_fixed_count_denominator() -> None:
    logits, labels, mask = torch.tensor([[-1., -1., -1.]]), torch.tensor([[1., 1., 1.]]), torch.ones(1, 3, dtype=torch.bool)
    low, _ = compute_patient_clean_nez_robust_loss(logits, labels, mask, torch.full_like(logits, .5))
    high, _ = compute_patient_clean_nez_robust_loss(logits, labels, mask, torch.ones_like(logits))
    assert low < high


def test_patient_loss_is_patient_balanced() -> None:
    logits = torch.tensor([[2., -1., 0., 0.], [2., -1., 2., -1.]])
    labels = torch.tensor([[0., 1., -1., -1.], [0., 1., 0., 1.]])
    loss, _ = compute_patient_clean_nez_robust_loss(logits, labels, labels >= 0, torch.ones_like(logits))
    compact, _ = compute_patient_clean_nez_robust_loss(torch.tensor([[2., -1.], [2., -1.]]), torch.tensor([[0., 1.], [0., 1.]]), torch.ones(2, 2, dtype=torch.bool), torch.ones(2, 2))
    assert torch.allclose(loss, compact)


def test_rank_uses_fixed_pair_denominator() -> None:
    labels = torch.tensor([[0., 1.]])
    logits = torch.tensor([[-10., 4.]])
    high, _ = compute_patient_ema_rank_loss(logits, labels, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[1., 1.]]), margin=.1)
    low, _ = compute_patient_ema_rank_loss(logits, labels, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[1., .5]]), margin=.1)
    assert low < high


def test_rank_loss_decreases_when_clean_nez_logit_increases() -> None:
    labels = torch.tensor([[0., 1.]])
    low, _ = compute_patient_ema_rank_loss(torch.tensor([[0., -1.]]), labels, torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2), margin=.1)
    high, _ = compute_patient_ema_rank_loss(torch.tensor([[3., -1.]]), labels, torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2), margin=.1)
    assert high < low


def test_ema_teacher_has_no_grad() -> None:
    teacher = _clone_ema_teacher(torch.nn.Linear(2, 1))
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_ema_update_matches_formula() -> None:
    student, teacher = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    student.weight.data.fill_(3); teacher.weight.data.fill_(1); _update_ema_teacher(teacher, student, .5)
    assert teacher.weight.item() == pytest.approx(2.)


def test_dual_view_backward_and_ema_update_are_finite() -> None:
    args = _model_args()
    student = NeuroEZCModel(args)
    batch = _batch()
    # One channel has no raw view, exercising forced feature fallback too.
    batch["raw_window_mask"][0, :, 3] = False
    # The production loop initializes lazy B0 layers before cloning EMA.
    _ = student(batch)
    teacher = _clone_ema_teacher(student)
    output = student(batch)
    with torch.no_grad():
        teacher_output = teacher(batch)
    loss, _ = compute_n6_total_loss(output, teacher_output, batch, args, epoch=6)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in student.b0_encoder.parameters())
    assert student.raw_window_encoder.stem[0].weight.grad is not None
    assert student.fusion_gate[-1].weight.grad is not None
    _update_ema_teacher(teacher, student, .995)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_teacher_does_not_enter_optimizer() -> None:
    student, teacher = torch.nn.Linear(2, 1), _clone_ema_teacher(torch.nn.Linear(2, 1))
    opt = torch.optim.AdamW(student.parameters())
    ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert not ids.intersection({id(p) for p in teacher.parameters()})


def test_balanced_auprc_hmean_penalizes_single_class_failure() -> None:
    args = SimpleNamespace(early_stop_metric="balanced_patient_auprc_hmean")
    assert exp_ez_hybrid._summary_score({"patient_macro_auprc_nez": 1., "patient_macro_auprc_ez": 0.}, args) == 0.


def test_threshold_selector_uses_validation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[set[str]] = []
    def summarize(records, classification_threshold):
        seen.append({r["subject_id"] for r in records}); score = 1 - abs(classification_threshold - .5)
        return {"patient_macro_f1": score, "patient_macro_balanced_accuracy": score, "patient_macro_nez_f1": score, "patient_macro_ez_f1": score}, list(records)
    monkeypatch.setattr(exp_ez_hybrid, "_summarize_prediction_records", summarize)
    _select_n6_validation_threshold([{"subject_id": "validation", "score_nez": torch.tensor([.4]).numpy(), "channel_mask": torch.tensor([True]).numpy()}])
    assert seen and all(rows == {"validation"} for rows in seen)


def test_final_runner_uses_raw_and_feature_caches() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "RawCachePath" in text and "--raw_window_cache_path" in text and "--use_n6_dual_view_ema" in text


def test_final_runner_contains_no_old_n5f_args() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "use_n5_" + "final_nez_pu" not in text and "n5f_" not in text


def test_protocol_forbids_center_input_and_requires_90(tmp_path: Path) -> None:
    raw_audit = tmp_path / "raw.json"; raw_audit.write_text(json.dumps({"status": "passed", "n_matched_patients": 90, "channel_match_rate": 1., "window_match_rate": 1., "raw_patient_coverage": 1.}))
    patients = {f"p{i:03d}": {} for i in range(90)}; splits = build_outer_splits(patients, split_strategy="5fold", n_splits=5, random_seed=42)
    args = SimpleNamespace(fixed_all90_protocol_name=STEP4B_N6_DUALVIEW_EMA_PROTOCOL, fixed_all90_cache_audit_path="", dual_view_cache_audit_path=str(raw_audit), positive_label="nez", score_semantics="nez_probability", split_strategy="5fold", n_splits=5, random_seed=42, drop_high_ez_fraction_lzu=False, require_n_patients=90, physics_state_features=",".join(STEP4B_STATIC_TOP20_FEATURES), physics_feature_parts="abs", use_physics_dynamics=True, use_channel_attention=True, use_patient_relative_z=True, group_robust_mode="none", use_diffusion_residual=False, use_ez_ranking_loss=False, use_hard_topk_loss=False, use_negative_anchor_head=False, use_two_expert_router=False, use_feature_separated_two_expert=False, use_broad_ez_mil_loss=False, use_a9v8_lcbo=False, use_teacher_anchor_eval=False, teacher_anchor_apply_to_train_loss=False, use_view_gated_fusion=False, use_edf_quality_weighting=False, train_subject_dropout_file="", train_subject_dropout_count=0, early_stop_metric="balanced_patient_auprc_hmean", loss_mode="n6_dualview_ema_robust", use_n6_dual_view_ema=True, raw_min_channel_match_rate=.95, raw_min_window_match_rate=.9, n6_ema_decay=.995, n6_warmup_epochs=5, n6_noise_discount=.5, n6_reliability_min=.5, n6_feature_aux_weight=.15, n6_raw_aux_weight=.15, n6_rank_loss_weight=.05, n6_rank_margin=.1, n6_gate_anchor=.7, n6_gate_loss_weight=.005, raw_window_cache_path="raw", window_cache_path="feature", raw_target_samples=500, raw_target_sampling_rate=250.)
    assert assert_fixed_all90_protocol(args, patients, splits, cache_feature_names=list(STEP4B_STATIC_TOP20_FEATURES))["method"] == "N6F_NEZ_DualView_EMA_RobustRank"
    args.model_input_features = "center_id"
    with pytest.raises(ValueError, match="center_id"):
        assert_fixed_all90_protocol(args, patients, splits, cache_feature_names=list(STEP4B_STATIC_TOP20_FEATURES))


def test_parser_n6_validation() -> None:
    args = build_parser().parse_args(["--use_n6_dual_view_ema", "--raw_window_cache_path", "raw.pkl"])
    validate_n6_args(args)
    assert args.loss_mode == "n6_dualview_ema_robust"
