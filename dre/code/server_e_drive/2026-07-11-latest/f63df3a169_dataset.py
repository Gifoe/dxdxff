from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Sequence
from neuroez_c.dataset import build_patient_examples,collate_patient_ez_batch,fit_window_tensor_normalizer
from neuroez_c.evidence_views import PRUNED_SPECTRAL_FEATURE_NAMES

PHYSICS_FEATURE_NAMES=("early_high_gamma_slope","early_line_length_slope","onset_latency_high_gamma","onset_latency_line_length","onset_rank_high_gamma","onset_rank_line_length","high_gamma_top20pct_mean","line_length_top20pct_mean","hfo80_150_event_rate","hfo80_150_duration_fraction","hfo80_150_mean_envelope_z","hfo80_150_max_envelope_z")
FEATURE_ARGS=dict(positive_label="nez",b0_feature_parts="abs,delta,zdelta,ratio",b0_feature_groups="spectral_classical",physics_feature_parts="abs",physics_state_features=",".join(PHYSICS_FEATURE_NAMES),use_diffusion_residual=False,use_edf_quality_weighting=False)
def feature_args(**overrides): return SimpleNamespace(**(FEATURE_ARGS|overrides))
def validate_feature_names(samples):
    available=None
    for sample in samples:
        names=sample.get("window_feature_names")
        if names is None: raise ValueError(f"Task 1 requires cache window_feature_names. requested={list(PHYSICS_FEATURE_NAMES)}, available=None, missing={list(PHYSICS_FEATURE_NAMES)}")
        names=list(map(str,names)); missing=[x for x in PHYSICS_FEATURE_NAMES if x not in names]
        if missing: raise ValueError(f"Missing required Task 1 physics features. requested={list(PHYSICS_FEATURE_NAMES)}, available={names}, missing={missing}")
        available=names
    if available is None: raise ValueError("Cannot validate Task 1 features: no samples")
    return available
def feature_audit(samples):
    available=validate_feature_names(samples); parts=("abs","delta","zdelta","ratio")
    return {"b0_feature_names":list(PRUNED_SPECTRAL_FEATURE_NAMES),"b0_feature_parts":list(parts),"b0_input_dim":len(PRUNED_SPECTRAL_FEATURE_NAMES)*4,"physics_feature_names":list(PHYSICS_FEATURE_NAMES),"physics_feature_parts":["abs"],"physics_input_dim":12,"expected_physics_dim":12,"actual_physics_dim":12,"all_required_physics_features_present":True,"available_window_feature_names":available,"neural_fragility":False}
def audit_window_time_coverage(samples,strict=True):
    out={"n_samples":len(samples),"n_with_pre_onset":0,"n_without_pre_onset":0,"n_with_post_onset":0,"n_without_post_onset":0,"n_nonfinite_centers":0,"n_nonmonotonic_centers":0,"n_center_length_mismatch":0,"subjects_without_pre_onset":[],"subjects_without_post_onset":[],"samples_without_pre_onset":[],"samples_without_post_onset":[],"samples_nonmonotonic":[],"samples_length_mismatch":[]}
    for x in samples:
      sid=str(x.get("subject_id")); rid=str(x.get("sample_id",x.get("run_id"))); c=x.get("window_relative_centers_sec"); t=int(x["window_features"].shape[0]); key=f"{sid}:{rid}"
      if c is None or len(c)!=t: out["n_center_length_mismatch"]+=1;out["samples_length_mismatch"].append(key);continue
      import numpy as np
      c=np.asarray(c,float)
      if not np.isfinite(c).all():out["n_nonfinite_centers"]+=1
      if np.any(np.diff(c)<0):out["n_nonmonotonic_centers"]+=1;out["samples_nonmonotonic"].append(key)
      pre=bool((c<0).any()); post=bool((c>=0).any()); out["n_with_pre_onset"]+=pre;out["n_without_pre_onset"]+=not pre;out["n_with_post_onset"]+=post;out["n_without_post_onset"]+=not post
      if not pre:out["subjects_without_pre_onset"].append(sid);out["samples_without_pre_onset"].append(key)
      if not post:out["subjects_without_post_onset"].append(sid);out["samples_without_post_onset"].append(key)
    out["pre_onset_coverage_rate"]=out["n_with_pre_onset"]/max(1,len(samples));out["post_onset_coverage_rate"]=out["n_with_post_onset"]/max(1,len(samples))
    if strict and (out["n_without_pre_onset"] or out["n_nonfinite_centers"] or out["n_nonmonotonic_centers"] or out["n_center_length_mismatch"]):raise ValueError(f"Task 1 window center audit failed: {out}")
    return out
def build_examples(samples:Sequence[dict[str,Any]],patient_index,*,normalizer=None,subject_ids=None,args=None):
    validate_feature_names(samples); args=args or feature_args(); args.positive_label="nez"
    result=build_patient_examples(samples,patient_index,normalizer=normalizer,subject_ids=subject_ids,args=args)
    if any(x.shape[-1]!=12 for e in result for x in e["physics_features"]): raise AssertionError("Task 1 actual_physics_dim must equal 12")
    return result
__all__=["PHYSICS_FEATURE_NAMES","build_examples","collate_patient_ez_batch","feature_args","feature_audit","fit_window_tensor_normalizer","validate_feature_names","audit_window_time_coverage"]
