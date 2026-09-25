"""Build the 32-dimension Task 1 Clean-NEZ cache from feature + raw caches."""
from __future__ import annotations
import argparse, csv, json, pickle, sys
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from ez_features import (_clinical_onset_core_features,_burstness_features,_hfo_lite_window_features,
 CLINICAL_ONSET_CORE_FEATURE_NAMES,BURSTNESS_FEATURE_NAMES,HFO_LITE_FEATURE_NAMES,WINDOW_NODE_FEATURE_NAMES)

TASK1_NAMES=("early_high_gamma_slope","early_line_length_slope","onset_latency_high_gamma","onset_latency_line_length","onset_rank_high_gamma","onset_rank_line_length","high_gamma_top20pct_mean","line_length_top20pct_mean","hfo80_150_event_rate","hfo80_150_duration_fraction","hfo80_150_mean_envelope_z","hfo80_150_max_envelope_z")

def _load(path):
 with open(path,"rb") as f:return pickle.load(f)
def _ledger(path):
 with open(path,newline="",encoding="utf-8-sig") as f:return {str(r["subject_id"]) for r in csv.DictReader(f)}
def _sample(r): return r.get("sample",{}) if isinstance(r.get("sample"),dict) else {}
def _value(r,key): return _sample(r).get(key,r.get(key))
def _compatible(feature,raw):
 fs,rs=_sample(feature),_sample(raw); fw=np.asarray(fs["window_features"]); rw=np.asarray(_value(raw,"raw_waveform"))
 return (list(feature.get("channel_names_norm",[]))==list(raw.get("channel_names_norm",[])) and np.array_equal(np.asarray(feature.get("labels")),np.asarray(raw.get("labels"))) and np.asarray(fs.get("window_relative_centers_sec")).shape==np.asarray(rs.get("window_relative_centers_sec",fs.get("window_relative_centers_sec"))).shape and rw.ndim==2 and rw.shape[0]==fw.shape[1])
def _equivalent(candidates):
 first=candidates[0]; fields=("channel_names_norm","labels")
 for r in candidates[1:]:
  if any(not np.array_equal(np.asarray(first.get(x)),np.asarray(r.get(x))) for x in fields):return False
  for x in ("raw_temporal_sfreq","raw_valid_samples","raw_valid_start_sample"):
   if _value(first,x)!=_value(r,x):return False
  if not np.array_equal(np.asarray(_value(first,"raw_waveform")),np.asarray(_value(r,"raw_waveform"))):return False
 return True
def resolve_raw_record(feature,candidates):
 compatible=[r for r in candidates if _compatible(feature,r)]
 if not compatible: raise ValueError(f"No compatible raw record for {(feature.get('subject_id'),feature.get('run_id'))}")
 for key in ("sample_id","source_seizure_id","seizure_onset_sec"):
  v=_value(feature,key); same=[r for r in compatible if v is not None and _value(r,key)==v]
  if len(same)==1:return same[0],False
  if same:compatible=same
 if len(compatible)==1:return compatible[0],False
 if _equivalent(compatible):return compatible[0],True
 info=[{"sample_id":_value(r,"sample_id"),"source_seizure_id":_value(r,"source_seizure_id"),"raw_shape":list(np.asarray(_value(r,"raw_waveform")).shape),"raw_sfreq":_value(r,"raw_temporal_sfreq")} for r in compatible]
 raise ValueError(f"Ambiguous non-equivalent raw records for subject_id={feature.get('subject_id')}, run_id={feature.get('run_id')}: {info}")
def _hfo(raw, sfreq, t):
 values=_hfo_lite_window_features(raw,float(sfreq))[:,:4]
 return np.broadcast_to(values[None],(t,*values.shape)).copy()
def enrich(feature,raw):
 s=_sample(feature); base=np.asarray(s["window_features"],np.float32); centers=np.asarray(s["window_relative_centers_sec"],np.float32); clinical=_clinical_onset_core_features(base,centers); burst=_burstness_features(base,centers)
 names=list(s.get("window_feature_names",WINDOW_NODE_FEATURE_NAMES)); idx={n:i for i,n in enumerate(CLINICAL_ONSET_CORE_FEATURE_NAMES)}
 clinical_pick=clinical[:,:, [idx[x] for x in TASK1_NAMES[:6]]]; burst_idx={n:i for i,n in enumerate(BURSTNESS_FEATURE_NAMES)}; burst_pick=burst[:,:, [burst_idx[x] for x in TASK1_NAMES[6:8]]]
 raw_wave=np.asarray(_value(raw,"raw_waveform"),np.float32); sfreq=float(_value(raw,"raw_temporal_sfreq") or _value(raw,"sfreq") or 0); hfo=_hfo(raw_wave,sfreq,base.shape[0]); extra=np.concatenate([clinical_pick,burst_pick,hfo],axis=-1).astype(np.float32)
 if extra.shape[-1]!=12:raise RuntimeError(f"Expected 12 Task 1 features, got {extra.shape[-1]}")
 out=dict(feature); os=dict(s); os["window_features"]=np.concatenate([base,extra],axis=-1); os["window_feature_names"]=names+list(TASK1_NAMES); os["task1_clean_nez_enriched"]=True; out["sample"]=os
 return out,float(np.count_nonzero(hfo)/max(1,hfo.size))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--feature-cache-path",required=True);p.add_argument("--raw-cache-path",required=True);p.add_argument("--allowed-subjects-ledger",required=True);p.add_argument("--require-n-patients",type=int,default=90);p.add_argument("--output-cache-path",required=True);p.add_argument("--audit-path",required=True);p.add_argument("--overwrite",action="store_true");a=p.parse_args()
 out=Path(a.output_cache_path); audit=Path(a.audit_path)
 if (out.exists() or audit.exists()) and not a.overwrite:raise FileExistsError("Output exists; pass --overwrite")
 allowed=_ledger(a.allowed_subjects_ledger); fp,rp=_load(a.feature_cache_path),_load(a.raw_cache_path); feature=[r for r in fp["run_records"] if str(r.get("subject_id")) in allowed]; index={k:v for k,v in fp["patient_index"].items() if str(k) in allowed}
 if len(index)!=a.require_n_patients:raise ValueError(f"Expected {a.require_n_patients} ledger patients, found {len(index)}")
 raw_index={};
 for r in rp["run_records"]:
  if str(r.get("subject_id")) in allowed:raw_index.setdefault((str(r.get("subject_id")),str(r.get("run_id"))),[]).append(r)
 duplicate=sum(len(v)>1 for v in raw_index.values()); records=[]; rows=[]; equivalent=0
 for fr in feature:
  rr,eq=resolve_raw_record(fr,raw_index.get((str(fr.get("subject_id")),str(fr.get("run_id"))),[])); equivalent+=eq; enriched,hfo_frac=enrich(fr,rr); records.append(enriched); s=enriched["sample"];rows.append({"subject_id":str(fr["subject_id"]),"run_id":str(fr["run_id"]),"n_windows":int(s["window_features"].shape[0]),"n_channels":int(s["window_features"].shape[1]),"raw_sfreq":_value(rr,"raw_temporal_sfreq"),"hfo_nonzero_fraction":hfo_frac})
 if any(r["sample"]["window_features"].shape[-1]!=32 or not np.isfinite(r["sample"]["window_features"]).all() for r in records):raise RuntimeError("Non-finite or non-32-dimensional enriched features")
 out.parent.mkdir(parents=True,exist_ok=True); audit.parent.mkdir(parents=True,exist_ok=True)
 with open(out,"wb") as f:pickle.dump({**fp,"run_records":records,"patient_index":index,"cache_version":"task1_clean_nez_v1"},f,protocol=pickle.HIGHEST_PROTOCOL)
 report={"n_patients":len(index),"n_run_records":len(records),"input_feature_dim":20,"task1_feature_dim":12,"output_feature_dim":32,"n_duplicate_raw_keys":duplicate,"n_records_with_equivalent_raw_duplicate_resolved":equivalent,"task1_feature_names":list(TASK1_NAMES),"all_finite":True,"records":rows}
 audit.write_text(json.dumps(report,indent=2),encoding="utf-8")
if __name__=="__main__":main()
