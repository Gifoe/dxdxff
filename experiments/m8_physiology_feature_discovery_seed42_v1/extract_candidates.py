"""Label-blind cached-signal extraction. Does not read model/test predictions.

The reconstruction metadata establishes identity and masks; no y field is used.
No actual patient/channel identity or path is written to public artifacts.
"""
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from features import extract,delta_summary,NAMES

TASK=Path(__file__).resolve().parent
SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
BASELINE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
OUT=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
sys.path.insert(0,str(SOURCE/'raw_delta_encoder_seed42_v1/code'))
from io_utils import sha,read,save,write
from raw_store import Store


def jr(p):return json.loads(Path(p).read_text())
lock=jr(TASK/'AUDIT_PROTOCOL.json');assert lock['candidate_names']==NAMES
registry_path=Path(r'D:\dgm_net_seed42_v1_runtime\CONDITIONING_REGISTRY_PRIVATE.pkl')
assert sha(registry_path)==lock['old_feature_registry_sha256']
oldreg=read(registry_path);rec=read(BASELINE/'RECONSTRUCTION_PRIVATE.pkl');meta=rec['meta'];store=Store(BASELINE)
gate=jr(BASELINE/'results/RAW_DYNAMIC_ALIGNMENT.json')
assert gate['status']=='PASS' and gate['valid_channel_windows']==1471965
assert gate['registry_sha256']==sha(BASELINE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')
assert rec['R14'].shape==(7635,14) and len(set(meta['patient']))==80
binding=dict(source={n:sha(TASK/n) for n in ['AUDIT_PROTOCOL.json','features.py','extract_candidates.py','test_features.py']},
    reconstruction_sha256=sha(BASELINE/'RECONSTRUCTION_PRIVATE.pkl'),conditioning_registry_sha256=sha(registry_path),
    raw_registry_sha256=sha(BASELINE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),raw_alignment_sha256=sha(BASELINE/'results/RAW_DYNAMIC_ALIGNMENT.json'),
    numpy=np.__version__,scipy=scipy.__version__)
OUT.mkdir(exist_ok=True);(OUT/'public').mkdir(exist_ok=True)
freeze=OUT/'CANDIDATE_INPUT_FREEZE_PRIVATE.json'
if freeze.exists():assert jr(freeze)==binding
else:write(freeze,binding)
result=np.full((7635,22),np.nan,np.float32);coverage=[];quality=[];files={};windows=seizures=0;parity=0.
lookup={tuple(k):i for i,k in enumerate(zip(meta['patient'],meta['channel']))}
for number,pid in enumerate(sorted(store.registry['plans']),1):
    plan=store.registry['plans'][pid];old=read(oldreg['files'][pid]['path'])
    assert old['raw_bank_sha256']==plan['sha256'] and old['patient']==pid and old['channels']==plan['channels']
    target=rec['R14'][old['archive_indices']];tol=8*np.abs(np.spacing(np.abs(target)))+2e-6*np.abs(target)
    assert np.all(np.abs(old['seven14']-target)<=tol)
    parity=max(parity,float(np.max(np.abs(old['seven14']-target))))
    path=OUT/f'candidate_windows_private/{number:03d}.pkl'
    if path.exists():q=read(path);assert q['binding']==binding and q['raw_bank_sha256']==plan['sha256']
    else:
        raw,mask,times,_=store.get(pid,plan['channels'],'cpu');raw=raw.numpy();mask=mask.numpy();times=times.numpy()
        S,C,W,L=raw.shape;assert L==500
        values=np.full((S,C,W,11),np.nan,np.float32);flat=np.flatnonzero(mask.reshape(-1))
        rv=raw.reshape(-1,500);vv=values.reshape(-1,11);fractions=[];constants=0
        for start in range(0,len(flat),2048):
            ix=flat[start:start+2048];v,qual=extract(rv[ix]);vv[ix]=v
            fractions.append(qual['rms_floor_fraction']);constants+=int(qual['constant'].sum())
        delta,summary=delta_summary(values,mask,times)
        archive=np.asarray([lookup[pid,ch] for ch in plan['channels']]);assert np.array_equal(archive,old['archive_indices'])
        q=dict(binding=binding,patient=pid,channels=plan['channels'],raw_bank_sha256=plan['sha256'],
            delta=delta,summary=summary,archive_indices=archive,windows=int(mask.sum()),seizures=S,
            finite_fraction=float(np.isfinite(values[mask]).mean()),constant_windows=constants,
            rms_floor_fraction=np.concatenate(fractions),names=NAMES,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        save(path,q);del raw,mask,times,values,rv,vv,delta
    result[q['archive_indices']]=q['summary'];windows+=q['windows'];seizures+=q['seizures']
    center=np.unique(meta['center'][q['archive_indices']]);assert len(center)==1
    coverage.append(dict(center=str(center[0]),channels=len(q['archive_indices']),seizures=q['seizures'],windows=q['windows'],
        finite_fraction=q['finite_fraction'],constant_windows=q['constant_windows']))
    quality.append((str(center[0]),q['rms_floor_fraction']))
    files[pid]=dict(path=str(path),sha256=sha(path),channels=q['channels'],archive_indices=q['archive_indices'])
    write(OUT/'public/CANDIDATE_EXTRACTION_STATUS.json',dict(status='RUNNING',completed_patients=number,total_patients=80,
        completed_valid_windows=windows,no_label_input_or_prediction_read=True))
    print('LABEL_BLIND_CANDIDATE_PATIENT_COMPLETE',number,'of80',flush=True)
    store.cache.clear();gc.collect()
assert (len(files),seizures,windows)==(80,256,1471965) and np.isfinite(result).all()
save(OUT/'CANDIDATE_FEATURES_PRIVATE.pkl',dict(binding=binding,original14=rec['R14'],candidate22=result,names=NAMES,
    patient=meta['patient'],channel=meta['channel'],center=meta['center']))
save(OUT/'CANDIDATE_REGISTRY_PRIVATE.pkl',dict(binding=binding,files=files))
cover=pd.DataFrame(coverage);summary=[]
for center,g in [('ALL',cover)]+list(cover.groupby('center')):
    summary.append(dict(center=center,patients=len(g),channels=int(g.channels.sum()),seizures=int(g.seizures.sum()),
        valid_channel_windows=int(g.windows.sum()),finite_fraction=float(np.average(g.finite_fraction,weights=g.windows)),
        constant_valid_windows=int(g.constant_windows.sum()),no_patient_or_channel_exclusion=True))
pd.DataFrame(summary).to_csv(OUT/'public/CANDIDATE_COVERAGE.csv',index=False)
stats=[]
for center in ['ALL']+sorted(set(c for c,_ in quality)):
    arrays=[v for c,v in quality if center=='ALL' or c==center];v=np.concatenate(arrays)
    stats.append(dict(center=center,patients=len(arrays),valid_windows=len(v),pooled_q10=float(np.quantile(v,.1)),
        pooled_median=float(np.median(v)),pooled_q90=float(np.quantile(v,.9)),
        fraction_floor_ge_half=float((v>=.5).mean()),fraction_floor_ge_90percent=float((v>=.9).mean()),
        median_patient_median=float(np.median([np.median(a) for a in arrays])),threshold_fixed_before_extraction=True))
pd.DataFrame(stats).to_csv(OUT/'public/OLD_RMS_NUMERICAL_FLOOR_AUDIT.csv',index=False)
write(OUT/'public/CANDIDATE_EXTRACTION_STATUS.json',dict(status='ALL11_CANDIDATES_EXTRACTED_FULL_COHORT',patients=80,channels=7635,
    seizures=256,valid_raw_channel_windows=1471965,candidate_window_dimensions=11,candidate_summary_dimensions=22,
    finite_fraction=1.,old7_actual_delta_to_R14_max_error=parity,source=binding['source'],
    feature_cache_sha256=sha(OUT/'CANDIDATE_FEATURES_PRIVATE.pkl'),no_labels_used=True,no_outer_scores_or_outcomes_read=True,
    efficacy_not_yet_evaluated=True,full_goal_complete=False))
print('ALL11_CANDIDATES_EXTRACTED_FULL_COHORT',flush=True)
