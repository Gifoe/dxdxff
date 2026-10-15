"""Same-source stable log-power controls; preserve all earlier discovery freezes."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from features import delta_summary

TASK=Path(__file__).resolve().parent;SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
BASELINE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime');OUT=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
sys.path.insert(0,str(SOURCE/'raw_delta_encoder_seed42_v1/code'))
from io_utils import sha,read,save,write
from raw_store import Store
from raw_physiological_features import Extractor
def jr(p):return json.loads(Path(p).read_text())
lock=jr(TASK/'NUMERICAL_CONTROLS_LOCK.json');store=Store(BASELINE);rec=read(BASELINE/'RECONSTRUCTION_PRIVATE.pkl')
regpath=Path(r'D:\dgm_net_seed42_v1_runtime\CONDITIONING_REGISTRY_PRIVATE.pkl');assert sha(regpath)==lock['old_registry_sha256']
oldreg=read(regpath);source_sha=oldreg['binding']['inputs']['ez_features.py'] if 'ez_features.py' in oldreg['binding'].get('inputs',{}) else None
# Exact source pin is also recorded in the original raw-extraction protocol.
original=jr(SOURCE/'raw_delta_encoder_seed42_v1/PROTOCOL_LOCK.json')
expected=original['source_hashes']['ez_features.py'];extractor=Extractor(SOURCE,expected)
assert extractor.module.BASE_SPECTRAL_FEATURE_NAMES[extractor.indices[0]]=='log_bp_beta'
binding=dict(source={n:sha(TASK/n) for n in ['NUMERICAL_CONTROLS_LOCK.json','extract_numeric_controls.py','features.py']},
    raw_registry_sha256=sha(BASELINE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),old_registry_sha256=sha(regpath),ez_features_sha256=expected)
freeze=OUT/'NUMERICAL_CONTROLS_INPUT_FREEZE_PRIVATE.json'
if freeze.exists():assert jr(freeze)==binding
else:write(freeze,binding)
summary=np.full((7635,6),np.nan,np.float32);files={};counts=[0,0,0];max_error=0.;max_tolerance=0.
for number,pid in enumerate(sorted(store.registry['plans']),1):
    plan=store.registry['plans'][pid];old=read(oldreg['files'][pid]['path']);path=OUT/f'numerical_control_windows_private/{number:03d}.pkl'
    if path.exists():q=read(path);assert q['binding']==binding and q['raw_bank_sha256']==plan['sha256']
    else:
        raw,mask,times,_=store.get(pid,plan['channels'],'cpu');raw=raw.numpy();mask=mask.numpy();times=times.numpy()
        values=np.full((*mask.shape,3),np.nan,np.float32);legacy=values.copy();flat=np.flatnonzero(mask.ravel())
        rv=raw.reshape(-1,500);v=values.reshape(-1,3);oldv=legacy.reshape(-1,3)
        for start in range(0,len(flat),4096):
            ix=flat[start:start+4096];base=extractor(rv[ix])[:,:3];oldv[ix]=base
            v[ix]=np.log(np.maximum(np.expm1(base.astype(float)),1e-30)).astype(np.float32)
        delta,vector=delta_summary(values,mask,times);old_delta,_=delta_summary(legacy,mask,times)
        tol=float(8*np.spacing(np.max(np.abs(legacy[mask]))))+1e-30
        error=float(np.max(np.abs(old_delta[mask]-old['delta'][mask,:3])))
        assert error<=tol,'Original same-source legacy replay failed'
        q=dict(binding=binding,patient=pid,channels=plan['channels'],archive_indices=old['archive_indices'],raw_bank_sha256=plan['sha256'],
            names=lock['names'],delta=delta,summary=vector,windows=int(mask.sum()),seizures=len(mask),
            finite_fraction=float(np.isfinite(values[mask]).mean()),legacy_delta_max_error=error,legacy_delta_registered_tolerance=tol)
        save(path,q);del raw,mask,times,values,legacy,rv,v,oldv,delta,old_delta
    summary[q['archive_indices']]=q['summary'];counts[0]+=1;counts[1]+=q['seizures'];counts[2]+=q['windows']
    max_error=max(max_error,q['legacy_delta_max_error']);max_tolerance=max(max_tolerance,q['legacy_delta_registered_tolerance'])
    files[pid]=dict(path=str(path),sha256=sha(path));store.cache.clear();gc.collect()
    print('NUMERIC_CONTROL_PATIENT_COMPLETE',number,'of80',flush=True)
assert counts==[80,256,1471965] and np.isfinite(summary).all()
save(OUT/'NUMERICAL_CONTROL_FEATURES_PRIVATE.pkl',dict(binding=binding,names=lock['names'],summary6=summary))
save(OUT/'NUMERICAL_CONTROL_REGISTRY_PRIVATE.pkl',dict(binding=binding,files=files))
write(OUT/'public/NUMERICAL_CONTROL_EXTRACTION_STATUS.json',dict(status='STABLE_LOG3_NUMERICAL_CONTROLS_COMPLETE',patients=80,channels=7635,
    seizures=256,valid_windows=1471965,finite_fraction=1.,legacy_replay_max_error=max_error,
    largest_registered_patient_tolerance=max_tolerance,no_filters_or_band_edge_changes=True,no_labels_or_Outer_read=True,
    source=binding['source'],ez_features_sha256=expected,feature_sha256=sha(OUT/'NUMERICAL_CONTROL_FEATURES_PRIVATE.pkl'),full_goal_complete=False))
print('STABLE_LOG3_NUMERICAL_CONTROLS_COMPLETE',flush=True)
