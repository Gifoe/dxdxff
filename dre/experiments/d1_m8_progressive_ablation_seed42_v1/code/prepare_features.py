"""Same pinned unfiltered Raw250 measurements; ABS14 and exact D1 DELTA14."""
import gc
import sys
from pathlib import Path
import numpy as np
from support import *

def run(a):
    lock,c,core,fac,trainer,records,rec,banks,binding=setup(a);pub=a.runtime/'results'
    paths=dict(raw=Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl'),window=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl'),archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'))
    for k,p in paths.items():assert sha(p)==lock['hashes'][k];print('INPUT_HASH_PASS',k,flush=True)
    ol=jread(BASE.parent/'PROTOCOL_LOCK.json')
    for rel,h in ol['source_hashes'].items():assert sha(a.source/rel)==h
    fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert jread(fp)==binding
    else:write(fp,binding)
    if (pub/'FEATURE_PARITY.json').exists():
        assert jread(pub/'FEATURE_PARITY.json')['binding']==binding
        assert jread(pub/'FEATURE_PARITY.json')['status']=='PASS';return
    sys.path.insert(0,str(a.source))
    original=load('progressive_pinned_aggregation',a.source/'task1_baselines/feature_aggregation.py')
    from neuroez_c.raw_brainbert_data import normalize_channel_name
    ex=Extractor(a.source,ol['source_hashes']['ez_features.py']);store=Store(a.baseline)
    lookup={tuple(k):i for i,k in enumerate(zip(rec['meta']['patient'],rec['meta']['channel']))}
    ledgers={q['patient']:q for q in rec['ledgers']}
    absolute=np.full_like(rec['R14'],np.nan);deltas=np.full_like(absolute,np.nan)
    windows=seizures=singletons=0;seen=[];maximum=0.;files={}
    for number,pid in enumerate(sorted(store.registry['plans']),1):
        plan=store.registry['plans'][pid];channels=plan['channels'];ledger=ledgers[pid]
        pp=a.runtime/f'features_private/{number:03d}.pkl'
        if pp.exists():
            q=read(pp);assert q['binding']==binding and q['raw_bank_sha256']==sha(Path(plan['path']))
        else:
            raw,mask,_,_=store.get(pid,channels,'cpu');raw=raw.numpy();mask=mask.numpy();S,C,W,_=raw.shape
            allf=np.full((S,C,W,len(ex.module.BASE_SPECTRAL_FEATURE_NAMES)),np.nan,np.float32)
            flat=np.flatnonzero(mask.reshape(-1));rv=raw.reshape(-1,500);af=allf.reshape(-1,allf.shape[-1])
            for start in range(0,len(flat),4096):
                ii=flat[start:start+4096];af[ii]=ex.all(rv[ii])
            norm={normalize_channel_name(ch):i for i,ch in enumerate(channels)}
            abmeans=[];dmeans=[];fullmeans=[]
            for s,r in enumerate(ledger['records']):
                t=np.asarray(r['times']);n=len(t);indices=[norm[name] for name in r['names']]
                v=allf[s,indices,:n].transpose(1,0,2).copy();m=mask[s,indices,:n].T.copy()
                assert np.array_equal(m,r['mask']) and np.array_equal(plan['times'][s,:n],t)
                seven=v[...,ex.indices];dr,pre,counts=window_delta(seven,t,m)
                assert np.array_equal(counts,r['baseline_counts']) and (counts>0).all()
                assert np.isfinite(seven[m]).all() and np.isfinite(dr[m]).all()
                aa=np.full((C,7),np.nan,np.float32);dd=aa.copy()
                for j,i in enumerate(indices):
                    aa[i]=seven[m[:,j],j].mean(0);dd[i]=dr[m[:,j],j].mean(0)
                src,names=historical_record_mean(original,v,list(ex.module.BASE_SPECTRAL_FEATURE_NAMES),t,m.any(1))
                ff=np.full((C,len(names)),np.nan,np.float32)
                for j,i in enumerate(indices):ff[i]=src[j]
                abmeans.append(aa);dmeans.append(dd);fullmeans.append(ff)
            abs14,ac=aggregate(abmeans);delta14,dc=aggregate(dmeans)
            full,_=aggregate(fullmeans);hj=[names.index('b0_delta__'+f) for f in FEATURES];full14=full[:,[k for j in hj for k in [2*j,2*j+1]]]
            ix=np.array([lookup[pid,ch] for ch in channels]);target=rec['R14'][ix]
            tol=8*np.abs(np.spacing(np.abs(target)))+2e-6*np.abs(target)
            assert np.all(np.abs(full14-target)<=tol) and np.all(np.abs(delta14-target)<=tol),'DELTA parity failed: stop before training'
            assert np.array_equal(delta14,target),'Exact seven-family parity required; no approximated substitution'
            assert np.array_equal(ac,dc) and np.all(ac>0)
            assert np.all(abs14[:,1::2][ac==1]==0) and np.all(delta14[:,1::2][dc==1]==0)
            assert np.isfinite(abs14).all() and np.isfinite(delta14).all()
            q=dict(binding=binding,raw_bank_sha256=sha(Path(plan['path'])),archive_indices=ix,ABS14=abs14,DELTA14=delta14,windows=int(mask.sum()),records=S,single_seizure_coordinates=int((ac==1).sum()))
            save(pp,q);del raw,mask,allf,rv,af
        ix=q['archive_indices'];seen.extend(ix.tolist());absolute[ix]=q['ABS14'];deltas[ix]=q['DELTA14']
        windows+=q['windows'];seizures+=q['records'];singletons+=q['single_seizure_coordinates']
        maximum=max(maximum,float(np.max(np.abs(q['DELTA14']-rec['R14'][ix]))));files[str(number)]=sha(pp)
        store.cache.clear();gc.collect();print('ABS_DELTA_PATIENT_VERIFIED',number,'of80',flush=True)
    assert sorted(seen)==list(range(7635)) and (windows,seizures)==(1471965,256)
    assert np.array_equal(deltas,rec['R14'])
    save(a.runtime/'FEATURES_PRIVATE.pkl',dict(binding=binding,ABS14=absolute,DELTA14=deltas,files=files))
    write(pub/'FEATURE_PARITY.json',dict(status='PASS',binding=binding,patients=80,channels=7635,records=256,genuine_channel_windows=windows,DELTA14_exact_original_R14=True,max_absolute_delta_error=maximum,source_masks_exact=True,single_seizure_zero_std_coordinates=singletons,no_labels_used_in_extraction=True,feature_file_sha256=sha(a.runtime/'FEATURES_PRIVATE.pkl'),features=lock['features']))
    write(pub/'RUN_STATUS.json',dict(status='FEATURE_GATES_PASS',training_cells_complete=0))

if __name__=='__main__':run(args())
