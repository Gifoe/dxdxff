"""Recompute label-blind window physiology; exact original full-view R14 layout."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from runtime_support import BASE_CODE,sha,read,save,write,load,Store,Extractor,FEATURES,window_delta,aggregate,historical_record_mean,context,make_base,original_modules
from controls import permutations

def prepare(a):
    root=Path(__file__).parents[1]; pub=a.runtime/'results'; pub.mkdir(parents=True,exist_ok=True)
    lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    original_lock=json.loads((BASE_CODE.parent/'PROTOCOL_LOCK.json').read_text())
    paths=dict(raw=Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl'),window=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl'),archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'))
    binding=dict(protocol=sha(root/'PROTOCOL_LOCK.json'),baseline_protocol=sha(BASE_CODE.parent/'PROTOCOL_LOCK.json'),reconstruction=sha(a.baseline/'RECONSTRUCTION_PRIVATE.pkl'),raw_registry=sha(a.baseline/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),D1_freeze=sha(a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json'),code={p.name:sha(p) for p in (root/'code').glob('*.py')},inputs={})
    for name,p in paths.items():
        binding['inputs'][name]=sha(p); assert binding['inputs'][name]==lock['hashes'][name]
        print('INPUT_HASH_PASS',name,flush=True)
    for rel,h in original_lock['source_hashes'].items():assert sha(a.source/rel)==h
    frozen=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if frozen.exists():assert json.loads(frozen.read_text())==binding, 'Immutable experiment input/code binding changed'
    else:write(frozen,binding)
    if (pub/'WINDOW_DELTA_PARITY.json').exists():
        done=json.loads((pub/'WINDOW_DELTA_PARITY.json').read_text()); assert done['status']=='PASS' and done['binding']==binding
        print('PREPARATION_ALREADY_VERIFIED',flush=True);return
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline)
    store=Store(a.baseline); frozen_d1=json.loads((a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text()); metrics,_=original_modules(); prs=[]; parity=[]
    for fold in range(1,6):
        row=next(q for q in frozen_d1['selections'] if q['arm']=='D1' and q['fold']==fold)
        cp=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt'; bp=cp.parent.parent/'BANK_PRIVATE.pt'
        assert sha(cp)==row['checkpoint_sha256'] and sha(bp)==row['bank_sha256']
        d=torch.load(bp,map_location='cpu',weights_only=False); b=make_base(core,fac,records,cp,42+1009*fold).cuda().eval()
        with torch.inference_mode():ll=b(torch.as_tensor(d['x'][d['val']],device='cuda')); scores=torch.sigmoid(ll).cpu().numpy()
        old=read(a.baseline/f'scores/D1_fold{fold}_PRIVATE.pkl'); drift=float(np.max(np.abs(scores-old['score'])))
        assert drift<=1e-6 and np.array_equal(scores>=row['threshold'],old['score']>=old['threshold'])
        assert core.select_threshold(d['y'][d['val']],scores,d['patient'][d['val']])['threshold']==row['threshold']
        prs.extend(metrics.patient_rows(c,d,scores,row['threshold'],'A0',fold))
        parity.append(dict(fold=fold,checkpoint_sha256=sha(cp),bank_sha256=sha(bp),score_max_drift=drift,decisions_exact=True))
        del b,d
    mean=pd.DataFrame(prs)[metrics.METRICS].mean(); assert np.allclose(mean[['ez_auroc','ez_auprc','macro_f1','ez_f1']],[.725978,.503601,.644546,.443537],rtol=0,atol=5.1e-7)
    save(a.runtime/'A0_DEVELOPMENT_PRIVATE.pkl',prs); write(pub/'D1_PARITY.json',dict(status='PASS',folds=parity,metrics=mean.to_dict(),D1_retrained=False))
    import sys
    sys.path.insert(0,str(a.source))
    original=load('dgm_pinned_aggregation',a.source/'task1_baselines/feature_aggregation.py')
    extractor=Extractor(a.source,original_lock['source_hashes']['ez_features.py']); ledger_by={q['patient']:q for q in rec['ledgers']}; outputs={}; windows=seizures=0; maxdiff=0.; perm_counts=dict(eligible=0,changed=0,singletons=0)
    reconstructed=np.full_like(rec['R14'],np.nan)
    for number,pid in enumerate(sorted(store.registry['plans']),1):
        plan=store.registry['plans'][pid]; ledger=ledger_by[pid]; channels=plan['channels']; rp=a.runtime/'conditioning_private'/f'{number:03d}.pkl'
        if rp.exists():
            q=read(rp); assert q['binding']==binding and q['raw_bank_sha256']==sha(Path(plan['path']))
        else:
            raw,mask,times,_=store.get(pid,channels,'cpu'); raw=raw.numpy(); mask=mask.numpy(); S,C,W,_=raw.shape
            allf=np.full((S,C,W,len(extractor.module.BASE_SPECTRAL_FEATURE_NAMES)),np.nan,np.float32)
            flat=np.flatnonzero(mask.reshape(-1)); rv=raw.reshape(-1,500); af=allf.reshape(-1,allf.shape[-1])
            for start in range(0,len(flat),4096):
                ix=flat[start:start+4096]; af[ix]=extractor.all(rv[ix])
            delta=np.zeros((S,C,W,7),np.float32); means=[]; source_means=[]; raw_to_seven_max=0.; normchannels={str(n):i for i,n in enumerate(channels)}
            # Ledger names are normalized; map with the pinned canonical normalizer.
            import sys; sys.path.insert(0,str(a.source))
            from neuroez_c.raw_brainbert_data import normalize_channel_name
            normchannels={normalize_channel_name(n):i for i,n in enumerate(channels)}
            for s,r in enumerate(ledger['records']):
                t=np.asarray(r['times']); n=len(t); indices=[normchannels[name] for name in r['names']]
                afull=allf[s,indices,:n].transpose(1,0,2).copy(); m=mask[s,indices,:n].T.copy()
                assert np.array_equal(m,r['mask'])
                seven=afull[...,extractor.indices]; dr,pre,counts=window_delta(seven,t,m); assert np.array_equal(counts,r['baseline_counts']) and (counts>0).all()
                assert np.isfinite(dr[m]).all()
                for j,i in enumerate(indices):delta[s,i,:n]=np.nan_to_num(dr[:,j],nan=0.)
                sr,names=historical_record_mean(original,afull,list(extractor.module.BASE_SPECTRAL_FEATURE_NAMES),t,m.any(1))
                hj=[names.index('b0_delta__'+f) for f in FEATURES]
                rr=np.full((C,len(names)),np.nan,np.float32); naive=np.full((C,7),np.nan,np.float32)
                for j,i in enumerate(indices):
                    rr[i]=sr[j]; naive[i]=dr[m[:,j],j].mean(0)
                raw_to_seven_max=max(raw_to_seven_max,float(np.nanmax(np.abs(rr[:,hj]-naive))))
                means.append(rr);source_means.append(naive)
            out,_=aggregate(means); pick=[k for j in hj for k in [2*j,2*j+1]]; r14=out[:,pick]
            # Also gate reaggregation of the actual seven-dimensional tensors used by A2.
            seven14,_=aggregate(source_means)
            ix=np.array([list(zip(rec['meta']['patient'],rec['meta']['channel'])).index((pid,ch)) for ch in channels])
            target=rec['R14'][ix]; tol=8*np.abs(np.spacing(np.abs(target)))+2e-6*np.abs(target)
            assert np.all(np.abs(r14-target)<=tol), 'Pinned full-view R14 reconstruction failed'
            assert np.all(np.abs(seven14-target)<=tol), 'New actual window DELTA cannot reaggregate to original R14'
            # Original registry has padded times; permutations must use the complete W shape.
            perm_plans=[dict(times=plan['times'][s],compound_key=r['compound_key']) for s,r in enumerate(ledger['records'])]
            perms,pa=permutations(pid,channels,perm_plans,mask)
            for s,p in enumerate(perms):assert np.array_equal(np.take_along_axis(mask[s],p,axis=1),mask[s])
            q=dict(binding=binding,patient=pid,channels=channels,raw_bank_sha256=sha(Path(plan['path'])),delta=delta,R14=r14,seven14=seven14,archive_indices=ix,windows=int(mask.sum()),records=S,perms=perms,permutation_audit=pa,full_view_vs_seven_mean_max=raw_to_seven_max)
            save(rp,q)
            del raw,mask,allf,af,rv
        reconstructed[q['archive_indices']]=q['R14']; windows+=q['windows'];seizures+=q['records']
        for k in perm_counts:perm_counts[k]+=q['permutation_audit'][k]
        maxdiff=max(maxdiff,float(np.max(np.abs(q['seven14']-rec['R14'][q['archive_indices']]))))
        outputs[pid]=dict(path=str(rp),sha256=sha(rp),channels=channels,raw_bank_sha256=q['raw_bank_sha256'])
        store.cache.clear(); gc.collect(); print('WINDOW_DELTA_VERIFIED',number,'of80',flush=True)
    assert (len(outputs),seizures,windows)==(80,256,1471965) and np.isfinite(reconstructed).all()
    registry=dict(binding=binding,files=outputs); save(a.runtime/'CONDITIONING_REGISTRY_PRIVATE.pkl',registry)
    scale_audit=[]
    for fold,d in enumerate(banks,1):
        ps=[]; fit=sorted(set(d['patient'][d['train']].astype(str)))
        for pid in fit:
            q=read(Path(outputs[pid]['path'])); b=torch.load(store.registry['plans'][pid]['path'],map_location='cpu',weights_only=False); valid=b['mask'][0].numpy()
            ps.append(np.median(np.abs(q['delta'][valid]),axis=0)); del b
        ps=np.stack(ps); typical=np.median(ps,axis=0); scale=np.maximum(typical,1e-6).astype(np.float32)
        save(a.runtime/f'scale_fold{fold}_PRIVATE.pkl',dict(binding=binding,scale=scale,fit_patients=fit,patient_magnitudes=ps))
        validation_ids=sorted(set(d['patient'][d['val']].astype(str)))
        for scope,ids in [('FIT',fit),('Development',validation_ids)]:
            frac=[]; quant=[]
            for pid in ids:
                q=read(Path(outputs[pid]['path'])); b=torch.load(store.registry['plans'][pid]['path'],map_location='cpu',weights_only=False); v=q['delta'][b['mask'][0].numpy()]/scale
                frac.append((np.abs(v)>8).mean(0)); quant.append(np.quantile(v,[.25,.5,.75],axis=0));del b
            for k,f in enumerate(FEATURES):
                scale_audit.append(dict(fold=fold,scope=scope,feature=f,patients=len(ids),typical_abs_magnitude=float(typical[k]),scale=float(scale[k]),floor_applied=bool(typical[k]<1e-6),patient_equal_clipped_fraction=float(np.mean(frac,0)[k]),median_patient_q25=float(np.median(quant,0)[0,k]),median_patient_q50=float(np.median(quant,0)[1,k]),median_patient_q75=float(np.median(quant,0)[2,k]),fitted_on='FIT only, no labels'))
    pd.DataFrame(scale_audit).to_csv(pub/'FEATURE_SCALE_AUDIT.csv',index=False)
    old_alignment=json.loads((a.baseline/'results/RAW_DYNAMIC_ALIGNMENT.json').read_text()); assert old_alignment['status']=='PASS'
    write(pub/'RAW_ALIGNMENT.json',dict(status='PASS',patients=80,records=256,channels=7635,valid_channel_windows=windows,new_checks='All current raw bank hashes matched original every-sample verified registry; compound channel/mask/reference identities rechecked during recomputation',historical_exact_sample_audit_sha256=sha(a.baseline/'results/RAW_DYNAMIC_ALIGNMENT.json'),new_source_hashes=binding['inputs'],new_inference_Outer=False))
    write(pub/'CONDITIONING_ALIGNMENT.json',dict(status='PASS',same_window_raw_and_phys=True,shuffle_counts=perm_counts,scale_fit_only=True,padded_windows_excluded=True,full_vector_shuffled=True))
    write(pub/'WINDOW_DELTA_PARITY.json',dict(status='PASS',binding=binding,patients=80,records=256,channels=7635,valid_channel_windows=windows,maximum_actual_7D_reaggregation_error=maxdiff,original_tolerance_unchanged=True,no_labels_used_in_extraction=True,registry_sha256=sha(a.runtime/'CONDITIONING_REGISTRY_PRIVATE.pkl')))
    write(pub/'RUN_STATUS.json',dict(status='PRETRAINING_DATA_IDENTITY_COMPLETE',models_trained=0,requested_arm='A2',Outer_accessed=False))
    print('PRETRAINING_DATA_GATES_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--source',type=Path,required=True);prepare(p.parse_args())
