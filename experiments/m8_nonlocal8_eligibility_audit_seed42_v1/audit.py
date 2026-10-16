"""Unlabeled nonlocal descriptor eligibility; no model or outcome modules."""
import ast
import gc
import hashlib
import pickle
import sys
import warnings
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
BASE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
PARENT=Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
CACHE=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl')
ROOT=Path(r'D:\m8_nonlocal8_eligibility_audit_seed42_v1_runtime');PUB=ROOT/'public'
sys.path.insert(0,str(SOURCE/'raw_delta_encoder_seed42_v1/code'))
from io_utils import read,save,sha,write,load
from nonlocal_helpers import NAMES,pure_functions,isolated,different,diagnostics


def norm(value):return ''.join(str(value).upper().split()).replace('-','').replace('_','')


def freeze(path,value):
    import json
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==value
    else:write(path,value)


def array_sha(a):
    a=np.ascontiguousarray(a)
    assert a.dtype.kind!='O'
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()


def native_geometry(record,field,t):
    keys=['raw_temporal_sfreq','raw_temporal_duration_sec','seizure_onset_sec','start_sec',
          'raw_valid_start_sample','raw_valid_samples','feature_scale_used_secs']
    value={k:field(record,k,None) for k in keys}
    missing=[k for k,v in value.items() if v is None]
    if missing:return None,dict(status='MISSING_METADATA',missing_fields=';'.join(missing))
    fs=float(value['raw_temporal_sfreq']);duration=float(value['raw_temporal_duration_sec'])
    if not np.isfinite(fs) or fs<=0:return None,dict(status='INVALID_SAMPLING_METADATA')
    left=float(value['raw_valid_start_sample'])/fs;end=left+float(value['raw_valid_samples'])/fs
    onset_position=float(value['seizure_onset_sec'])-float(value['start_sec'])+left
    widths=np.asarray(value['feature_scale_used_secs'],float)
    sampling=float(record.get('sfreq',fs));tolerance=max(1/fs,1/sampling)+1e-5
    valid=fs>0 and 0<=left<end<=duration+1/fs and widths.shape==t.shape and np.all(widths==2.)
    valid=valid and np.isfinite(onset_position) and abs(onset_position-duration/2)<=tolerance
    if not valid:return None,dict(status='INVALID_GEOMETRY',feature_sfreq=fs)
    mask=(t+onset_position-widths/2>=left-1e-6)&(t+onset_position+widths/2<=end+1e-6)
    return mask,dict(status='METADATA_GEOMETRY_DERIVED_NOT_CLINICAL_VERIFICATION',feature_sfreq=fs,
        onset_position_seconds=onset_position,left_padding_seconds=left,valid_end_seconds=end,
        invalid_windows=int((~mask).sum()))


def distributions(values):
    result=[]
    for (mode,center),blocks in values.items():
        a=np.concatenate(blocks,axis=0)
        for j,name in enumerate(NAMES):
            finite=np.isfinite(a[:,j]);v=a[finite,j]
            result.append(dict(mode=mode,center=center,feature=name,record_channel_units=len(a),
                finite_units=int(finite.sum()),finite_rate=float(finite.mean()),
                min=float(v.min()) if len(v) else None,q25=float(np.quantile(v,.25)) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,q75=float(np.quantile(v,.75)) if len(v) else None,
                max=float(v.max()) if len(v) else None,mean=float(v.mean()) if len(v) else None,
                inferential_independent_sample_count=False))
    return pd.DataFrame(result)


def main():
    import json
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    for relative,expected in lock['source_hashes'].items():assert sha(SOURCE/relative)==expected
    rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');registry=read(BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')
    poolreg=read(PARENT/'POOL_REGISTRY_PRIVATE.pkl');poolbinding=json.loads((PARENT/'POOL_INPUT_FREEZE_PRIVATE.json').read_text())
    assert sha(BASE/'RECONSTRUCTION_PRIVATE.pkl')==poolbinding['reconstruction']
    assert sha(BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')==poolbinding['raw_registry']
    assert sha(CACHE)==rec['binding']['window']
    assert poolreg['binding']['ez_features']==lock['source_hashes']['ez_features.py']
    assert len(rec['ledgers'])==len(registry['plans'])==len(poolreg['files'])==80
    assert sum(len(r['channels']) for r in rec['ledgers'])==7635
    assert sum(len(r['records']) for r in rec['ledgers'])==256
    bound=dict(source={n:sha(HERE/n) for n in ['PROTOCOL_LOCK.json','audit.py','nonlocal_helpers.py']},
        measurement_sources=lock['source_hashes'],record_accessor_source_sha256=sha(SOURCE/'neuroez_c/dual_view_data.py'),feature_cache_sha256=sha(CACHE),
        reconstruction_sha256=sha(BASE/'RECONSTRUCTION_PRIVATE.pkl'),pool_registry_sha256=sha(PARENT/'POOL_REGISTRY_PRIVATE.pkl'),
        raw_registry_sha256=sha(BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),
        split_array_sha256={k:array_sha(rec['meta'][k]) for k in ['split_patient','split_role','split_fold']})
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',bound)
    if (PUB/'AUDIT_STATUS.json').exists():
        assert json.loads((PUB/'AUDIT_STATUS.json').read_text())['input_freeze_sha256']==sha(ROOT/'INPUT_FREEZE_PRIVATE.json')
        print('NONLOCAL8_AUDIT_ALREADY_COMPLETE_NO_RECOMPUTATION',flush=True);return
    ez=load('nonlocal8_pinned_ez_features',SOURCE/'ez_features.py')
    # Only pure key/field accessors and the pure derivation function are compiled.
    access=pure_functions(SOURCE/'neuroez_c/dual_view_data.py',['_sample','_field','_record_key'],dict(np=np,Mapping=Mapping))
    builder=pure_functions(SOURCE/'prepare_step4b_static_top20_all90_cache.py',['derive_step4b_static_top20'],
        dict(np=np,WINDOW_NODE_FEATURE_NAMES=ez.WINDOW_NODE_FEATURE_NAMES,STEP4B_STATIC_TOP20_FEATURES=tuple(NAMES),
             CLINICAL_ONSET_CORE_FEATURE_NAMES=ez.CLINICAL_ONSET_CORE_FEATURE_NAMES,BURSTNESS_FEATURE_NAMES=ez.BURSTNESS_FEATURE_NAMES,
             _clinical_onset_core_features=ez._clinical_onset_core_features,_burstness_features=ez._burstness_features))
    derive=builder['derive_step4b_static_top20'];field=access['_field'];record_key=access['_record_key']
    wanted={tuple(item['compound_key']) for patient in rec['ledgers'] for item in patient['records']}
    assert len(wanted)==256
    with CACHE.open('rb') as stream:payload=pickle.load(stream)
    lookup={}
    for value in payload['run_records']:
        key=record_key(value)
        if key in wanted:
            assert key not in lookup
            lookup[key]=value
    assert set(lookup)==wanted
    # Whitelist numeric configuration only; never read patient_index/label values.
    config={};containers=[payload]+[payload[k] for k in ['args','config','build_config','metadata'] if k in payload]
    fields=['spectral_min_freq','spectral_max_freq','raw_target_sampling_rate','raw_temporal_sfreq']
    for name in fields:
        found=[]
        for container in containers:
            value=container.get(name) if isinstance(container,Mapping) else getattr(container,name,None)
            if isinstance(value,(float,int,np.number)) and np.isfinite(value):found.append(float(value))
        config[name]=sorted(set(found)) if found else 'UNKNOWN_NOT_EMBEDDED'
    write(PUB/'CACHE_CONFIG_METADATA.json',dict(numeric_fields=config,
        source_function_hash_is_not_cache_creation_provenance=True,exact_creation_CLI='UNKNOWN',
        cache_creation_source_SHA='UNKNOWN',acquisition_filter='UNKNOWN',independent_clinical_onset=False))
    # Record rows remain private; public export is feature/source aggregates only.
    private=[];differences=[];diag=[];coverage=[];values={};geometry=[];rates=Counter();units=windows=record_windows=0
    for number,ledger in enumerate(rec['ledgers'],1):
        pid=ledger['patient'];entry=poolreg['files'][pid];assert sha(entry['path'])==entry['sha256']
        q=read(Path(entry['path']));assert q['raw_bank_sha256']==registry['plans'][pid]['sha256']
        assert q['binding']==poolreg['binding']
        assert list(q['channels'])==list(ledger['channels'])==list(registry['plans'][pid]['channels'])
        centers=np.unique(rec['meta']['center'][rec['meta']['patient']==pid]);assert len(centers)==1;center=str(centers[0])
        channel_lookup={norm(c):i for i,c in enumerate(q['channels'])};assert len(channel_lookup)==len(q['channels'])
        for seizure,item in enumerate(ledger['records']):
            value=lookup[tuple(item['compound_key'])];sample=value['sample'];names=list(map(str,sample['window_feature_names']))
            assert len(names)==len(set(names))==28
            x=np.asarray(sample['window_features']);assert x.dtype==np.float32
            t=np.asarray(field(value,'window_relative_centers_sec'),np.float32)
            assert np.array_equal(t,np.asarray(item['times'],np.float32)) and np.all(np.diff(t)>0)
            source_names=list(field(value,'channel_names_norm'));assert list(map(norm,source_names))==list(map(norm,item['names']))
            c=len(source_names);mask=np.asarray(item['mask']);assert mask.shape==(len(t),c)
            assert x.shape==(*mask.shape,28) and len(set(map(norm,source_names)))==c
            ix=np.array([channel_lookup[norm(n)] for n in source_names])
            assert np.array_equal(mask,q['mask'][seizure,ix,:len(t)].T)
            assert np.array_equal(t,q['times'][seizure,:len(t)].astype(np.float32))
            uniform=np.array_equal(mask,np.broadcast_to(mask[:,:1],mask.shape))
            all_base=np.take(x,[names.index(n) for n in ez.WINDOW_NODE_FEATURE_NAMES],axis=2)
            stored=np.take(x,[names.index(n) for n in NAMES],axis=2)
            expected=derive(all_base,t)
            hg=all_base[:,:,list(ez.WINDOW_NODE_FEATURE_NAMES).index('log_bp_high_gamma')]
            ll=all_base[:,:,list(ez.WINDOW_NODE_FEATURE_NAMES).index('line_length_per_sec')]
            scalar=isolated(ez,hg,ll,t)
            assert np.array_equal(expected,np.broadcast_to(scalar[None],expected.shape))
            current=stored[0];replay_error=different(expected,stored)
            constant=np.all(stored==stored[:1],axis=(0,1))
            native,native_meta=native_geometry(value,field,t.astype(float));geometry.append(native_meta)
            if 'feature_sfreq' in native_meta:rates[native_meta['feature_sfreq']]+=1
            rawmode=np.full((c,8),np.nan,np.float32);cache_raw=rawmode.copy();cache_native=rawmode.copy()
            if uniform:
                valid=mask[:,0]
                if valid.any() and (t[valid]<0).any() and (t[valid]>=0).any():
                    cache_raw=isolated(ez,hg[valid],ll[valid],t[valid])
                    rh=q['values'][seizure,ix,:len(t),2].T;rl=q['values'][seizure,ix,:len(t),5].T
                    assert np.isfinite(rh[valid]).all() and np.isfinite(rl[valid]).all()
                    rawmode=isolated(ez,rh[valid],rl[valid],t[valid])
                if native is not None and native.any() and (t[native]<0).any() and (t[native]>=0).any():
                    cache_native=isolated(ez,hg[native],ll[native],t[native])
                for mode,h,l,tt in [('CACHE_ALL',hg,ll,t),('RAW250_VALID',
                        q['values'][seizure,ix,:len(t),2].T[valid],q['values'][seizure,ix,:len(t),5].T[valid],t[valid])]:
                    if not len(tt):continue
                    for feature,series in [('high_gamma',h),('line_length',l)]:
                        info=diagnostics(ez,series,tt)
                        assert not info['unsupported'],'Missing genuine pre/post support; do not invoke a silent baseline fallback'
                        diag.append(dict(mode=mode,center=center,feature=feature,**info))
                    original=isolated(ez,h,l,tt);permuted=isolated(ez,h[:,::-1],l[:,::-1],tt)[::-1]
                    bad=different(permuted,original)
                    for j,name in enumerate(NAMES):
                        differences.append(dict(mode=mode,center=center,feature=name,units=c,
                            permutation_violating_units=int(bad[:,j].sum()),
                            permutation_max_abs=float(np.max(np.abs(permuted[:,j].astype(float)-original[:,j])))))
            for mode,a in [('CACHE_STORED',current),('CACHE_RAW_VALID_MASK',cache_raw),
                           ('CACHE_NATIVE_VALID_MASK',cache_native),('RAW250_VALID',rawmode)]:
                values.setdefault((mode,center),[]).append(a)
                values.setdefault((mode,'ALL'),[]).append(a)
            for j,name in enumerate(NAMES):
                raw_changed=different(cache_raw[:,j],current[:,j])
                native_changed=different(cache_native[:,j],current[:,j])
                pre=mask&(t[:,None]<0)
                count=pre.sum(0);mean=np.divide((stored[:,:,j]*pre).sum(0),count,out=np.zeros(c,np.float32),where=count>0)
                numerical_delta=(stored[:,:,j]-mean[None])[mask]
                paired=np.isfinite(rawmode[:,j])&np.isfinite(cache_raw[:,j])
                correlation=np.nan
                if paired.sum()>2 and np.ptp(rawmode[paired,j])>0 and np.ptp(cache_raw[paired,j])>0:
                    correlation=float(spearmanr(rawmode[paired,j],cache_raw[paired,j]).statistic)
                private.append(dict(patient=pid,seizure=seizure,center=center,feature=name,units=c,
                    stored=current[:,j],cached_formula=expected[0,:,j],cached_Raw_mask=cache_raw[:,j],cached_native_mask=cache_native[:,j],
                    Raw250_mask=rawmode[:,j],broadcast_exact=bool(constant[j]),formula_exact=bool(np.array_equal(expected[:,:,j],stored[:,:,j])),
                    formula_violating_units=int(replay_error[:,:,j].any(0).sum()),
                    formula_max_abs=float(np.max(np.abs(expected[:,:,j].astype(float)-stored[:,:,j]))),
                    Raw_mask_changed_units=int(raw_changed.sum()),native_mask_changed_units=int(native_changed.sum()),
                    valid_Delta_roundoff_max=float(np.max(np.abs(numerical_delta))) if len(numerical_delta) else None,
                    Raw250_vs_cached_spearman=correlation,correlation_not_source_equivalence=True))
            units+=c;windows+=int(mask.sum());record_windows+=int(mask.any(1).sum())
            coverage.append(dict(center=center,record_channel_units=c,valid_Raw_channel_windows=int(mask.sum()),
                valid_Raw_record_windows=int(mask.any(1).sum()),raw_invalid_record_windows=int((~mask.any(1)).sum()),
                uniform_channel_mask=bool(uniform),native_geometry_available=native is not None,
                native_invalid_record_windows=int((~native).sum()) if native is not None else 0,
                native_vs_Raw_mask_disagreement_record_windows=int((native!=mask[:,0]).sum()) if native is not None and uniform else 0,
                valid_pre_windows=int((mask.any(1)&(t<0)).sum()),valid_post_windows=int((mask.any(1)&(t>=0)).sum())))
        write(PUB/'AUDIT_PROGRESS.json',dict(status='UNLABELED_SOURCE_REPLAY',patients=number,total_patients=80))
        print('UNLABELED_NONLOCAL8_PATIENT_COMPLETE',number,'of80',flush=True)
        del q;gc.collect()
    assert len(coverage)==256 and units==24995 and windows==1471965 and record_windows==15074
    save(ROOT/'RECORD_DIAGNOSTICS_PRIVATE.pkl',dict(binding=bound,records=private,coverage=coverage,geometry=geometry,latency=diag,permutation=differences))
    records=pd.DataFrame([{k:v for k,v in row.items() if k not in ['patient','seizure','stored','cached_formula','cached_Raw_mask','cached_native_mask','Raw250_mask']} for row in private])
    aggregate=[]
    for group in [['feature'],['center','feature']]:
        for key,g in records.groupby(group):
            if not isinstance(key,tuple):key=(key,)
            aggregate.append(dict(dict(zip(group,key)),scope='overall' if len(group)==1 else 'source',records=len(g),
                units=int(g.units.sum()),broadcast_all_exact=bool(g.broadcast_exact.all()),formula_exact_records=int(g.formula_exact.sum()),
                formula_violating_units=int(g.formula_violating_units.sum()),formula_max_abs=float(g.formula_max_abs.max()),
                Raw_mask_changed_units=int(g.Raw_mask_changed_units.sum()),native_mask_changed_units=int(g.native_mask_changed_units.sum()),
                valid_Delta_roundoff_max=float(g.valid_Delta_roundoff_max.max()),
                median_record_Raw250_cached_spearman=float(g.Raw250_vs_cached_spearman.median()),
                finite_correlation_records=int(g.Raw250_vs_cached_spearman.notna().sum())))
    pd.DataFrame(aggregate).to_csv(PUB/'FORMULA_MASK_SUMMARY.csv',index=False)
    distribution_table=distributions(values)
    distribution_table.to_csv(PUB/'DESCRIPTOR_DISTRIBUTIONS.csv',index=False)
    perm=pd.DataFrame(differences)
    pp=[]
    for keys in [['mode','feature'],['mode','center','feature']]:
        for key,g in perm.groupby(keys):
            pp.append(dict(dict(zip(keys,key)),scope='overall' if len(keys)==2 else 'source',records=len(g),
                units=int(g.units.sum()),permutation_violating_units=int(g.permutation_violating_units.sum()),
                permutation_max_abs=float(g.permutation_max_abs.max())))
    pd.DataFrame(pp).to_csv(PUB/'PERMUTATION_EQUIVARIANCE.csv',index=False)
    pd.DataFrame(diag).groupby(['mode','center','feature'],as_index=False).agg(
        records=('channels','size'),channels=('channels','sum'),zero_pre_std_channels=('zero_pre_std_channels','sum'),
        no_crossing_channels=('no_crossing_channels','sum'),tied_latency_channels=('tied_latency_channels','sum'),
        min_pre_windows=('pre_windows','min'),min_post_windows=('post_windows','min'),min_early_windows=('early_windows','min')
    ).to_csv(PUB/'LATENCY_TIE_DIAGNOSTICS.csv',index=False)
    cc=pd.DataFrame(coverage);out=[]
    for center,g in [('ALL',cc)]+list(cc.groupby('center')):
        out.append(dict(center=center,records=len(g),**{n:int(g[n].sum()) for n in ['record_channel_units','valid_Raw_channel_windows',
            'valid_Raw_record_windows','raw_invalid_record_windows','native_invalid_record_windows','native_vs_Raw_mask_disagreement_record_windows']},
            uniform_mask_records=int(g.uniform_channel_mask.sum()),native_geometry_records=int(g.native_geometry_available.sum()),
            min_valid_pre_windows=int(g.valid_pre_windows.min()),min_valid_post_windows=int(g.valid_post_windows.min())))
    pd.DataFrame(out).to_csv(PUB/'ALIGNMENT_COVERAGE.csv',index=False)
    # The gate concerns source and operator properties, never prediction quality.
    status=[];overall=pd.DataFrame(aggregate).query("scope == 'overall'").set_index('feature')
    for name in NAMES:
        a=overall.loc[name];p=next(r for r in pp if r['scope']=='overall' and r['mode']=='RAW250_VALID' and r['feature']==name)
        distribution=distribution_table[(distribution_table['mode']=='RAW250_VALID')&(distribution_table.center=='ALL')&(distribution_table.feature==name)].iloc[0]
        native_distribution=distribution_table[(distribution_table['mode']=='CACHE_NATIVE_VALID_MASK')&(distribution_table.center=='ALL')&(distribution_table.feature==name)].iloc[0]
        support_complete=bool(cc.uniform_channel_mask.all() and (cc.valid_pre_windows>0).all() and (cc.valid_post_windows>0).all())
        if name.startswith('early_'):
            kind='high_gamma' if 'high_gamma' in name else 'line_length'
            support_complete=support_complete and all((r['early_windows'] if r['early_windows']>=2 else r['post_windows'])>=2 for r in diag if r['mode']=='RAW250_VALID' and r['feature']==kind)
        source_pass=bool(support_complete and distribution['finite_units']==24995 and p['units']==24995 and p['permutation_violating_units']==0)
        status.append(dict(feature=name,cache_formula_replay_within_frozen_tolerance=bool(a.formula_violating_units==0),
            broadcast_record_descriptor_not_window_feature=True,cache_safe_unchanged=bool(native_distribution.finite_units==24995 and a.native_mask_changed_units==0 and a.formula_violating_units==0),
            Raw250_valid_mask_scalar_computable=bool(distribution['finite_units']==24995),
            Raw250_channel_order_equivariant=bool(p['permutation_violating_units']==0),
            eligible_for_later_distinct_candidate_admission=source_pass,
            decision='DISTINCT_RAW250_MASKED_SCALAR_CAN_BE_CONSIDERED_NOT_YET_A_MODEL_INPUT' if source_pass else 'DO_NOT_ADMIT_UNCHANGED_OPERATOR_ARTIFACT_OR_COVERAGE_GAP',
            biological_ineffectiveness_claim=False,training_or_feature_pool_change_authorized=False))
    pd.DataFrame(status).to_csv(PUB/'ELIGIBILITY_DECISIONS.csv',index=False)
    assert sha(CACHE)==bound['feature_cache_sha256']
    assert sha(BASE/'RECONSTRUCTION_PRIVATE.pkl')==bound['reconstruction_sha256']
    write(PUB/'AUDIT_STATUS.json',dict(status='COMPLETE_UNLABELED_NONLOCAL8_ELIGIBILITY_AUDIT',
        patients=80,canonical_channels=7635,records=256,record_channel_units=units,valid_channel_windows=windows,
        valid_record_windows=record_windows,native_feature_rate_metadata=dict(rates),Raw_sfreq=250,Raw_Nyquist=125,
        source_mask_records_complete=len(coverage),private_diagnostic_sha256=sha(ROOT/'RECORD_DIAGNOSTICS_PRIVATE.pkl'),
        input_freeze_sha256=sha(ROOT/'INPUT_FREEZE_PRIVATE.json'),model_fits=0,labels_used=False,new_Outer_access=False,
        previous_features_models_thresholds_and_scores_unchanged=True,clinical_or_acquisition_provenance_proven=False,
        goal_full_completion_proven=False))
    print('NONLOCAL8_SOURCE_AUDIT_COMPLETE_NO_TRAINING',flush=True)


if __name__=='__main__':
    main()
