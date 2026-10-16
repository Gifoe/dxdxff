"""Unlabeled qualification on the actual original cohort and20 inner roles."""
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from measurement import GROUPS,SOURCE,crossing,aggregate,patient_z,prepare,nonconstant_rate

HERE=Path(__file__).resolve().parent
BASE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
PARENT=Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
AUDIT=Path(r'D:\m8_nonlocal8_eligibility_audit_seed42_v1_runtime')
CODE=Path(r'D:\m8_nonlocal8_eligibility_audit_seed42_v1')
ROOT=Path(r'D:\m8_nonlocal6_measurement_qualification_seed42_v1_runtime');PUB=ROOT/'public'
SRC=Path(r'E:\DRE-nips\new-pipeline\7-11')
sys.path.insert(0,str(SRC/'raw_delta_encoder_seed42_v1/code'))
from io_utils import read,save,write,sha,load
sys.path.insert(0,str(CODE))
from nonlocal_helpers import isolated,NAMES


def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(s):return ''.join(str(s).upper().split()).replace('-','').replace('_','')
def freeze(p,v):
    if p.exists():assert jr(p)==v
    else:write(p,v)


def identity_roles(rec):
    """Trusted historical bank materializes, but only identity/membership fields are accessed."""
    amendment=SRC/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json'
    assert sha(amendment)=='452a4f6957d18125a9ea1b32d827c5f50fbc81c22d9d41100173bddcd2e22503'
    expected=jr(amendment)['bank_sha256'];meta=rec['meta'];roles=[];bound={}
    lookup={(str(a),str(b)):i for i,(a,b) in enumerate(zip(meta['patient'],meta['channel']))}
    assert len(lookup)==7635
    for f in range(1,6):
        bp=Path(r'C:\a0_source_posterior_decoder_seed42_runtime')/f'fold{f}/BANK_PRIVATE.pt'
        assert sha(bp)==expected[f-1];bound[f'bank{f}']=sha(bp)
        b=torch.load(bp,map_location='cpu',weights_only=False)
        patients=np.asarray(b['patient']).astype(str);channels=np.asarray(b['channel']).astype(str)
        ix=np.array([lookup[(p,c)] for p,c in zip(patients,channels)])
        fit=set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']=='fit')])
        excluded=set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']!='fit')])
        assert set(patients[b['train']])==fit and set(patients).isdisjoint(set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']=='test')]))
        for s in range(4):
            p=PARENT/'screens_private'/(f'fold{f}_screen{s}.pkl' if s else f'fold{f}_final_path.pkl')
            bound[f'roles{f}_{s}']=sha(p);q=read(p)
            names=['train','cal','choice']+(['held'] if s else [])
            rr={n:ix[q[n]] for n in names};sets=[set(meta['patient'][rr[n]]) for n in names]
            assert all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i))
            assert set.union(*sets)==fit and set.union(*sets).isdisjoint(excluded)
            roles.append(dict(fold=f,screen=s,indices=rr))
        del b,q;gc.collect()
    return roles,bound


def main():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    lock=jr(HERE/'PROTOCOL_LOCK.json');prior=jr(AUDIT/'public/AUDIT_STATUS.json')
    assert prior['status']=='COMPLETE_UNLABELED_NONLOCAL8_ELIGIBILITY_AUDIT'
    assert sha(AUDIT/'RECORD_DIAGNOSTICS_PRIVATE.pkl')==lock['parent_private_diagnostic_sha256']
    evidence=read(AUDIT/'RECORD_DIAGNOSTICS_PRIVATE.pkl')
    rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');reg=read(PARENT/'POOL_REGISTRY_PRIVATE.pkl')
    assert sha(BASE/'RECONSTRUCTION_PRIVATE.pkl')==evidence['binding']['reconstruction_sha256']
    assert sha(PARENT/'POOL_REGISTRY_PRIVATE.pkl')==evidence['binding']['pool_registry_sha256']
    ezpath=SRC/'ez_features.py';assert sha(ezpath)==evidence['binding']['measurement_sources']['ez_features.py']
    ez=load('qualification_pinned_ez',ezpath)
    roles,identity=identity_roles(rec)
    bound=dict(code={n:sha(HERE/n) for n in ['PROTOCOL_LOCK.json','measurement.py','qualify.py','test_measurement.py']},
        parent_private=sha(AUDIT/'RECORD_DIAGNOSTICS_PRIVATE.pkl'),parent_input=sha(AUDIT/'INPUT_FREEZE_PRIVATE.json'),
        reconstruction=sha(BASE/'RECONSTRUCTION_PRIVATE.pkl'),registry=sha(PARENT/'POOL_REGISTRY_PRIVATE.pkl'),
        identity=identity,source=sha(ezpath),parent_Outer_seal=sha(PARENT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'))
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',bound)
    if (PUB/'QUALIFICATION_STATUS.json').exists():
        assert jr(PUB/'QUALIFICATION_STATUS.json')['input_freeze_sha256']==sha(ROOT/'INPUT_FREEZE_PRIVATE.json')
        print('QUALIFICATION_ALREADY_COMPLETE',flush=True);return
    audit_lookup={(r['patient'],r['seizure'],r['feature']):r['Raw250_mask'] for r in evidence['records']}
    x=np.full((7635,12),np.nan,float);seen=np.zeros(7635,int);private=[];rows=[]
    units=windows=0;max_error=0.;permutations=0
    for number,ledger in enumerate(rec['ledgers'],1):
        pid=ledger['patient'];entry=reg['files'][pid];assert sha(entry['path'])==entry['sha256']
        q=read(Path(entry['path']));assert q['binding']==reg['binding']
        assert q['channels']==ledger['channels']
        canon={norm(c):i for i,c in enumerate(q['channels'])};assert len(canon)==len(q['channels'])
        S=len(ledger['records']);C=len(q['channels'])
        scalar=np.full((S,C,6),np.nan,float);present=np.zeros((S,C),bool)
        eligible=np.zeros((S,C,2),bool);observed=eligible.copy();fraction=np.full((S,C,2),np.nan,float)
        center=np.unique(rec['meta']['center'][rec['meta']['patient']==pid]);assert len(center)==1
        for s,item in enumerate(ledger['records']):
            ix=np.array([canon[norm(c)] for c in item['names']]);t=np.asarray(item['times'],np.float32)
            mask=np.asarray(item['mask']);assert np.array_equal(mask,q['mask'][s,ix,:len(t)].T)
            assert np.array_equal(mask,np.broadcast_to(mask[:,:1],mask.shape))
            valid=mask[:,0];tt=t[valid]
            hg=q['values'][s,ix,:len(t),2].T[valid];ll=q['values'][s,ix,:len(t),5].T[valid]
            a=isolated(ez,hg,ll,tt)
            for j in SOURCE:
                ref=audit_lookup[(pid,s,NAMES[j])]
                assert np.array_equal(a[:,j],ref)
                max_error=max(max_error,float(np.max(np.abs(a[:,j].astype(float)-ref))))
            scalar[s,ix]=a[:,SOURCE];present[s,ix]=True
            for k,series in enumerate([hg,ll]):
                e,o,frac=crossing(series,tt);eligible[s,ix,k]=e;observed[s,ix,k]=o;fraction[s,ix,k]=frac
                np.testing.assert_allclose(frac[o],a[o,2+k],atol=1e-7,rtol=1e-6)
            units+=len(ix);windows+=int(mask.sum())
        result=aggregate(scalar,present,eligible,observed,fraction,np.arange(S))
        rev=aggregate(scalar[:,::-1],present[:,::-1],eligible[:,::-1],observed[:,::-1],fraction[:,::-1],np.arange(S))[::-1]
        assert np.array_equal(result,rev,equal_nan=True)
        rev=aggregate(scalar[::-1],present[::-1],eligible[::-1],observed[::-1],fraction[::-1],np.arange(S)[::-1])
        assert np.array_equal(result,rev,equal_nan=True);permutations+=2
        ix=np.asarray(q['archive_indices']);assert np.array_equal(rec['meta']['patient'][ix],np.full(C,pid))
        x[ix]=result;seen[ix]+=1
        private.append(dict(archive_indices=ix,scalar=scalar,present=present,eligible=eligible,observed=observed,fraction=fraction))
        for j,name in enumerate(GROUPS):
            row=dict(center=str(center[0]),feature=name,patients=1,channels=C,record_channel_units=int(present.sum()),
                source_finite_units=int(np.isfinite(scalar[:,:,j][present]).sum()),
                primary0_finite_channels=int(np.isfinite(result[:,2*j]).sum()),primary1_finite_channels=int(np.isfinite(result[:,2*j+1]).sum()))
            k=j-2
            row.update(baseline_eligible_units=int(eligible[:,:,k][present].sum()) if j in [2,3] else int(present.sum()),
                observed_crossing_units=int(observed[:,:,k][present].sum()) if j in [2,3] else 0,
                eligible_no_crossing_units=int((eligible[:,:,k]&~observed[:,:,k])[present].sum()) if j in [2,3] else 0)
            rows.append(row)
        print('UNLABELED_QUALIFIED_PATIENT',number,'of80',flush=True);del q;gc.collect()
    assert units==24995 and windows==1471965 and (seen==1).all()
    patients=np.asarray(rec['meta']['patient']);z=patient_z(x,patients);legacy=patient_z(x,patients,True)
    stress=[]
    for scale in lock['quality_gates']['unit_stress_scales']:
        zs=patient_z(x*scale,patients);ls=patient_z(x*scale,patients,True)
        for j,name in enumerate(GROUPS):
            err=float(np.nanmax(np.abs(zs[:,2*j:2*j+2]-z[:,2*j:2*j+2])))
            assert err<=lock['quality_gates']['unit_equivariance_atol']
            stress.append(dict(feature=name,coordinate_scale=scale,unit_equivariant_max_abs_error=err,
                legacy_floor_max_abs_change=float(np.nanmax(np.abs(ls[:,2*j:2*j+2]-legacy[:,2*j:2*j+2]))),
                raw_voltage_unit_invariance_tested=False))
    rr=[];roleprep=[]
    for r in roles:
        fitted,prep=prepare(z,r['indices']['train']);old,_=prepare(legacy,r['indices']['train'])
        assert np.isfinite(fitted).all() and np.isfinite(old).all()
        rates=nonconstant_rate(x[r['indices']['train']],patients[r['indices']['train']])
        for role,ix in r['indices'].items():
            for j,name in enumerate(GROUPS):
                rr.append(dict(fold=r['fold'],screen=r['screen'],role=role,feature=name,patients=len(set(patients[ix])),channels=len(ix),
                    coordinate0_nonconstant_train_patient_fraction=rates[2*j],coordinate1_nonconstant_train_patient_fraction=rates[2*j+1],
                    all_fitted_inputs_finite=bool(np.isfinite(fitted[ix,2*j:2*j+2]).all()),
                    unit_equivariant_max_abs_fitted_input=float(np.max(np.abs(fitted[ix,2*j:2*j+2]))),
                    legacy_floor_max_abs_fitted_input=float(np.max(np.abs(old[ix,2*j:2*j+2]))),
                    imputer_and_scaler_selection_train_only=True))
        roleprep.append(dict(fold=r['fold'],screen=r['screen'],indices=r['indices'],pre=prep))
    save(ROOT/'MEASUREMENTS_PRIVATE.pkl',dict(binding=bound,values=x,records=private,roles=roleprep))
    frame=pd.DataFrame(rows);coverage=[]
    numeric=[c for c in frame if c not in ['center','feature']]
    for center,g in [('ALL',frame)]+list(frame.groupby('center')):
        for name,part in g.groupby('feature'):
            out=dict(center=center,feature=name,**{c:int(part[c].sum()) for c in numeric})
            out.update(source_finite_rate=out['source_finite_units']/out['record_channel_units'],
                baseline_eligible_rate=out['baseline_eligible_units']/out['record_channel_units'],
                primary0_coverage=out['primary0_finite_channels']/out['channels'],primary1_coverage=out['primary1_finite_channels']/out['channels'])
            coverage.append(out)
    cv=pd.DataFrame(coverage);ro=pd.DataFrame(rr);decisions=[]
    for j,name in enumerate(GROUPS):
        a=cv[(cv.center=='ALL')&(cv.feature==name)].iloc[0];g=ro[(ro.feature==name)&(ro.role=='train')]
        source_pass=a.source_finite_rate>=.99 and a.baseline_eligible_rate>=.99
        cover=a.primary1_coverage>=.99 and (j in [2,3] or a.primary0_coverage>=.99)
        fraction=float(g[['coordinate0_nonconstant_train_patient_fraction','coordinate1_nonconstant_train_patient_fraction']].max(axis=1).min())
        passed=source_pass and cover and fraction>=.5
        decisions.append(dict(feature=name,source_quality_pass=bool(source_pass),primary_coverage_pass=bool(cover),
            minimum_20_partition_nonconstant_patient_fraction=fraction,all20_inner_partitions_pass=bool(fraction>=.5),
            eligible_for_future_preregistered_supervised_comparison=bool(passed),predictive_usefulness_proven=False))
    cv.to_csv(PUB/'SOURCE_COVERAGE.csv',index=False);ro.to_csv(PUB/'INNER_NUMERIC_QUALIFICATION.csv',index=False)
    pd.DataFrame(stress).to_csv(PUB/'UNIT_STRESS.csv',index=False);pd.DataFrame(decisions).to_csv(PUB/'QUALIFICATION_DECISIONS.csv',index=False)
    assert sha(PARENT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==bound['parent_Outer_seal']
    write(PUB/'QUALIFICATION_STATUS.json',dict(status='COMPLETE_UNLABELED_MEASUREMENT_QUALIFICATION',patients=80,channels=7635,
        records=256,record_channel_units=units,valid_channel_windows=windows,groups=6,coordinates=12,inner_partitions=20,
        source_scalar_replay_max_abs_error=max_error,exact_permutation_checks=permutations,
        eligible_groups=sum(d['eligible_for_future_preregistered_supervised_comparison'] for d in decisions),
        input_freeze_sha256=sha(ROOT/'INPUT_FREEZE_PRIVATE.json'),private_measurements_sha256=sha(ROOT/'MEASUREMENTS_PRIVATE.pkl'),
        labels_indexed=False,classifier_fits=0,new_Outer_access=False,existing_models_thresholds_and_pool_unchanged=True,
        goal_stable_minimum_or_predictive_increment_proven=False))
    print('MEASUREMENT_QUALIFICATION_COMPLETE_NO_LABELS_NO_FITS',flush=True)


if __name__=='__main__':main()
