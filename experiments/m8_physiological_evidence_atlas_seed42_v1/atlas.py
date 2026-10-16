"""Audit old patient effects and describe12 qualified record coordinates."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from statistics_core import effect,interval,pvalue,bh,old_summary

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_physiological_evidence_atlas_seed42_v1_runtime');PUB=ROOT/'public'
OLD=Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
QUAL=Path(r'D:\m8_nonlocal6_measurement_qualification_seed42_v1_runtime\constant_safe')
SRC=Path(r'E:\DRE-nips\new-pipeline\7-11');BASE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
sys.path.insert(0,str(SRC/'raw_delta_encoder_seed42_v1/code'))
from io_utils import read,save,write,sha

GROUPS=['early_partial_HG_80_125_slope','early_LL_slope','baseline_crossing_partial_HG_80_125',
        'baseline_crossing_LL','post_top20_partial_HG_80_125','post_top20_LL']
COORDS=[name+'__'+suffix for j,name in enumerate(GROUPS) for suffix in (['observed_position','crossing_rate'] if j in [2,3] else ['seizure_mean','seizure_SD'])]


def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def freeze(p,v):
    if p.exists():assert jr(p)==v
    else:write(p,v)


def audit_old(frame):
    public=pd.read_csv(OLD/'public/PHYSIOLOGICAL_EFFECTS.csv');rows=[];maximum=0.;cells=0
    assert not frame.duplicated(['outer_fold','patient','feature','phase']).any()
    for key,g in frame.groupby(['outer_fold','feature','phase']):
        assert len(g)==g.patient.nunique()==51
        rows.append(dict(outer_fold=key[0],feature=key[1],phase=key[2],**old_summary(g)))
    calculated=pd.DataFrame(rows);calculated['BH_q']=np.nan
    for fold,ix in calculated.groupby('outer_fold').groups.items():calculated.loc[ix,'BH_q']=bh(calculated.loc[ix,'pvalue'])
    calculated['operationally_reproducible']=((calculated.BH_q<.05)&((calculated.ci_low>0)|(calculated.ci_high<0))&(calculated.patient_direction_fraction>=.6)&(calculated.same_direction_source_groups>=3)&(calculated.repeat_seizure_patients>=10)&(calculated.median_seizure_sign_agreement>=.7))
    checked=calculated.sort_values(['outer_fold','feature','phase']).reset_index(drop=True);reference=public.sort_values(['outer_fold','feature','phase']).reset_index(drop=True)
    for column in checked.columns:
        if column in ['feature','phase']:assert checked[column].tolist()==reference[column].tolist();continue
        np.testing.assert_allclose(checked[column].to_numpy(float),reference[column].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
        values=np.abs(checked[column].to_numpy(float)-reference[column].to_numpy(float));maximum=max(maximum,float(np.nanmax(values)));cells+=len(values)
    source=pd.read_csv(OLD/'public/PHYSIOLOGY_SOURCE_EFFECTS.csv');fields=0
    for key,g in frame.groupby(['outer_fold','feature','phase','center']):
        ref=source[(source.outer_fold==key[0])&(source.feature==key[1])&(source.phase==key[2])&(source.center==key[3])].iloc[0]
        values=interval(g.effect.to_numpy());np.testing.assert_allclose(values,[ref.mean_effect,ref.ci_low,ref.ci_high],atol=1e-12,rtol=0,equal_nan=True)
        assert np.isfinite(g.effect).sum()==ref.finite_patients;fields+=1
    assert len(calculated)==800 and fields==3200
    write(PUB/'OLD_EFFECT_STATISTICAL_AUDIT.json',dict(status='PASS',effect_rows_reaggregated=800,source_rows_reaggregated=fields,
        numeric_or_flag_cells_checked=cells,maximum_old_numeric_error=maximum,primary_unique_patients=51,
        private_patient_effect_rows=len(frame),old_primary_BH_family_unchanged=160,raw_waveform_measurement_replay=False))
    return calculated[calculated.outer_fold==1].copy()


def legal_primary(rec,binding):
    path=Path(r'C:\a0_source_posterior_decoder_seed42_runtime\fold1\BANK_PRIVATE.pt')
    qbind=jr(QUAL/'INPUT_FREEZE_PRIVATE.json');assert sha(path)==qbind['identity']['bank1']
    b=torch.load(path,map_location='cpu',weights_only=False);tr=b['train'];meta=rec['meta']
    fit=set(meta['split_patient'][(meta['split_fold']==1)&(meta['split_role']=='fit')]);assert len(fit)==51
    pid=np.asarray(b['patient'])[tr].astype(str);ch=np.asarray(b['channel'])[tr].astype(str);assert set(pid)==fit
    lookup={(str(p),str(c)):i for i,(p,c) in enumerate(zip(meta['patient'],meta['channel']))}
    ix=np.array([lookup[p,c] for p,c in zip(pid,ch)]);y=(np.asarray(b['y'])[tr]==0).astype(int)
    return {int(i):int(v) for i,v in zip(ix,y)},fit


def main():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True);lock=jr(HERE/'PROTOCOL_LOCK.json')
    assert sha(QUAL/'MEASUREMENTS_PRIVATE.pkl')==lock['qualified_measurements_sha256']
    assert sha(QUAL/'INPUT_FREEZE_PRIVATE.json')==lock['qualified_input_sha256']
    assert jr(QUAL/'public/INDEPENDENT_QUALIFICATION_VERIFICATION.json')['status']=='PASS'
    assert sha(Path(r'D:\m8_physiology_discovery_selection_seed42_v2\biology.py'))==lock['old_biology_source_sha256']
    assert sha(OLD/'public/PHYSIOLOGICAL_EFFECTS.csv')==lock['published_old_effect_table_sha256']
    assert sha(OLD/'public/PHYSIOLOGY_SOURCE_EFFECTS.csv')==lock['published_old_source_table_sha256']
    rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');qbind=jr(QUAL/'INPUT_FREEZE_PRIVATE.json');assert sha(BASE/'RECONSTRUCTION_PRIVATE.pkl')==qbind['reconstruction']
    binding=dict(code={n:sha(HERE/n) for n in ['PROTOCOL_LOCK.json','statistics_core.py','atlas.py','test_statistics.py']},
        qualification=lock['qualified_measurements_sha256'],qualification_input=lock['qualified_input_sha256'],
        old_effect_rows=sha(OLD/'BIOLOGY_PATIENT_EFFECTS_PRIVATE.pkl'),old_effect_table=sha(OLD/'public/PHYSIOLOGICAL_EFFECTS.csv'),
        old_source_table=sha(OLD/'public/PHYSIOLOGY_SOURCE_EFFECTS.csv'),old_BIOLOGY_lock=sha(Path(r'D:\m8_physiology_discovery_selection_seed42_v2\BIOLOGY_LOCK.json')),
        parent_Outer_seal=sha(OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),reconstruction=sha(BASE/'RECONSTRUCTION_PRIVATE.pkl'))
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    if (PUB/'ATLAS_STATUS.json').exists():
        assert jr(PUB/'ATLAS_STATUS.json')['input_freeze_sha256']==sha(ROOT/'INPUT_FREEZE_PRIVATE.json');print('ATLAS_ALREADY_COMPLETE',flush=True);return
    old=read(OLD/'BIOLOGY_PATIENT_EFFECTS_PRIVATE.pkl');primary=audit_old(old)
    labels,fit=legal_primary(rec,binding);measure=read(QUAL/'MEASUREMENTS_PRIVATE.pkl');rows=[];run_rows=[]
    for record in measure['records']:
        ix=record['archive_indices'];patients=np.unique(rec['meta']['patient'][ix]);assert len(patients)==1;pid=patients[0]
        if pid not in fit:continue
        assert set(map(int,ix)).issubset(labels);y=np.array([labels[int(i)] for i in ix]);center=np.unique(rec['meta']['center'][ix]);assert len(center)==1
        values=measure['values'][ix]
        for c,name in enumerate(COORDS):
            rows.append(dict(patient=pid,center=str(center[0]),coordinate=name,group=GROUPS[c//2],effect=effect(values[:,c],y),
                finite_EZ_channels=int((np.isfinite(values[:,c])&(y==1)).sum()),finite_NEZ_channels=int((np.isfinite(values[:,c])&(y==0)).sum())))
            j=c//2;applicable=(c%2==0 or j in [2,3])
            if not applicable:continue
            for s in range(record['present'].shape[0]):
                if j in [2,3]:
                    k=j-2;a=record['fraction'][s,:,k].copy() if c%2==0 else record['observed'][s,:,k].astype(float)
                    if c%2:a[~record['eligible'][s,:,k]]=np.nan
                else:a=record['scalar'][s,:,j].copy()
                a[~record['present'][s]]=np.nan
                run_rows.append(dict(patient=pid,center=str(center[0]),coordinate=name,seizure=s,effect=effect(a,y)))
    frame=pd.DataFrame(rows);runs=pd.DataFrame(run_rows);assert len(frame)==51*12 and frame.patient.nunique()==51
    save(ROOT/'NEW_COORDINATE_PATIENT_EFFECTS_PRIVATE.pkl',dict(binding=binding,patient_effects=frame,record_effects=runs))
    output=[];sources=[];repeat=[]
    for name,g in frame.groupby('coordinate'):
        a=g.effect.to_numpy();a=a[np.isfinite(a)];mu,lo,hi=interval(a);same=eligible=0
        for center,h in g.groupby('center'):
            values=h.effect.to_numpy();values=values[np.isfinite(values)];sm,sl,sh=interval(values)
            sources.append(dict(coordinate=name,center=center,finite_patients=len(values),mean_effect=sm,ci_low=sl,ci_high=sh))
            if len(values)>=5:eligible+=1;same+=np.sign(sm)==np.sign(mu)
        c=COORDS.index(name);applicable=c%2==0 or c//2 in [2,3];directed=[];legacy=[];finite_count=zero_count=0
        own=runs[runs.coordinate==name]
        for pid,h in own.groupby('patient'):
            v=h.effect.to_numpy();v=v[np.isfinite(v)];nonzero=v[v!=0];finite_count+=len(v);zero_count+=int((v==0).sum())
            if len(v)>=2:directed.append(float(((v!=0)&(np.sign(v)==np.sign(mu))).mean()))
            if len(nonzero)>=2:legacy.append(float(max((nonzero>0).sum(),(nonzero<0).sum())/len(nonzero)))
        repeat.append(dict(coordinate=name,record_analogue_applicable=bool(applicable),finite_record_effects=finite_count,zero_record_effects=zero_count,
            repeated_finite_seizure_patients=len(directed),median_inclusive_directed_agreement=float(np.median(directed)) if directed else np.nan,
            repeated_nonzero_seizure_patients=len(legacy),median_legacy_zero_excluding_majority=float(np.median(legacy)) if legacy else np.nan))
        output.append(dict(coordinate=name,group=g.group.iloc[0],patients=len(a),mean_effect=mu,ci_low=lo,ci_high=hi,pvalue=pvalue(a),
            patient_direction_fraction=float((np.sign(a)==np.sign(mu)).mean()) if len(a) else np.nan,
            eligible_source_groups=eligible,same_direction_source_groups=int(same),record_analogue_applicable=bool(applicable)))
    table=pd.DataFrame(output);table['BH_q_new12']=bh(table.pvalue.to_numpy())
    table['association_support']=((table.BH_q_new12<.05)&((table.ci_low>0)|(table.ci_high<0))&(table.patient_direction_fraction>=.6)&(table.same_direction_source_groups>=3))
    rep=pd.DataFrame(repeat);rep['repeated_direction_support']=rep.record_analogue_applicable & (rep.repeated_finite_seizure_patients>=10)&(rep.median_inclusive_directed_agreement>=.7)
    table.to_csv(PUB/'NEW12_PATIENT_ASSOCIATIONS.csv',index=False);pd.DataFrame(sources).to_csv(PUB/'NEW12_SOURCE_ASSOCIATIONS.csv',index=False)
    rep.to_csv(PUB/'NEW12_REPEAT_SEIZURE_SUPPORT.csv',index=False)
    atlas=[]
    for r in primary.itertuples(index=False):atlas.append(dict(measurement_family='original_window_phase',feature=r.feature,coordinate_or_phase=r.phase,
        patients=r.patients,mean_effect=r.mean_effect,ci_low=r.ci_low,ci_high=r.ci_high,pvalue=r.pvalue,original_family_BH_q=r.BH_q))
    for r in table.itertuples(index=False):atlas.append(dict(measurement_family='qualified_record_coordinate',feature=r.group,coordinate_or_phase=r.coordinate.split('__')[-1],
        patients=r.patients,mean_effect=r.mean_effect,ci_low=r.ci_low,ci_high=r.ci_high,pvalue=r.pvalue,original_family_BH_q=r.BH_q_new12))
    atlas=pd.DataFrame(atlas);assert len(atlas)==172;atlas['exploratory_joint172_BH_q']=bh(atlas.pvalue.to_numpy());atlas.to_csv(PUB/'PRIMARY172_ASSOCIATION_ATLAS.csv',index=False)
    assert sha(OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==binding['parent_Outer_seal']
    write(PUB/'ATLAS_STATUS.json',dict(status='COMPLETE_DESCRIPTIVE_PHYSIOLOGICAL_EVIDENCE_ATLAS',primary_patients=51,
        new_coordinate_endpoints=12,new_patient_effect_rows=len(frame),new_record_effect_rows=len(runs),old_primary_endpoints=160,
        old_overlapping_effect_rows_audited=800,old_source_rows_audited=3200,combined_exploratory_test_family=172,
        new_association_support_count=int(table.association_support.sum()),new_record_repetition_applicable=int(rep.record_analogue_applicable.sum()),
        new_repeated_direction_support_count=int(rep.repeated_direction_support.sum()),
        input_freeze_sha256=sha(ROOT/'INPUT_FREEZE_PRIVATE.json'),private_new_effects_sha256=sha(ROOT/'NEW_COORDINATE_PATIENT_EFFECTS_PRIVATE.pkl'),
        model_fits=0,new_feature_selection=False,new_Outer_access=False,previous_predictive_allocation_gates_unchanged=True,
        stable_minimum_or_Raw_increment_proven=False,independent_clinical_SOZ_or_onset_proven=False))
    print('PHYSIOLOGICAL_EVIDENCE_ATLAS_COMPLETE_NO_MODELS',flush=True)


if __name__=='__main__':main()
