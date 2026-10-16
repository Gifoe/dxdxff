"""Replay every sealed readout, then report all internal outcomes, never Outer."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_readouts as run
from readout import Readout
io=run.io
ROOT=run.ROOT;PUB=run.PUB
METRICS=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
UTILITY=['auroc','auprc','macro_f1','f1']


def paired(a,ids):
    unique=np.unique(ids);rng=np.random.default_rng(42)
    draws=rng.integers(len(unique),size=(10000,len(unique)))
    weights=np.array([np.bincount(v,minlength=len(unique)) for v in draws],float)
    sums=np.array([np.nansum(a[ids==who],axis=0) for who in unique])
    counts=np.array([np.isfinite(a[ids==who]).sum(0) for who in unique])
    den=weights@counts;dist=np.divide(weights@sums,den,out=np.full(den.shape,np.nan),where=den>0)
    return np.nanmean(a,axis=0),np.nanpercentile(dist,[2.5,97.5],axis=0)


def summaries(frame,keys):
    result=[]
    for values,g in frame.groupby(keys):
        if not isinstance(values,tuple):values=(values,)
        result.append(dict(zip(keys,values),patient_appearances=len(g),unique_patients=g.patient.nunique(),
                           **{m:float(g[m].mean()) for m in METRICS}))
    return pd.DataFrame(result)


def main():
    torch.set_num_threads(1)
    common,data,_,_,p=io.old_modules();_,core,_,_,_,_,banks,_=common.legal();engine=common.ev()
    seal=io.jr(ROOT/'READOUT_SCORE_SEAL_PRIVATE.json');binding=io.jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    assert seal['binding']==binding and io.bindings()==binding['parent_binding']['original_scientific_binding']
    for name,h in binding['source'].items():assert io.sha(HERE/name)==h
    for name,h in seal['files'].items():assert io.sha(ROOT/name)==h
    registry=io.read(ROOT/'READOUT_REGISTRY_PRIVATE.pkl');assert registry['binding']==binding
    entries=io.read(ROOT/'PLAN_PRIVATE.pkl')['entries'];pool=io.read(io.OLDROOT/'POOL_PRIVATE.pkl')
    prepared={};replay=[];patient=[];probabilities={}
    for entry in registry['entries']:
        fold,screen=entry['fold'],entry['screen'];bank=banks[fold-1];fit=bank['train'];key=(fold,screen)
        plan=next(q for q in entries if (q['fold'],q['screen'])==key)
        held=plan['roles']['held'];pid=bank['patient'][fit];channels=bank['channel'][fit];center=bank['center'][fit]
        if key not in prepared:
            raw=io.read(Path(plan['raw_entry']['path']));assert io.sha(Path(plan['raw_entry']['path']))==plan['raw_entry']['sha256']
            x=np.c_[raw['raw'],core.patient_z(pool['summary'][bank['archive_indices'][fit]],pid)]
            refit=np.r_[plan['roles']['train'],plan['roles']['choice']]
            z,pre=run.prepared(x,refit);saved=io.read(ROOT/f'fold{fold}_screen{screen}_PREPROCESS_PRIVATE.pkl')
            assert saved['binding']==binding
            for name in pre:assert np.array_equal(pre[name],saved['pre'][name])
            sets=[set(pid[ix]) for ix in plan['roles'].values()]
            assert all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i))
            assert set.union(*sets)==bank['roles']['fit'] and set.union(*sets).isdisjoint(bank['roles']['validation']|bank['roles']['test'])
            prepared[key]=z
        z=prepared[key];seed=entry['head_seed']+1009*fold+50021*screen
        cell=Path(entry['cell']);assert io.sha(cell/'BEST_PRIVATE.pt')==entry['checkpoint_sha256']
        checkpoint=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        model=Readout(entry['groups'],seed);assert core.state_hash(model.network.state_dict())==entry['initial']
        model.load_state_dict(checkpoint['model']);logit=run.prediction(model,z[held])
        saved=io.read(Path(entry['score']));assert np.array_equal(saved['logit_nez'],logit)
        assert saved['checkpoint_sha256']==entry['checkpoint_sha256']
        assert np.array_equal(saved['archive_indices'],bank['archive_indices'][fit[held]])
        cal=plan['roles']['cal'];ycal=1-bank['y'][fit[cal]]
        pc=expit(-run.prediction(model,z[cal]).astype(float));theta=p.threshold(ycal,pc,pid[cal])
        assert theta==entry['threshold']==saved['threshold']
        cp=io.jr(cell/'COMPLETE_PRIVATE.json')
        assert cp['best_sha256']==entry['checkpoint_sha256']
        last=torch.load(cell/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        best=max(last['history'],key=lambda r:(r['macro_f1'],r['ez_f1'],-r['epoch']))
        assert best['epoch']==entry['best_epoch']==checkpoint['epoch'] and best['threshold']==theta
        probability=expit(-logit.astype(float));yy=1-bank['y'][fit[held]]
        independent=pd.DataFrame(engine.patient_rows(pd.DataFrame(dict(model=entry['arm'],version=str(entry['head_seed']),
            fold=fold,patient=pid[held],channel=channels[held],center=center[held],y=yy,
            probability=probability,prediction=probability>=theta)))).sort_values('patient')
        primitive=pd.DataFrame(p.metric_rows(yy,probability,pid[held],channels[held],center[held],theta)).sort_values('patient')
        assert np.allclose(independent[METRICS],primitive[METRICS],atol=1e-12,rtol=0,equal_nan=True)
        patient.append(independent.assign(arm=entry['arm'],screen=screen,head_seed=entry['head_seed']))
        probabilities[entry['arm'],fold,screen,entry['head_seed']]=(probability,pid[held])
        replay.append(dict(fold=fold,screen=screen,head_seed=entry['head_seed'],arm=entry['arm'],
                           logit_replay_max_error=0.,threshold_replay_exact=True,patient_metric_replay=True,
                           checkpoint_and_initialization_match=True))
        print('SEALED_READOUT_REPLAYED',len(replay),'of90',flush=True)
    assert len(replay)==90
    frame=pd.concat(patient,ignore_index=True);assert len(frame)==1530
    io.save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',dict(binding=binding,frame=frame))
    pd.DataFrame(replay).to_csv(PUB/'READOUT_REPLAY_VERIFICATION.csv',index=False)
    io.write(PUB/'REPLAY_STATUS.json',dict(status='PASS',logical_cells=90,patient_metric_rows=len(frame),
        every_logit_replay_exact=True,all_calibration_thresholds_and_epochs_match=True,
        independent_patient_metrics=True,role_preprocessing_and_initialization_checks=True,
        no_neural_training_or_new_Outer=True,source_sha256=io.sha(Path(__file__))))
    summaries(frame,['arm','head_seed']).to_csv(PUB/'READOUT_SEED_RESULTS.csv',index=False)
    average=frame.groupby(['arm','fold','screen','patient','center'],as_index=False)[METRICS].mean()
    summaries(average,['arm']).to_csv(PUB/'READOUT_MAIN_RESULTS.csv',index=False)
    summaries(average,['arm','fold']).to_csv(PUB/'READOUT_FOLD_RESULTS.csv',index=False)
    source=summaries(average,['arm','center']);source.to_csv(PUB/'READOUT_SOURCE_RESULTS.csv',index=False)
    byidentity=average.groupby(['arm','patient','center'],as_index=False)[METRICS].mean()
    summaries(byidentity,['arm']).to_csv(PUB/'READOUT_UNIQUE_ID_BALANCED_RESULTS.csv',index=False)
    ordered={arm:g.sort_values(['fold','screen','patient']) for arm,g in average.groupby('arm')}
    selected=ordered['SELECTED'];intervals=[];utilities=[]
    for arm in ['RAW_ONLY','OLD7']:
        ref=ordered[arm];assert list(zip(selected.fold,selected.screen,selected.patient))==list(zip(ref.fold,ref.screen,ref.patient))
        delta=selected[METRICS].to_numpy()-ref[METRICS].to_numpy()
        mu,ci=paired(delta,selected.patient.to_numpy())
        for j,m in enumerate(METRICS):intervals.append(dict(model='SELECTED',reference=arm,metric=m,
            delta=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],draws=10000,exploratory_reused_internal=True,
            cardinality_assessment_labels_not_used=True,fixed_encoders_and_two_head_seeds=True))
        utility=(selected[UTILITY].mean(axis=1).to_numpy()-ref[UTILITY].mean(axis=1).to_numpy())[:,None]
        mu,ci=paired(utility,selected.patient.to_numpy())
        utilities.append(dict(reference=arm,utility_delta=mu[0],ci_low=ci[0,0],ci_high=ci[1,0]))
    pd.DataFrame(intervals).to_csv(PUB/'READOUT_PAIRED_INTERVALS.csv',index=False)
    pd.DataFrame(utilities).to_csv(PUB/'READOUT_UTILITY_INTERVALS.csv',index=False)
    seedmeans=summaries(frame,['arm','head_seed']).set_index(['arm','head_seed'])
    seed_pass=all(seedmeans.loc[('SELECTED',s),m]>seedmeans.loc[('RAW_ONLY',s),m] for s in [42,43] for m in UTILITY)
    sm=source.set_index(['arm','center']);source_pass=all(sm.loc[('SELECTED',c),'auroc']-sm.loc[('RAW_ONLY',c),'auroc']>=-.02 for c in source.center.unique())
    utility_pass=utilities[0]['ci_low']>0
    stability=[]
    for arm in ['RAW_ONLY','SELECTED','OLD7']:
        cor=[]
        for fold in range(1,6):
            for screen in range(1,4):
                a,ids=probabilities[arm,fold,screen,42];b,ids2=probabilities[arm,fold,screen,43];assert np.array_equal(ids,ids2)
                for who in np.unique(ids):
                    index=ids==who
                    if np.ptp(a[index])>0 and np.ptp(b[index])>0:cor.append(float(spearmanr(a[index],b[index]).statistic))
        stability.append(dict(arm=arm,finite_patient_appearances=len(cor),total_patient_appearances=255,
            mean_spearman=float(np.mean(cor)),median_spearman=float(np.median(cor)),
            q25=float(np.quantile(cor,.25)),q75=float(np.quantile(cor,.75)),
            head_seed_only_Raw_encoder_fixed=True))
    pd.DataFrame(stability).to_csv(PUB/'READOUT_HEAD_SEED_STABILITY.csv',index=False)
    assert io.bindings()==binding['parent_binding']['original_scientific_binding']
    io.write(PUB/'ALLOCATION_GATE.json',dict(status='PASS' if seed_pass and source_pass and utility_pass else 'FAIL',
        four_endpoint_utility_CI_lower_positive=bool(utility_pass),each_primary_mean_positive_in_both_head_seeds=bool(seed_pass),
        all_source_AUROC_deltas_at_least_minus002=bool(source_pass),
        no_full_Raw_M8_training_automatically_authorized=True,not_independent_confirmation=True,
        stable_minimal_feature_set_not_proven=True))
    io.write(PUB/'REPORT_STATUS.json',dict(status='COMPLETE_VERIFIED_BOUNDED_INTERNAL_READOUT_TRANSFER',
        rows=len(frame),patient_appearances_per_procedure=255,unique_patients=79,
        original_Development_or_Outer_used=False,full_Raw_encoder_retraining=0,
        full_M8_or_end_to_end_Concat_tested_in_this_supplement=False,
        goal_full_completion_proven=False))
    print('INTERNAL_READOUT_REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
