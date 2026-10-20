"""Independent CAL selection/LOPO and descriptive-stat verification; no Raw reread."""
import csv
import hashlib
import json
import math
import zipfile
from sklearn.metrics import f1_score
from audit_support import *

ALLOW=['SOURCE_PUSH_BEFORE_AUDIT.json','TESTS.json','ROLE_SCOPE.csv','CAL_STABILITY.csv','RAW_ANCHOR_DRIFT.csv',
       'LABEL_FREE_SCORE_SUMMARY.csv','CAL_SELECTED_REPLAY.csv','RUN_STATUS.json','INDEPENDENT_VERIFICATION.json','DIAGNOSTIC_REPORT.md']


def check(row,key,value,tol=1e-12):
    assert math.isfinite(float(value)) and abs(float(row[key])-float(value))<=tol,(key,row[key],value)


def main():
    binding,core,fac,records,rec,banks,fr=setup();store=inputs();seal=jr(ROOT/'AUDIT_CACHE_SEAL_PRIVATE.json')
    assert seal['input']==digest(binding) and seal['all45_before_aggregate_reporting'] and len(seal['files'])==45
    cells=[r for r in fr['cells'] if r['arm'] in ARMS]
    stability=table('CAL_STABILITY');latent=table('RAW_ANCHOR_DRIFT');scores=table('LABEL_FREE_SCORE_SUMMARY');replay=table('CAL_SELECTED_REPLAY')
    assert len(stability)==len(replay)==45 and len(latent)==60 and len(scores)==75
    assert {(r['arm'],int(r['seed']),int(r['fold'])) for r in stability}==expected()
    assert {(r['arm'],int(r['seed']),int(r['fold']),r['phase']) for r in latent}=={(a,s,f,p) for a in LATENT for s in SEEDS for f in range(1,6) for p in ['TRAIN','CAL']}
    assert {(r['arm'],int(r['seed']),int(r['fold']),r['phase']) for r in scores}=={(a,s,f,p) for a in ARMS for s in SEEDS for f in range(1,6) for p in (['CAL'] if a=='B0' else ['TRAIN','CAL'])}
    def row_for(rows,r,phase=None):return next(z for z in rows if (z['arm'],int(z['seed']),int(z['fold']))==(r['arm'],r['seed'],r['fold']) and (phase is None or z['phase']==phase))
    loo_count=0;maximum_metric_error=0.
    for r in cells:
        d,scale=rdv.data(r['fold'],core,rec,banks,store,jr(rdv.ROOT/'INPUT_FREEZE_PRIVATE.json'))
        path=cell_path(r['arm'],r['seed'],r['fold']);assert sha(path)==seal['files'][f"{r['arm']}_seed{r['seed']}_fold{r['fold']}"]
        q=read(path);assert q['binding']==cell_binding(binding,r,d) and q['threshold']==r['threshold']
        old=read(r['calibration']);assert sha(r['calibration'])==r['calibration_sha256'] and sha(r['checkpoint'])==r['checkpoint_sha256']
        assert np.array_equal(q['cal_logit'],old['logit']) and np.array_equal(q['cal_indices'],d['archive_indices'][d['val']])
        assert np.array_equal(q['cal_y'],d['y'][d['val']]) and np.array_equal(q['cal_patient'],d['patient'][d['val']])
        y,pid,prob=q['cal_y'],q['cal_patient'],rdv.probability(q['cal_logit']);sel=core.select_threshold(y,prob,pid)
        assert sel['threshold']==r['threshold'];actual=row_for(stability,r);rr=row_for(replay,r)
        assert rr['stored_CAL_hash_verified']=='True' and rr['no_model_or_threshold_change']=='True'
        assert (rr['reused_stored_CAL_only']=='True')==(r['arm']=='B0')
        assert (rr['new_CAL_forward_bit_exact']=='True')==(r['arm']!='B0')
        assert actual['checkpoint_selected_on_all_CAL']=='True' and actual['not_independent_validation']=='True'
        taus=[];macros=[];ezs=[];originals=[];changed=[]
        for held in sorted(set(pid)):
            j=pid==held;fit=core.select_threshold(y[~j],prob[~j],pid[~j]);tau=fit['threshold'];taus.append(tau)
            pred=prob[j]>=tau;original=prob[j]>=r['threshold']
            macros.append(f1_score(y[j],pred,labels=[0,1],average='macro',zero_division=0))
            ezs.append(f1_score(1-y[j],1-pred.astype(int),zero_division=0))
            originals.append(f1_score(y[j],original,labels=[0,1],average='macro',zero_division=0))
            changed.append(float(np.mean(pred!=original)));loo_count+=1
        assert len(taus)==8 and int(actual['cal_patients'])==8 and int(actual['cal_channels'])==len(y)
        expected_values=dict(selected_threshold=r['threshold'],cal_macro_f1=np.mean(originals),leave_one_CAL_threshold_macro_f1=np.mean(macros),
                             difference=np.mean(macros)-np.mean(originals),loo_threshold_min=min(taus),loo_threshold_q25=np.quantile(taus,.25),
                             loo_threshold_median=np.median(taus),loo_threshold_q75=np.quantile(taus,.75),loo_threshold_max=max(taus),
                             loo_threshold_sd=np.std(taus,ddof=1),mean_abs_threshold_change=np.mean(np.abs(np.array(taus)-r['threshold'])),
                             mean_fraction_held_decisions_changed=np.mean(changed))
        for k,v in expected_values.items():
            check(actual,k,v);maximum_metric_error=max(maximum_metric_error,abs(float(actual[k])-v))
        phases=q['phases'] if r['arm']!='B0' else {'CAL':dict(indices=q['cal_indices'],patient=pid,logit=q['cal_logit'])}
        for phase,values in phases.items():
            ix=d['train'] if phase=='TRAIN' else d['val'];assert phase in ['TRAIN','CAL']
            assert np.array_equal(values['indices'],d['archive_indices'][ix]) and np.array_equal(values['patient'],d['patient'][ix])
            assert set(values['patient'])==d['roles']['train' if phase=='TRAIN' else 'cal']
            assert set(values['patient']).isdisjoint(d['roles']['assessment']|d['roles']['test'])
            z=np.asarray(values['logit'],float);score=rdv.probability(values['logit']);loc=np.array([np.median(z[values['patient']==p]) for p in np.unique(values['patient'])]);out=row_for(scores,r,phase)
            assert int(out['channels'])==len(z) and int(out['patients'])==len(loc)
            expected_values=dict(logit_mean=z.mean(),logit_population_sd=z.std(ddof=0),patient_logit_median_q25=np.quantile(loc,.25),
                                 patient_logit_median_median=np.median(loc),patient_logit_median_q75=np.quantile(loc,.75),patient_logit_median_sd=np.std(loc,ddof=1),
                                 fraction_abs_logit_above20=np.mean(np.abs(z)>20),probability_saturation_fraction=np.mean((score==0)|(score==1)),
                                 unique_logit_fraction=len(np.unique(z))/len(z),unique_probability_fraction=len(np.unique(score))/len(z))
            for k,v in expected_values.items():check(out,k,v,1e-12+1e-10*abs(v))
            if r['arm'] in LATENT:
                assert q['model_state_unchanged'] and np.array_equal(q['phases']['CAL']['logit'],q['cal_logit'])
                anchor=read(rdv.anchor_path(r['fold'],r['seed'],'pre'))
                assert np.array_equal(q['anchor_mean'],anchor['mean'].numpy()) and np.array_equal(q['anchor_scale'],anchor['scale'].numpy())
                g=(np.asarray(values['v'],np.float64)-q['anchor_mean'])/q['anchor_scale'];out=row_for(latent,r,phase)
                assert g.shape==(len(z),96) and np.isfinite(g).all() and int(out['valid_windows'])>0
                expected_values=dict(mean_coordinate_rms=np.linalg.norm(g.mean(axis=0))/np.sqrt(96),
                                     population_sd_coordinate_rms=np.linalg.norm(np.sqrt(np.var(g,axis=0,ddof=0)))/np.sqrt(96),
                                     abs_q50=np.quantile(abs(g),.5),abs_q95=np.quantile(abs(g),.95),abs_q99=np.quantile(abs(g),.99),abs_max=abs(g).max(),
                                     fraction_abs_above5=np.mean(abs(g)>5),fraction_abs_above10=np.mean(abs(g)>10),
                                     anchor_scale_min=q['anchor_scale'].min(),anchor_scale_median=np.median(q['anchor_scale']),anchor_scale_max=q['anchor_scale'].max())
                for k,v in expected_values.items():check(out,k,v,1e-12+1e-10*abs(v))
        print('TRAINCAL_INDEPENDENT_CELL_PASS',r['arm'],r['seed'],r['fold'],flush=True)
    assert loo_count==360
    write(PUB/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',selected_CAL_cells=45,leave_one_CAL_checks=360,latent_phase_cells=60,score_phase_cells=75,
                                                 maximum_CAL_metric_discrepancy=maximum_metric_error,all_CAL_forward_or_prior_reuse_bindings_verified=True,
                                                 TRAIN_CAL_only=True,neural_fits=0,assessment_metrics=0,new_Outer=False,active_goal_achieved=False))
    lines=['# Frozen dual-view TRAIN/CAL diagnostic','',
           'Zero neural fits,zero deployment threshold changes,no ASSESS/Development/Outer scoring. These are diagnostic cell-equal summaries,not patient-average goal evidence. The selected checkpoints had already used all8CAL patients for epoch selection;leave-oneCAL threshold sensitivity is selection-conditioned,not independent validation.','',
           '| Body | Mean CAL Macro-F1 | Mean leave-oneCAL threshold Macro-F1 | Mean threshold SD | Mean changed held-CAL decisions |',
           '| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        rows=[r for r in stability if r['arm']==arm]
        lines.append('| '+arm+' | '+' | '.join(f"{np.mean([float(r[k]) for r in rows]):.6f}" for k in ['cal_macro_f1','leave_one_CAL_threshold_macro_f1','loo_threshold_sd','mean_fraction_held_decisions_changed'])+' |')
    lines+=['','## Initial-anchor standardized raw summaries','',
            '| Body/phase | Mean coordinate-mean RMS | Mean coordinate-SD RMS | Mean fraction abs(g)>5 | Mean fraction abs(g)>10 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for arm in LATENT:
        for phase in ['TRAIN','CAL']:
            rows=[r for r in latent if r['arm']==arm and r['phase']==phase]
            lines.append('| '+arm+'/'+phase+' | '+' | '.join(f"{np.mean([float(r[k]) for r in rows]):.6f}" for k in ['mean_coordinate_rms','population_sd_coordinate_rms','fraction_abs_above5','fraction_abs_above10'])+' |')
    lines+=['','Initial anchors were fitted with the untrained encoder,not the evolving selected encoder. Drift is descriptive and is not proof that normalization caused the F1 decline or that refitting anchors would improve it. No patient or coordinate is excluded by the5/10cutoffs.','',
            'Historical Raw-only margin-grid/patientZ/fixed-fraction calibration failed to provide reliable F1 gains;A0Gaussian posterior identification also failed. These are different bodies,not a theorem against all calibration,but prevent claiming a ready-made remedy. No new decoder or architecture is selected from this diagnostic.','',
            'Independent verification covers45selectedCALcells,360leave-oneCALthreshold/metric checks,60raw-statphase cells and75unlabeled-scorephase cells. Private derived arrays and identities stay on the server;only anonymous tables,hashes andthisreport are exported. Active goal remains unachieved.','']
    (PUB/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','y','path'}
    for name in ALLOW:
        p=PUB/name;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(newline='',encoding='utf-8') as f:assert not set(next(csv.reader(f)))&forbidden
    manifest=dict(status='EXPLICIT_ANONYMOUS_ALLOWLIST',files={n:sha(PUB/n) for n in ALLOW},private_artifacts_exported=False)
    target=ROOT/'public_diagnostic_export_v1.zip'
    if target.exists():
        with zipfile.ZipFile(target) as z:assert json.loads(z.read('PACKAGE_MANIFEST.json'))==manifest
    else:
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n in ALLOW:z.write(PUB/n,n)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print('TRAINCAL_DIAGNOSTIC_ALL45_360_60_75_PASS',sha(target),target.stat().st_size,flush=True)


if __name__=='__main__':main()
