"""Independent rank/count endpoints, paired intervals, and coverage checks."""
import sys,csv,ast
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from runtime import *
from metrics import KEYS,patients

def csv_out(name,rows):
    assert rows and all(set(r)==set(rows[0]) for r in rows),name
    path=PUBLIC/(name+'.csv');path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def independent(y,score,p,t,channel,kind):
    y=np.asarray(y,int);n=len(y);n1=int(y.sum());n0=n-n1;assert n1 and n0
    ranks=rankdata(score,method='average');auc=(ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
    order=np.argsort(-score,kind='stable');truth=y[order];s=score[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),n-1];tp_curve=np.cumsum(truth)[ends]
    ap=np.sum(tp_curve/(ends+1)*np.diff(np.r_[0,tp_curve])/n1)
    prediction=np.asarray(p)>=t if kind=='EZ_GE' else (1-np.asarray(p))<t
    correct_pos=np.count_nonzero(prediction&(y==1));correct_neg=np.count_nonzero(~prediction&(y==0))
    positive_pred=int(prediction.sum());f1=2*correct_pos/(n1+positive_pred);f0=2*correct_neg/(n0+n-positive_pred)
    ranking=np.lexsort((np.asarray(channel,str),-score));first=np.flatnonzero(y[ranking])[0]
    return dict(auroc=float(auc),ap=float(ap),macro_f1=(f1+f0)/2,ez_f1=f1,nez_f1=f0,
                accuracy=(correct_pos+correct_neg)/n,mrr=1/(int(first)+1),top1=float(y[ranking[0]]),
                ez_fraction_bias=positive_pred/n-n1/n)

def pair_audit(y,anchor,rank,dr):
    before=anchor[y==1,None]-anchor[y==0][None,:];after=rank[y==1,None]-rank[y==0][None,:]
    den=before.size;old=(before>0)+.5*(before==0);new=(after>0)+.5*(after==0)
    corrected=np.count_nonzero((before<=0)&(after>0));broken=np.count_nonzero((before>0)&(after<=0))
    order_changed=not np.array_equal(np.argsort(anchor,kind='stable'),np.argsort(rank,kind='stable'))
    return dict(mean_abs_residual=float(np.abs(dr).mean()),within_patient_residual_sd=float(dr.std(ddof=0)),
                ranking_changed=float(order_changed),corrected_pair_fraction=corrected/den,
                newly_broken_pair_fraction=broken/den,net_correct_pair_fraction=float((new-old).mean()),
                corrected_pairs=int(corrected),newly_broken_pairs=int(broken),ez_nez_pairs=int(den))

def main(stage):
    fr=read_json(ROOT/(stage+'_FREEZE.json'));seal=read_json(ROOT/(stage+'_SCORE_SEAL.json'))
    assert fr['source']==source_hashes() and seal['freeze']==sha(ROOT/(stage+'_FREEZE.json'))
    assert seal['all_scores_sealed_before_metrics']
    for name in ['train_svm_plugin.py','train_drechan_joint.py','training.py']:
        tree=ast.parse((HERE/name).read_text())
        for call in [n for n in ast.walk(tree) if isinstance(n,ast.Call)]:
            assert not any(k.arg in ['outer','labels'] and isinstance(k.value,ast.Constant) and k.value.value is True for k in call.keywords)
    rows=[];audits=[];calibration=[];max_error=0.;coverage=[]
    for cell in seal['files']:
        assert sha(cell['path'])==cell['sha256'];q=load(cell['path']);fold=cell['fold'];out=q['components']
        data=bank_data(fold,outer=True,labels=True);assert np.array_equal(q['archive_indices'],data['archive_indices'])
        expected=patients(data,out,cell['threshold'],cell['threshold_kind'])
        p=out.get('probability_exact',expit(out['final']))
        for patient,reference in zip(sorted(set(data['patient'])),expected):
            ix=data['patient']==patient;y=data['y'][ix];rank=out['rank_score'][ix];ch=data['channel'][ix]
            check=independent(y,rank,p[ix],cell['threshold'],ch,cell['threshold_kind'])
            error=max(abs(reference[k]-check[k]) for k in KEYS);assert error<1e-12;max_error=max(max_error,error)
            rows.append(dict(stage=stage,arm=cell['arm'],fold=fold,patient=str(patient),source=str(data['source'][ix][0]),channels=int(ix.sum()),**check))
            if cell['arm'] in ['S3_EXPANDED_ABS','S4_PR2_RANK','S5_PR2_FULL','D1_EXPANDED_ABS','D2_PR2_FULL']:
                a=pair_audit(y,out['anchor'][ix],rank,out['delta_rank'][ix]);a.update(stage=stage,arm=cell['arm'],patient=str(patient),fold=fold)
                # Weighted correctly-ordered pair change must equal independent AUROC gain.
                baseline=independent(y,out['anchor'][ix],p[ix],cell['threshold'],ch,cell['threshold_kind'])
                assert abs(check['auroc']-baseline['auroc']-a['net_correct_pair_fraction'])<1e-12;audits.append(a)
                translated=independent(y,out['final'][ix],p[ix],cell['threshold'],ch,cell['threshold_kind'])
                assert all(abs(translated[k]-check[k])<1e-12 for k in ['auroc','ap','mrr','top1'])
                assert np.ptp(out['delta_cal'][ix])==0
                off=independent(y,rank,expit(out['rank'][ix]),cell['threshold'],ch,cell['threshold_kind'])
                calibration.append(dict(stage=stage,arm=cell['arm'],patient=str(patient),fold=fold,delta=float(out['delta_cal'][ix][0]),
                                        delta_macro_f1=check['macro_f1']-off['macro_f1'],delta_ez_f1=check['ez_f1']-off['ez_f1'],
                                        ranking_metric_error=max(abs(translated[k]-check[k]) for k in ['auroc','ap','mrr','top1'])))
        coverage.append(dict(arm=cell['arm'],fold=fold,patients=len(expected),channels=len(data['patient'])))
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80
    arms=sorted({r['arm'] for r in rows});matrices={}
    for arm in arms:
        selected=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['patient'])
        assert [r['patient'] for r in selected]==ids and sum(r['channels'] for r in selected)==7635
        matrices[arm]=np.array([[r[k] for k in KEYS] for r in selected]);assert np.isfinite(matrices[arm]).all()
    # Frozen legacy baseline row must reproduce the pre-existing historical aggregate.
    if stage=='A':
        path=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1_runtime\public\MAIN_RESULTS.csv')
        old=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        mapping={'auroc':'auroc','ap':'auprc','macro_f1':'macro_f1','ez_f1':'f1','nez_f1':'nez_f1','accuracy':'accuracy','mrr':'mrr','top1':'hit_at_1','ez_fraction_bias':'ez_fraction_bias'}
        for arm,legacy_arm in [('S0_NO_PR','Linear_SVM_NO_PR'),('S1_LEGACY_PR','Linear_SVM_PR')]:
            reference=next(r for r in old if r['arm']==legacy_arm)
            assert all(abs(matrices[arm].mean(0)[KEYS.index(k)]-float(reference[v]))<1e-12 for k,v in mapping.items())
        assert np.max(abs(matrices['S5_PR2_FULL'][:,:2]-matrices['S5_NO_CAL'][:,:2]))==0
    else:
        hist=matrices['D_HISTORICAL_FULL'].mean(0)
        assert abs(hist[KEYS.index('auroc')]-.7658883112922217)<1e-12
        assert abs(hist[KEYS.index('macro_f1')]-.6463387559404331)<1e-12
        for k in ['auroc','ap','mrr','top1']:
            assert np.array_equal(matrices['D2_PR2_FULL'][:,KEYS.index(k)],matrices['D2_NO_CAL'][:,KEYS.index(k)])
    private=dict(rows=rows,matrices=matrices,ranking=audits,calibration=calibration)
    save(ROOT/(stage+'_VERIFIED_ENDPOINTS_PRIVATE.pkl'),private)
    prefix='SVM' if stage=='A' else 'DRECHAN'
    model_rows=[];fold_rows=[];source_rows=[];source_list=['hup','lzu','multicenter','pediatric']
    for arm in arms:
        scientific=[r for r in fr['cells'] if r['arm']==arm]
        infer=not scientific
        base=scientific[0] if scientific else next(r for r in fr['cells'] if r['arm']==('S5_PR2_FULL' if stage=='A' else 'D2_PR2_FULL'))
        model_rows.append(dict(stage=stage,arm=arm,patients=80,channels=7635,added_parameters=int(base['added_parameters']),inference_only=infer,
                               **dict(zip(KEYS,map(float,matrices[arm].mean(0))))))
        for fold in range(1,6):
            z=[r for r in rows if r['arm']==arm and r['fold']==fold]
            fold_rows.append(dict(stage=stage,arm=arm,fold=fold,patients=len(z),**{k:float(np.mean([r[k] for r in z])) for k in KEYS}))
        for source in source_list:
            z=[r for r in rows if r['arm']==arm and r['source']==source]
            assert len(z)=={'hup':36,'lzu':21,'multicenter':15,'pediatric':8}[source]
            source_rows.append(dict(stage=stage,arm=arm,source=source,patients=len(z),**{k:float(np.mean([r[k] for r in z])) for k in KEYS}))
    csv_out(prefix+'_MAIN_RESULTS',model_rows);csv_out('FOLD_RESULTS_'+stage,fold_rows);csv_out('SOURCE_RESULTS_'+stage,source_rows)
    comparisons=([('S5_PR2_FULL',b) for b in ['S0_NO_PR','S1_LEGACY_PR','S2_OUTPUT_Z','S3_EXPANDED_ABS','S4_PR2_RANK','S5_NO_CAL']]
                 if stage=='A' else [('D2_PR2_FULL',b) for b in ['D_HISTORICAL_FULL','D0_RANK_LOSS','D1_EXPANDED_ABS','D2_NO_CAL','D2_NO_RANK','D2_NO_BOTH']]
                 +[('D0_RANK_LOSS','D_HISTORICAL_FULL'),('D1_EXPANDED_ABS','D0_RANK_LOSS')])
    draw=np.random.default_rng(42).integers(0,80,size=(10000,80));intervals=[];distributions=[]
    for a,b in comparisons:
        delta=matrices[a]-matrices[b];samples=np.concatenate([delta[draw[j:j+250]].mean(1) for j in range(0,10000,250)])
        bounds=np.quantile(samples,[.025,.975],axis=0)
        for j,k in enumerate(KEYS):
            directions=[next(r[k] for r in fold_rows if r['arm']==a and r['fold']==f)-next(r[k] for r in fold_rows if r['arm']==b and r['fold']==f) for f in range(1,6)]
            intervals.append(dict(stage=stage,comparison=a+'-'+b,metric=k,delta=float(delta[:,j].mean()),ci_low=float(bounds[0,j]),ci_high=float(bounds[1,j]),
                                  positive_patients=int((delta[:,j]>0).sum()),patient_positive_fraction=float((delta[:,j]>0).mean()),
                                  positive_folds=int(sum(v>0 for v in directions)),draws=10000,patients=80))
        z=delta[:,KEYS.index('auroc')]
        for quantile,value in zip([0,.1,.25,.5,.75,.9,1],np.quantile(z,[0,.1,.25,.5,.75,.9,1])):
            distributions.append(dict(stage=stage,comparison=a+'-'+b,metric='delta_auroc',quantile=quantile,value=float(value)))
    csv_out(prefix+'_PAIRED_CI',intervals);csv_out('PATIENT_EFFECT_DISTRIBUTION_'+stage,distributions)
    aggregate=[]
    for arm in sorted({r['arm'] for r in audits}):
        z=[r for r in audits if r['arm']==arm];assert len(z)==80
        fields=[k for k in z[0] if k not in ['stage','arm','patient','fold']]
        aggregate.append(dict(stage=stage,arm=arm,patients=80,**{k:float(np.mean([r[k] for r in z])) for k in fields}))
    csv_out('RANKING_CORRECTION_AUDIT_'+stage,aggregate)
    aggregate=[]
    for arm in sorted({r['arm'] for r in calibration}):
        z=[r for r in calibration if r['arm']==arm]
        aggregate.append(dict(stage=stage,arm=arm,patients=80,mean_calibration=float(np.mean([r['delta'] for r in z])),
                              mean_abs_calibration=float(np.mean([abs(r['delta']) for r in z])),
                              calibration_macro_f1_delta=float(np.mean([r['delta_macro_f1'] for r in z])),
                              calibration_ez_f1_delta=float(np.mean([r['delta_ez_f1'] for r in z])),
                              maximum_ranking_error=float(max(r['ranking_metric_error'] for r in z))))
    csv_out('CALIBRATION_AUDIT_'+stage,aggregate)
    parameter_rows=[{k:r.get(k,0) for k in ['arm','fold','stage','added_parameters','trainable_parameters','epoch','threshold','threshold_kind','reused','epochs','training_seconds']} for r in fr['cells']]
    csv_out('MODEL_PARAMETER_AUDIT_'+stage,parameter_rows);csv_out('COVERAGE_'+stage,coverage)
    write(PUBLIC/(stage+'_VERIFIED.json'),dict(status='PASS',stage=stage,patients=80,channels_each=7635,models=len(arms),
             verified_patient_rows=len(rows),verified_endpoint_values=len(rows)*9,maximum_independent_error=max_error,
             score_seal=sha(ROOT/(stage+'_SCORE_SEAL.json')),endpoint_private_sha256=sha(ROOT/(stage+'_VERIFIED_ENDPOINTS_PRIVATE.pkl'))))
    print('PR2_STAGE_INDEPENDENT_PASS',stage,len(rows),len(rows)*9,'max_error',max_error,flush=True)
    for row in model_rows:print('PR2_AGGREGATE',row['arm'],*[round(row[k],6) for k in ['auroc','ap','macro_f1']],flush=True)

if __name__=='__main__':main(sys.argv[1])
