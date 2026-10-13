"""Independent sklearn metrics and identity-cluster intervals from sealed scores.

No inference, refitting, threshold selection, or outer outcome access.
Run in the separately validated analysis environment.
"""
import json
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,confusion_matrix
from runtime import args,setup,checkpoint


def independent(y,score,tau):
    z=np.asarray(y)==0;p=np.asarray(score)<tau
    tn,fp,fn,tp=confusion_matrix(z,p,labels=[False,True]).ravel()
    f=f1_score(z,p,labels=[False,True],average=None,zero_division=0)
    se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan
    order=np.argsort(score,kind='stable');positive=np.flatnonzero(z[order])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ap=average_precision_score(z,-np.asarray(score)) if z.any() else 0.
    return dict(macro_f1=f.mean(),ez_f1=f[1],nez_f1=f[0],balanced_accuracy=np.nanmean([se,sp]),
        ez_auprc=ap,ez_auroc=roc_auc_score(z,-np.asarray(score)) if z.any() and (~z).any() else np.nan,
        ez_mrr=1/(positive[0]+1) if len(positive) else 0.,top1_is_ez=float(z[order[0]]),
        sensitivity=se,specificity=sp,accuracy=(tp+tn)/len(y),TP=tp,FP=fp,TN=tn,FN=fn,
        channels=len(y),predicted_ez_fraction=p.mean(),observed_ez_fraction=z.mean())


def intervals(left,right,metrics):
    ll=left.set_index(['fold','patient']).sort_index();rr=right.set_index(['fold','patient']).sort_index()
    assert ll.index.equals(rr.index)
    delta=ll[metrics].to_numpy()-rr[metrics].to_numpy();patient=np.asarray(ll.index.get_level_values('patient')).astype(str)
    ids=np.unique(patient);assert len(ids)==47 and len(patient)==65
    draws=np.random.default_rng(42).integers(0,47,size=(10000,47))
    # Independent multiplicity-weighted implementation: each sampled identity
    # retains every fold appearance; denominator is drawn finite appearances.
    counts=np.stack([(draws==i).sum(1) for i in range(47)],axis=1)
    totals=np.stack([np.nansum(delta[patient==p],axis=0) for p in ids])
    n=np.stack([np.isfinite(delta[patient==p]).sum(0) for p in ids])
    den=counts@n;num=counts@totals
    boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    return {m:(np.nanmean(delta[:,j]),*np.nanpercentile(boot[:,j],[2.5,97.5])) for j,m in enumerate(metrics)}


def main():
    a=args();lock,ref,c,core,original,trainer,models,banks,binding=setup(a);pub=a.runtime/'public'
    pre=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());eng=json.loads((pub/'ENGINEERING_TESTS.json').read_text())
    assert pre['status']==eng['status']=='PASS' and pre['binding']==eng['binding']==binding
    frozen=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert frozen['binding']==binding
    fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json');seal=json.loads((a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json').read_text())
    assert seal['binding']==fs and not seal['outer_accessed']
    recalculated=[];maxdrift=0.;cells=0
    for fold in range(1,6):
        bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
        d=torch.load(bp,map_location='cpu',weights_only=False);va=d['val']
        for arm in lock['arms']:
            rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold)
            cp=checkpoint(a,arm,fold);assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256']
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            factory=models.constructor(original.SourceA0,arm,d['stats_normalizer']);m=factory();m.load_state_dict(ck['model'],strict=True)
            assert sum(p.numel() for p in m.parameters())==lock['parameters'][arm]
            if arm=='B3':
                assert np.array_equal(m.stat_mean.numpy(),d['stats_normalizer']['mean'])
                assert np.array_equal(m.stat_scale.numpy(),d['stats_normalizer']['scale'])
            path=bp.parent/f'{arm}_VAL_PRIVATE.pt';assert c.sha(path)==seal['files'][f'{arm}_{fold}']
            cell=torch.load(path,map_location='cpu',weights_only=False);assert np.array_equal(cell['indices'],va)
            assert cell['checkpoint_sha256']==c.sha(cp) and cell['threshold']==ck['threshold']==rec['threshold']
            if arm=='B0':assert cell['binding']==binding
            else:assert cell['binding']==fs
            variants=[(arm,cell['scores'])]
            if arm=='B3':
                mp=bp.parent/'MECHANISM_PRIVATE.pt';assert c.sha(mp)==seal['files'][f'mechanism_{fold}']
                mm=torch.load(mp,map_location='cpu',weights_only=False)
                assert mm['binding']==fs and mm['threshold']==ck['threshold'] and np.array_equal(mm['indices'],va)
                assert np.array_equal(mm['full_scores'],cell['scores']) and mm['x_and_h_unchanged'] and mm['stats_multiset_preserved']
                for pat in np.unique(d['patient'][va]):
                    ix=np.flatnonzero(d['patient'][va]==pat);assert sorted(mm['permutation'][ix])==ix.tolist()
                variants.extend([('B3_SHUFFLED',mm['shuffled_scores']),('B3_DISABLED',mm['disabled_scores'])])
            for variant,scores in variants:
                rows=[]
                for pat in sorted(set(d['patient'][va])):
                    k=np.flatnonzero(d['patient'][va]==pat);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')]
                    ix=va[k];y=d['y'][ix];score=scores[k]
                    independent_metrics=independent(y,score,ck['threshold']);existing=c.metric(y,score,ck['threshold'])
                    for key in independent_metrics:
                        assert np.isclose(independent_metrics[key],existing[key],atol=1e-12,rtol=0,equal_nan=True),(fold,variant,key)
                        if np.isfinite(independent_metrics[key]):maxdrift=max(maxdrift,abs(independent_metrics[key]-existing[key]))
                    rows.append(dict(arm=variant,fold=fold,patient=pat,center=str(d['center'][ix[0]]),**independent_metrics))
                assert len(rows)==13;cells+=len(rows);recalculated.extend(rows)
                if variant==arm:
                    assert abs(np.mean([r['macro_f1'] for r in rows])-ck['macro_f1'])<1e-12
                    assert abs(np.mean([r['ez_f1'] for r in rows])-ck['ez_f1'])<1e-12
    df=pd.DataFrame(recalculated);assert cells==390
    for filename,keys,subset in [
        ('DEVELOPMENT_SUMMARY.csv',['arm'],df[df.arm.isin(lock['arms'])]),
        ('DEVELOPMENT_BY_FOLD.csv',['arm','fold'],df[df.arm.isin(lock['arms'])]),
        ('DEVELOPMENT_BY_CENTER.csv',['arm','center'],df[df.arm.isin(lock['arms'])]),
        ('MECHANISM_SUMMARY.csv',['arm'],df[df.arm.isin(['B3','B3_SHUFFLED','B3_DISABLED'])])]:
        actual=pd.read_csv(pub/filename).set_index(keys).sort_index();expected=subset.groupby(keys)[c.METRICS].mean().sort_index()
        assert actual.index.equals(expected.index)
        assert np.allclose(actual[c.METRICS],expected,atol=1e-12,rtol=0,equal_nan=True),filename
    for filename,pairs in [('PAIRED_BOOTSTRAP.csv',lock['comparisons']),
        ('MAGNITUDE_SHUFFLE_CONTROL.csv',[['B3','B3_SHUFFLED'],['B3','B3_DISABLED']])]:
        actual=pd.read_csv(pub/filename)
        for left,right in pairs:
            calculated=intervals(df[df.arm==left],df[df.arm==right],c.METRICS)
            for metric,values in calculated.items():
                row=actual[(actual.comparison==left+' minus '+right)&(actual.metric==metric)].iloc[0]
                assert np.allclose(values,[row.delta,row.ci_low,row.ci_high],atol=1e-12,rtol=0,equal_nan=True),(filename,metric)
    t=df[df.arm.isin(lock['arms'])].groupby('arm')[c.METRICS].mean()
    f=df[df.arm.isin(lock['arms'])].groupby(['fold','arm']).ez_auroc.mean().unstack()
    criteria=dict(B3_minus_B0_AUROC_ge_0_010=t.loc['B3','ez_auroc']-t.loc['B0','ez_auroc']>=.010,
        B3_minus_B2_AUROC_ge_0_005=t.loc['B3','ez_auroc']-t.loc['B2','ez_auroc']>=.005,
        B3_AP_not_lower_than_B0=t.loc['B3','ez_auprc']>=t.loc['B0','ez_auprc'],
        B3_MacroF1_delta_ge_minus_0_005=t.loc['B3','macro_f1']-t.loc['B0','macro_f1']>=-.005,
        B3_EZF1_delta_ge_minus_0_005=t.loc['B3','ez_f1']-t.loc['B0','ez_f1']>=-.005,
        positive_AUROC_folds_ge4=(f.B3>f.B0).sum()>=4,engineering_integrity_numerical_checks=True)
    gate=json.loads((pub/'VALIDATION_GATE.json').read_text())
    assert list(map(bool,criteria.values()))==list(map(bool,[gate['criteria'][k] for k in [
        'B3_minus_B0_AUROC_ge_0.010','B3_minus_B2_AUROC_ge_0.005','B3_AP_not_lower_than_B0',
        'B3_MacroF1_delta_ge_minus_0.005','B3_EZF1_delta_ge_minus_0.005','positive_AUROC_folds_ge4','engineering_integrity_numerical_checks']]))
    assert gate['pass']==all(criteria.values()) and setup(a)[-1]==binding
    c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',metric_backend='independent sklearn',
        patient_metric_cells=390,max_absolute_metric_drift=maxdrift,paired_intervals_verified=66,
        unique_ID_cluster_bootstrap_draws=10000,cluster_IDs=47,retained_appearances=65,
        parameters_and_stat_buffers_verified=True,checkpoint_threshold_bank_seal_hashes_verified=True,
        all_aggregate_and_gate_parity=True,outer_accessed=False,development_freeze_sha256=fs,
        analysis_environment=dict(torch=torch.__version__,numpy=np.__version__,sklearn=__import__('sklearn').__version__)))
    print('INDEPENDENT_PMA_VERIFICATION_PASS',cells,maxdrift,flush=True)


if __name__=='__main__':main()
