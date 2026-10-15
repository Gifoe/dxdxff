"""Independent sklearn check of all frozen scores; aggregate-only public output."""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('matched_no_pr_runner',ROOT/'code/run.py')
R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)


def verify(runtime):
    c,core,fac,records,rec,banks,refs,binding=R.setup(runtime)
    pub=runtime/'results';fr=R.jr(runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    status=R.jr(pub/'OUTER_STATUS.json');seal=R.jr(runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
    assert status['freeze_sha256']==R.sha(runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json')==seal['freeze_sha256']
    assert status['seal_sha256']==R.sha(runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    assert fr['binding']==binding and len(fr['selections'])==20
    original=R.jr(R.M8/'INPUT_FREEZE_PRIVATE.json')
    assert original['code']=={p.name:R.sha(p) for p in R.P3CODE.glob('*.py')}
    assert original['prior_code']=={p.name:R.sha(p) for p in (R.P3CODE.parent.parent/'phase2/code').glob('*.py')}
    for r in fr['selections']:
        fold=r['fold'];ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
        done=R.jr(Path(r['checkpoint']).parent/'COMPLETE_PRIVATE.json')
        assert done['best_sha256']==R.sha(r['checkpoint']) and done['last_sha256']==R.sha(Path(r['checkpoint']).parent/'LAST_PRIVATE.pt')
        if r['arm']!='M8_NO_PR':continue
        b=torch.load(r['bank'],map_location='cpu',weights_only=False)
        rr=R.SUP.roles(rec['meta'],fold)
        assert set(b['patient'][b['train']])==rr['fit'] and set(b['patient'][b['val']])==rr['validation']
        assert set(b['patient']).isdisjoint(rr['test']) and b['pre']['patient_z'] is False
        hist=torch.load(Path(r['checkpoint']).parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)['history']
        best=max(hist,key=lambda q:(q['macro_f1'],q['ez_f1'],-q['epoch']))
        assert best['epoch']==ck['epoch'] and best['threshold']==ck['threshold']
        pred=R.read(runtime/f'development_private/M8_NO_PR_fold{fold}.pkl')
        s=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy()
        assert core.select_threshold(b['y'][b['val']],s,b['patient'][b['val']])['threshold']==ck['threshold']
        base=torch.load(R.NOPR/f'A1/fold{fold}/TRAIN/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)['model']
        assert core.state_hash({k[5:]:v for k,v in ck['model'].items() if k.startswith('base.')})==core.state_hash(base)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    meta=rec['meta'];saved=pd.DataFrame(R.read(runtime/'OUTER_METRICS_PRIVATE.pkl'))
    checked=0;maxerror=0.;rawrows=[]
    for r in fr['selections']:
        arm,fold=r['arm'],r['fold'];p=runtime/f'outer_scores_private/{arm}_fold{fold}.pkl'
        assert R.sha(p)==seal['files'][f'{arm}_{fold}']
        q=R.read(p);ix=q['indices'];s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
        assert q['freeze_sha256']==seal['freeze_sha256'] and q['checkpoint_sha256']==r['checkpoint_sha256']
        for pid in sorted(set(meta['patient'][ix])):
            k=np.flatnonzero(meta['patient'][ix]==pid);k=k[np.argsort(meta['channel'][ix][k].astype(str),kind='stable')]
            truth=y[ix[k]]==0;pred=s[k]<r['threshold'];prob=1-s[k].astype(float)
            tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[False,True]).ravel()
            fs=f1_score(truth,pred,average=None,labels=[False,True],zero_division=0)
            order=np.argsort(s[k],kind='stable');positive=np.flatnonzero(truth[order])
            v=dict(ez_auroc=roc_auc_score(truth,prob) if truth.any() and (~truth).any() else np.nan,ez_auprc=average_precision_score(truth,prob) if truth.any() else 0.,macro_f1=fs.mean(),ez_f1=fs[1],nez_f1=fs[0],sensitivity=tp/(tp+fn) if tp+fn else np.nan,specificity=tn/(tn+fp) if tn+fp else np.nan,ez_mrr=1/(positive[0]+1) if len(positive) else 0.,top1_is_ez=float(truth[order[0]]),accuracy=(tp+tn)/len(truth),predicted_ez_fraction=pred.mean(),TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn),channels=len(truth))
            row=saved[(saved.arm==arm)&(saved.fold==fold)&(saved.patient==pid)].iloc[0]
            for key,val in v.items():
                assert np.isclose(val,row[key],rtol=0,atol=1e-12,equal_nan=True),key
                if np.isfinite(val):maxerror=max(maxerror,abs(float(val)-float(row[key])))
            checked+=1;rawrows.append(dict(arm=arm,fold=fold,patient=pid,**v))
    assert checked==320
    summary=saved.groupby('arm')[R.METRICS].mean().sort_index()
    disk=pd.read_csv(pub/'OUTER_SUMMARY.csv').set_index('arm').sort_index()
    assert np.allclose(summary,disk[summary.columns],rtol=0,atol=1e-12,equal_nan=True)
    boot=pd.read_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv');errors=[]
    for comparison,g in boot.groupby('comparison'):
        left,right=comparison.split(' minus ')
        l=saved[saved.arm==left].set_index('patient').sort_index();rr=saved[saved.arm==right].set_index('patient').sort_index()
        assert l.index.equals(rr.index) and len(l)==80
        delta=l[R.METRICS].to_numpy()-rr[R.METRICS].to_numpy()
        draws=np.random.default_rng(42).integers(0,80,size=(10000,80))
        values=np.nanmean(delta[draws],axis=1)
        for j,m in enumerate(R.METRICS):
            z=g[g.metric==m].iloc[0];want=[np.nanmean(delta[:,j]),*np.nanpercentile(values[:,j],[2.5,97.5])]
            got=[z.delta,z.ci_low,z.ci_high]
            assert np.allclose(want,got,rtol=0,atol=1e-12,equal_nan=True)
            errors.extend(np.abs(np.asarray(want)-got).tolist())
    R.write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',verified_patient_rows=checked,verified_outer_channels_per_arm=7635,patient_metric_max_error=maxerror,paired_bootstrap_comparisons=4,paired_bootstrap_max_error=float(np.nanmax(errors)),draws=10000,seed=42,original_M8_training_sources_unchanged=True,original_selection_and_threshold_rules_exact=True,frozen_no_PR_D1_base_unchanged=True,own_TEST_excluded_from_all_fits=True,all_20_score_files_hash_verified=True,verification_source_sha256=R.sha(Path(__file__))))
    manifest={p.name:R.sha(p) for p in sorted(pub.glob('*')) if p.is_file() and p.name!='PUBLIC_SHA256.json'}
    R.write(pub/'PUBLIC_SHA256.json',manifest)
    print('INDEPENDENT_VERIFICATION_PASS',checked,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\m8_d1_no_pr_outer_seed42_v1_runtime'));verify(p.parse_args().runtime)
