"""Independent sklearn/cluster-count verification of sealed Outer; no inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluate_outer_m8 import prepare,cached,sha,read,write

def main(a):
    lock,c,core,fac,records,rec,banks,sels,binding,pub=prepare(a)
    fp=a.out/'FROZEN_BEFORE_OUTER_PRIVATE.json';fh=sha(fp);fr=json.loads(fp.read_text())
    assert fr['binding']==binding and fr['selections']==sels
    status=json.loads((pub/'OUTER_STATUS.json').read_text());seal=a.out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';ss=json.loads(seal.read_text())
    assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE' and status['freeze_sha256']==fh and status['seal_sha256']==sha(seal)
    assert ss['freeze_sha256']==fh and len(ss['files'])==10
    from explore import original_modules
    metrics,boot=original_modules()
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    meta=rec['meta'];rs=[];scores={};decisions={}
    for r in sels:
        arm,fold=r['arm'],r['fold'];pp=a.out/f'scores_private/{arm}_fold{fold}.pkl'
        assert sha(pp)==ss['files'][f'{arm}_{fold}'];q=read(pp);ix=q['indices'];cached(pp,fh,r['checkpoint_sha256'],ix)
        s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();tau=r['threshold']
        if arm not in scores:scores[arm]=np.full(7635,np.nan);decisions[arm]=np.zeros(7635,bool)
        assert np.isnan(scores[arm][ix]).all();scores[arm][ix]=s;decisions[arm][ix]=s<tau
        for pid in sorted(set(meta['patient'][ix])):
            k=np.flatnonzero(meta['patient'][ix]==pid);k=k[np.argsort(meta['channel'][ix][k].astype(str),kind='stable')];ii=ix[k]
            rs.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ii[0]]),**metrics.independent(y[ii],s[k],tau)))
    df=pd.DataFrame(rs);saved=pd.DataFrame(read(a.out/'OUTER_METRICS_PRIVATE.pkl'));keys=['arm','fold','patient']
    l=df.set_index(keys).sort_index();r=saved.set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[metrics.METRICS],r[metrics.METRICS],atol=1e-12,rtol=0,equal_nan=True)
    for arm,g in df.groupby('arm'):assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
    for name,keys in [('OUTER_SUMMARY.csv',['arm']),('OUTER_BY_FOLD.csv',['arm','fold']),('OUTER_BY_CENTER.csv',['arm','center'])]:
        v=pd.read_csv(pub/name).set_index(keys).sort_index();q=df.groupby(keys)[metrics.METRICS].mean().sort_index()
        assert v.index.equals(q.index) and np.allclose(v[metrics.METRICS],q[metrics.METRICS],atol=1e-12,rtol=0,equal_nan=True)
    paired=pd.read_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv').set_index('metric')
    ll=df[df.arm=='M8_MACRO_AWARE'].set_index('patient').sort_index();rr=df[df.arm=='D1'].set_index('patient').sort_index();assert ll.index.equals(rr.index)
    delta=ll[metrics.METRICS].to_numpy()-rr[metrics.METRICS].to_numpy();ids=np.random.default_rng(42).integers(0,80,size=(10000,80))
    counts=np.stack([(ids==j).sum(1) for j in range(80)],axis=1);b=counts@delta/80
    for j,m in enumerate(metrics.METRICS):
        assert np.allclose(paired.loc[m,['delta','ci_low','ci_high']].to_numpy(float),[delta[:,j].mean(),*np.percentile(b[:,j],[2.5,97.5])],rtol=0,atol=1e-12)
    absolute=pd.read_csv(pub/'OUTER_ABSOLUTE_INTERVALS.csv').set_index(['arm','metric'])
    for arm,g in df.groupby('arm'):
        v=g.set_index('patient').sort_index()[metrics.METRICS].to_numpy();b=counts@v/80
        for j,m in enumerate(metrics.METRICS):assert np.allclose(absolute.loc[(arm,m),['estimate','ci_low','ci_high']].to_numpy(float),[v[:,j].mean(),*np.percentile(b[:,j],[2.5,97.5])],rtol=0,atol=1e-12)
    pooled=pd.read_csv(pub/'OUTER_POOLED_CHANNELS.csv').set_index('arm')
    for arm,s in scores.items():
        assert np.isfinite(s).all();z=y==0;p=decisions[arm];tp=int((z&p).sum());fpv=int((~z&p).sum());tn=int((~z&~p).sum());fn=int((z&~p).sum());ef=2*tp/(2*tp+fpv+fn);nf=2*tn/(2*tn+fpv+fn)
        q=metrics.independent(y,s,.5);q.update(TP=tp,FP=fpv,TN=tn,FN=fn,macro_f1=(ef+nf)/2,ez_f1=ef,nez_f1=nf,sensitivity=tp/(tp+fn),specificity=tn/(tn+fpv),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fpv))/2,accuracy=(tp+tn)/7635,predicted_ez_fraction=p.mean());q.pop('ez_mrr');q.pop('top1_is_ez')
        for k,v in q.items():assert np.isclose(pooled.loc[arm,k],v,rtol=0,atol=1e-12)
    write(pub/'OUTER_INDEPENDENT_VERIFICATION.json',dict(status='PASS',patients_per_arm=80,channels_per_arm=7635,patient_metric_cells=160,model_fold_cells=10,independent_sklearn_metrics=True,independent_bootstrap_counts=True,pooled_confusions_checked=True,all_frozen_dependencies_unchanged=True,no_model_or_raw_replay=True,aggregate_sha256={p.name:sha(p) for p in pub.glob('*.csv')}))
    print('M8_D1_OUTER_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));p.add_argument('--out',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime'));main(p.parse_args())
