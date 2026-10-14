"""Independent sealed-score finalOuter metrics/decisions/bootstrap audit."""
import json
import numpy as np
import pandas as pd
import torch
from evaluate import arguments,prepare,sha,read,write,original_modules,transform_scaled,apply_map,cutoff

def main(a):
    c,core,fac,records,rec,banks,sels,maps,binding,pub=prepare(a);fp=a.out/'FROZEN_BEFORE_OUTER_PRIVATE.json';fh=sha(fp);fr=json.loads(fp.read_text());assert fr['binding']==binding and fr['selections']==sels and fr['calibration_maps']==maps;ss=json.loads((a.out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json').read_text());status=json.loads((pub/'OUTER_STATUS.json').read_text());assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE' and status['freeze_sha256']==fh and ss['freeze_sha256']==fh and status['seal_sha256']==sha(a.out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json') and len(ss['files'])==20
    from pathlib import Path
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    meta=rec['meta'];m,b=original_modules();rows=[];pooled={}
    for arm in ['D1','M8_MACRO_AWARE','M18_TIME_CONTROL','FINAL_DECODER']:
        full=np.empty(7635);pred=np.zeros(7635,bool)
        for fold in range(1,6):
            base='M18_TIME_CONTROL' if arm=='FINAL_DECODER' else arm;pp=a.out/f'scores_private/{base}_fold{fold}.pkl';assert sha(pp)==ss['files'][f'{base}_{fold}'];q=read(pp);assert q['freeze_sha256']==fh and 'y' not in q;r=next(v for v in sels if v['arm']==base and v['fold']==fold);assert q['checkpoint_sha256']==r['checkpoint_sha256'];ix=q['indices'];expected=np.flatnonzero(np.isin(meta['patient'],list(banks[fold-1]['roles']['test'])));assert np.array_equal(ix,expected);s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
            if arm=='FINAL_DECODER':
                dp=a.out/f'decisions_private/FINAL_DECODER_fold{fold}.pkl';assert sha(dp)==ss['files'][f'FINAL_DECODER_{fold}'];dd=read(dp);assert dd['freeze_sha256']==fh and dd['score_sha256']==sha(pp) and np.array_equal(dd['indices'],ix);params=maps[fold-1];shift,median,scale=transform_scaled(s,meta['patient'][ix],params['coefficient'],params['exponent']);t=apply_map(meta['center'][ix],params['default'],params['mapping']);tau=cutoff(t,median,scale,params['coefficient']);assert np.array_equal(dd['cutoff'],tau) and np.array_equal(dd['prediction_nez'],s>=tau) and np.array_equal(shift>=t,s>=tau)
            else:tau=np.full(len(ix),r['threshold'])
            full[ix]=s;pred[ix]=s>=tau
            for pid in sorted(set(meta['patient'][ix])):
                k=np.flatnonzero(meta['patient'][ix]==pid);k=k[np.argsort(meta['channel'][ix][k].astype(str),kind='stable')];ii=ix[k];rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ii[0]]),**m.independent(y[ii],s[k],float(tau[k][0]))))
        ez=y==0;p=~pred;tp=int((ez&p).sum());fpv=int((~ez&p).sum());tn=int((~ez&~p).sum());fn=int((ez&~p).sum());ef=2*tp/(2*tp+fpv+fn);nf=2*tn/(2*tn+fpv+fn);v=m.independent(y,full,.5);v.update(TP=tp,FP=fpv,TN=tn,FN=fn,macro_f1=(ef+nf)/2,ez_f1=ef,nez_f1=nf,sensitivity=tp/(tp+fn),specificity=tn/(tn+fpv),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fpv))/2,accuracy=(tp+tn)/7635,predicted_ez_fraction=p.mean());v.pop('ez_mrr');v.pop('top1_is_ez');pooled[arm]=v
    df=pd.DataFrame(rows);saved=pd.DataFrame(read(a.out/'OUTER_METRICS_PRIVATE.pkl'));keys=['arm','fold','patient'];l=df.set_index(keys).sort_index();r=saved.set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[m.METRICS],r[m.METRICS],atol=1e-12,rtol=0,equal_nan=True)
    for arm,g in df.groupby('arm'):assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
    for name,keys in [('OUTER_SUMMARY.csv',['arm']),('OUTER_BY_FOLD.csv',['arm','fold']),('OUTER_BY_CENTER.csv',['arm','center'])]:
        table=pd.read_csv(pub/name).set_index(keys).sort_index();expected=df.groupby(keys)[m.METRICS].mean().sort_index();assert table.index.equals(expected.index) and np.allclose(table[m.METRICS],expected,atol=1e-12,rtol=0,equal_nan=True)
    bs=pd.read_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv')
    for left,right in [('FINAL_DECODER','D1'),('FINAL_DECODER','M8_MACRO_AWARE'),('FINAL_DECODER','M18_TIME_CONTROL'),('M18_TIME_CONTROL','D1')]:
        for row in b.compare(c,df[df.arm==left],df[df.arm==right],m.METRICS):
            q=bs[(bs.comparison==left+' minus '+right)&(bs.metric==row['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[row['delta'],row['ci_low'],row['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
    table=pd.read_csv(pub/'OUTER_POOLED_CHANNELS.csv').set_index('arm')
    for arm,v in pooled.items():
        for k,value in v.items():assert np.isclose(table.loc[arm,k],value,atol=1e-12,rtol=0,equal_nan=True)
    first=df[df.arm=='FINAL_DECODER'].set_index('patient').sort_index();second=df[df.arm=='M18_TIME_CONTROL'].set_index('patient').sort_index();assert np.array_equal(first[['ez_auroc','ez_auprc','ez_mrr','top1_is_ez']],second[['ez_auroc','ez_auprc','ez_mrr','top1_is_ez']],equal_nan=True)
    write(pub/'OUTER_INDEPENDENT_VERIFICATION.json',dict(status='PASS',patient_metric_cells=320,patients_per_arm=80,channels_per_arm=7635,all15_checkpoint_bank_hashes_and5_maps_unchanged=True,decoder_recomputed_without_labels=True,ranking_exact_M18_unchanged=True,independent_sklearn_metrics=True,independent_bootstrap_multiplicity=True,pooled_confusions_verified=True,aggregate_sha256={p.name:sha(p) for p in pub.glob('*.csv')},no_new_model_inference_or_training=True));print('MACRO70_FINAL_OUTER_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':p=arguments();a=p.parse_args();main(a)
