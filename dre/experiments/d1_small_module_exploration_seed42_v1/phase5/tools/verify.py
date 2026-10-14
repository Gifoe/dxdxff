"""Post-training saved-score/checkpoint audit, no new inference or Outer."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'code'))
from explore import read,sha,write,context,original_modules

def main():
    rt=Path(r'D:\d1_macro70_phase5_runtime');baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime');fr=json.loads((rt/'MODEL_FREEZE_PRIVATE.json').read_text());bind=json.loads((rt/'INPUT_FREEZE_PRIVATE.json').read_text());assert fr['binding']==bind and bind['code']=={p.name:sha(p) for p in (ROOT/'phase5/code').glob('*.py')};assert len(fr['selections'])==10
    c,core,fac,trainer,records,rec,banks,hist,deps=context(Path(r'E:\DRE-nips\new-pipeline\7-11'),baseline);m,b=original_modules();rows=[];audits=[]
    for r in fr['selections']:
        arm,fold=r['arm'],r['fold'];cp=rt/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==r['checkpoint_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt');assert ck['threshold']==r['threshold'] and ck['epoch']==r['epoch'];baseck=torch.load(baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(ck['model']['base.'+k],v) for k,v in baseck['model'].items())
        bp=baseline/f'D1/fold{fold}/BANK_PRIVATE.pt';assert sha(bp)==r['D1_bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test']);q=read(rt/f'predictions_private/{arm}_fold{fold}.pkl');assert q['binding']==bind and q['checkpoint_sha256']==sha(cp) and q['threshold']==r['threshold']
        score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
        for pid in sorted(set(d['patient'][d['val']])):
            k=np.flatnonzero(d['patient'][d['val']]==pid);k=k[np.argsort(d['channel'][d['val']][k].astype(str),kind='stable')];ii=d['val'][k];rows.append(dict(arm=arm,fold=fold,patient=str(pid),**m.independent(d['y'][ii],score[k],r['threshold'])))
        audits.append(dict(arm=arm,fold=fold,D1_parameters_exact_unchanged=True,checkpoint_sha256=sha(cp),actual_training_epochs=len(pd.read_csv(rt/'results/TRAINING_HISTORY.csv').query('arm==@arm and fold==@fold and epoch>0'))))
    actual=pd.DataFrame(rows);saved=pd.DataFrame(read(rt/'DEVELOPMENT_METRICS_PRIVATE.pkl'));keys=['arm','fold','patient'];l=actual.set_index(keys).sort_index();r=saved[saved.arm!='D1'].set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[m.METRICS],r[m.METRICS],rtol=0,atol=1e-12,equal_nan=True)
    means=actual.groupby('arm')[m.METRICS].mean();summary=pd.read_csv(rt/'results/DEVELOPMENT_SUMMARY.csv').set_index('arm');assert np.allclose(summary.loc[means.index,m.METRICS],means,rtol=0,atol=1e-12,equal_nan=True);chosen=means.sort_values(['macro_f1','ez_f1'],ascending=False).index[0];assert chosen==fr['selected']
    bs=pd.read_csv(rt/'results/PAIRED_BOOTSTRAP.csv')
    for arm in means.index:
        assert len(actual[actual.arm==arm])==65 and actual[actual.arm==arm].patient.nunique()==47
        for row in b.compare(c,saved[saved.arm==arm],saved[saved.arm=='D1'],m.METRICS):
            q=bs[(bs.comparison==arm+' minus D1')&(bs.metric==row['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[row['delta'],row['ci_low'],row['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
    write(rt/'results/INDEPENDENT_VERIFICATION.json',dict(status='PASS',actual10_training_cells=True,all_checkpoint_hashes_verified=True,all_D1_weights_unchanged=True,independent_sklearn_cells=130,bootstrap_independent_multiplicity=True,no_TEST_access=True,selected=chosen,folds=audits));print('PHASE5_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
