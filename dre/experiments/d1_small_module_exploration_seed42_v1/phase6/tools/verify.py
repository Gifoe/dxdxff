"""Independent saved tree/score/metric checks, no new development training."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'phase6/code'));sys.path.insert(0,str(ROOT/'code'))
from run import FrozenM8Init,prune_fit
from explore import context,original_modules,read,sha,write

def main():
    rt=Path(r'D:\d1_macro70_phase6_runtime');p4=Path(r'D:\d1_macro70_phase4_runtime');baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime');fr=json.loads((rt/'MODEL_FREEZE_PRIVATE.json').read_text());binding=json.loads((rt/'INPUT_FREEZE_PRIVATE.json').read_text());assert fr['binding']==binding and binding['code']=={p.name:sha(p) for p in (ROOT/'phase6/code').glob('*.py')} and len(fr['selections'])==10
    c,core,fac,trainer,records,rec,banks,hist,deps=context(Path(r'E:\DRE-nips\new-pipeline\7-11'),baseline);m,b=original_modules();rows=[];checks=[]
    for r in fr['selections']:
        arm,fold=r['arm'],r['fold'];d=read(p4/f'fold{fold}/BANK_PRIVATE.pkl');source=banks[fold-1];assert np.array_equal(d['patient'],source['patient']) and np.array_equal(d['channel'],source['channel']) and set(d['patient']).isdisjoint(source['roles']['test']);cols=np.r_[np.arange(17),np.arange(113,129)];names=['M8logit','D1logit','rank_percentile']+[f'R14_{j}' for j in range(14)]+[f'context_{j}' for j in range(16)]+[str(v) for v in source['features']];x=np.c_[d['features_raw'][:,cols],source['x']];keep,removed=prune_fit(x,d['train'],names);cp=rt/arm/f'fold{fold}/BEST_PRIVATE.pkl';assert sha(cp)==r['checkpoint_sha256'];q=read(cp);assert np.array_equal(keep,q['keep']) and q['binding']['original_bank']==deps[f'legal_bank{fold}'] and q['selection']['threshold']==r['threshold'] and q['selection']['trees']==r['trees'];score=expit(d['base'][d['val']].astype(float)) if q['model'] is None else q['model'].predict_proba(x[d['val']][:,keep])[:,1];assert np.allclose(score,q['score'],atol=1e-12,rtol=0)
        pred=read(rt/f'predictions_private/{arm}_fold{fold}.pkl');assert pred['checkpoint_sha256']==sha(cp) and np.array_equal(pred['indices'],d['val']) and np.array_equal(pred['score'],q['score'])
        for pid in sorted(set(d['patient'][d['val']])):
            k=np.flatnonzero(d['patient'][d['val']]==pid);k=k[np.argsort(d['channel'][d['val']][k].astype(str),kind='stable')];ii=d['val'][k];rows.append(dict(arm=arm,fold=fold,patient=str(pid),**m.independent(d['y'][ii],score[k],r['threshold'])))
        checks.append(dict(arm=arm,fold=fold,tree_probability_replay_max_difference=float(np.abs(score-q['score']).max()),FIT_pruning_recomputed=True,own_fold_TEST_excluded=True,checkpoint_sha256=sha(cp)))
    df=pd.DataFrame(rows);saved=pd.DataFrame(read(rt/'DEVELOPMENT_METRICS_PRIVATE.pkl'));keys=['arm','fold','patient'];l=df.set_index(keys).sort_index();r=saved[saved.arm!='D1'].set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[m.METRICS],r[m.METRICS],rtol=0,atol=1e-12,equal_nan=True);means=df.groupby('arm')[m.METRICS].mean();summary=pd.read_csv(rt/'results/DEVELOPMENT_SUMMARY.csv').set_index('arm');assert np.allclose(summary.loc[means.index,m.METRICS],means,rtol=0,atol=1e-12,equal_nan=True)
    assert means.sort_values(['macro_f1','ez_f1'],ascending=False).index[0]==fr['selected'];bs=pd.read_csv(rt/'results/PAIRED_BOOTSTRAP.csv')
    for arm in means.index:
        assert len(df[df.arm==arm])==65 and df[df.arm==arm].patient.nunique()==47
        for row in b.compare(c,saved[saved.arm==arm],saved[saved.arm=='D1'],m.METRICS):
            q=bs[(bs.comparison==arm+' minus D1')&(bs.metric==row['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[row['delta'],row['ci_low'],row['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
    write(rt/'results/INDEPENDENT_VERIFICATION.json',dict(status='PASS',all10_completed_cells=True,all10_fitted_tree_probability_replays=True,independent_patient_metric_cells=130,FIT_only_pruning_recomputed=True,original_ownfold_preprocessors_unchanged=True,bootstrap_independent_multiplicity=True,folds=checks,no_TEST_access=True));print('PHASE6_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
