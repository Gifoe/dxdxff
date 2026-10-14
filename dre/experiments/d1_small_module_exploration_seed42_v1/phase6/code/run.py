"""Frozen-evidence conditional tree residuals; no Outer loader."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import sklearn
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
import torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'code'))

class FrozenM8Init(ClassifierMixin,BaseEstimator):
    def fit(self,X,y,sample_weight=None):self.classes_=np.array([0,1]);return self
    def predict_proba(self,X):
        p=expit(np.asarray(X[:,0],dtype=float));return np.c_[1-p,p]

def weights(patient,y):
    pos=max(int((1-y).sum()),1)/max(int(y.sum()),1);w=np.empty(len(y))
    for p in set(patient):w[patient==p]=1/(patient==p).sum()
    return w*np.where(y==1,pos,1)

def prune_fit(x,tr,names):
    keep=[0];reason=[]
    for j in range(1,x.shape[1]):
        if np.ptp(x[tr,j])==0:reason.append(dict(feature=names[j],reason='FIT_exact_constant'));continue
        duplicate=next((k for k in keep if np.array_equal(x[tr,j],x[tr,k])),None)
        if duplicate is not None:reason.append(dict(feature=names[j],reason='FIT_exact_duplicate_of_'+names[duplicate]));continue
        keep.append(j)
    return np.asarray(keep),reason

def main(a):
    from explore import context,original_modules,read,save,sha,write
    a.runtime.mkdir(parents=True,exist_ok=True);pub=a.runtime/'results';pub.mkdir(exist_ok=True);phase=ROOT/'phase6'
    lock=json.loads((phase/'PROTOCOL_LOCK.json').read_text());assert json.loads((a.phase5/'results/RUN_STATUS.json').read_text())['status']=='ALL10_JOINT_DEVELOPMENT_CELLS_COMPLETE';assert json.loads((a.phase4/'results/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
    binding=dict(protocol=sha(phase/'PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},phase4freeze=sha(a.phase4/'MODEL_FREEZE_PRIVATE.json'),phase4input=sha(a.phase4/'INPUT_FREEZE_PRIVATE.json'),phase5status=sha(a.phase5/'results/RUN_STATUS.json'),sklearn=sklearn.__version__)
    fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert json.loads(fp.read_text())==binding
    else:write(fp,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);m,b=original_modules();rows=read(a.phase1/'D1_METRICS_PRIVATE.pkl');sels=[];allhist=[];audits=[]
    for arm in lock['models']:
        for fold in range(1,6):
            d=read(a.phase4/f'fold{fold}/BANK_PRIVATE.pkl');source=banks[fold-1];assert np.array_equal(d['patient'],source['patient']) and np.array_equal(d['channel'],source['channel']) and np.array_equal(d['y'],source['y']) and np.array_equal(d['train'],source['train']) and np.array_equal(d['val'],source['val']);assert set(d['patient']).isdisjoint(source['roles']['test']) and set(source['pre']['fit_patient_ids'])==source['roles']['fit']
            evidence=d['features_raw'];cols=np.r_[np.arange(17),np.arange(113,129)];names=['M8logit','D1logit','rank_percentile']+[f'R14_{j}' for j in range(14)]+[f'context_{j}' for j in range(16)]+[str(v) for v in source['features']]
            x=np.c_[evidence[:,cols],source['x']];assert x.shape[1]==len(names)==121 and np.isfinite(x).all();tr,va=d['train'],d['val'];keep,removed=prune_fit(x,tr,names);x=x[:,keep];w=weights(d['patient'][tr],d['y'][tr]);bank_binding=dict(input=binding,phase4_bank=sha(a.phase4/f'fold{fold}/BANK_PRIVATE.pkl'),original_bank=deps[f'legal_bank{fold}'],kept=keep.tolist())
            cell=a.runtime/arm/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True);target=cell/'BEST_PRIVATE.pkl';done=cell/'COMPLETE_PRIVATE.json'
            if done.exists():
                q=read(target);cc=json.loads(done.read_text());assert q['binding']==bank_binding and cc['checkpoint_sha256']==sha(target);s=q['score'];best=q['selection'];history=q['history'];selected=q['model']
            else:
                config=lock['configs'][arm];model=GradientBoostingClassifier(loss='log_loss',learning_rate=.03,n_estimators=200,subsample=1,max_features=None,init=FrozenM8Init(),random_state=42+1009*fold,**config)
                initial=expit(d['base'][va].astype(float));sel=core.select_threshold(d['y'][va],initial,d['patient'][va]);best=dict(trees=0,**sel);key=(sel['macro_f1'],sel['ez_f1'],0);selected=None;s=initial;history=[dict(arm=arm,fold=fold,**best)]
                model.fit(x[tr],d['y'][tr],sample_weight=w)
                for step,proba in enumerate(model.staged_predict_proba(x[va]),1):
                    if step not in lock['training']['validation_checkpoints']:continue
                    cur=proba[:,1];sel=core.select_threshold(d['y'][va],cur,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-step);history.append(dict(arm=arm,fold=fold,trees=step,**sel));print('CONDITIONAL_TREE_VALIDATION',arm,fold,step,flush=True)
                    if candidate>key:
                        key=candidate;best=dict(trees=step,**sel);selected=copy.deepcopy(model);selected.estimators_=selected.estimators_[:step];selected.n_estimators=step;selected.n_estimators_=step;s=cur.copy()
                if selected is not None:assert np.allclose(selected.predict_proba(x[va])[:,1],s,atol=1e-12,rtol=0)
                save(target,dict(binding=bank_binding,model=selected,selection=best,score=s,history=history,keep=keep,names=[names[k] for k in keep]));write(done,dict(checkpoint_sha256=sha(target),trees=best['trees'],threshold=best['threshold']))
            rows.extend(m.patient_rows(c,d,s,best['threshold'],arm,fold));save(a.runtime/f'predictions_private/{arm}_fold{fold}.pkl',dict(binding=bank_binding,score=s,indices=va,threshold=best['threshold'],checkpoint_sha256=sha(target)));sels.append(dict(arm=arm,fold=fold,trees=best['trees'],threshold=best['threshold'],checkpoint_sha256=sha(target),input_features=121,kept_features=len(keep)));allhist.extend(history);audits.extend(dict(arm=arm,fold=fold,**q) for q in removed);pd.DataFrame(sels).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(allhist).to_csv(pub/'VALIDATION_GRID.csv',index=False);save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows);write(pub/'RUN_STATUS.json',dict(status='CONDITIONAL_TREE_TRAINING',arm=arm,fold=fold,no_TEST_access=True));print('TREE_CELL_COMPLETE',arm,fold,flush=True)
    df=pd.DataFrame(rows);means=df.groupby('arm')[m.METRICS].mean();means.reset_index().to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:df.groupby(['arm',field])[m.METRICS].mean().reset_index().to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    pd.DataFrame(audits).to_csv(pub/'FIT_ONLY_FEATURE_PRUNING.csv',index=False);boots=[]
    for arm in lock['models']:boots.extend(dict(comparison=arm+' minus D1',**r) for r in b.compare(c,df[df.arm==arm],df[df.arm=='D1'],m.METRICS))
    pd.DataFrame(boots).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);chosen=means.loc[lock['models']].sort_values(['macro_f1','ez_f1'],ascending=False).index[0];write(a.runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=sels,selected=chosen));write(pub/'RUN_STATUS.json',dict(status='ALL10_CONDITIONAL_TREE_CELLS_COMPLETE',selected=chosen,MacroF1=float(means.loc[chosen,'macro_f1']),target_over_070=bool(means.loc[chosen,'macro_f1']>.7),no_TEST_access=True));print('CONDITIONAL_TREE_PHASE6_COMPLETE',chosen,float(means.loc[chosen,'macro_f1']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_macro70_phase6_runtime'));p.add_argument('--phase4',type=Path,default=Path(r'D:\d1_macro70_phase4_runtime'));p.add_argument('--phase5',type=Path,default=Path(r'D:\d1_macro70_phase5_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
