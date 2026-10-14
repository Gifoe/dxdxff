"""Development-only threshold ceiling; oracle is diagnostic, never performance."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'code'))
from explore import read,sha,original_modules

def run():
    runtime=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime');baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
    fr=json.loads((runtime/'MODEL_FREEZE_PRIVATE.json').read_text());out=[];curves=[]
    for fold in range(1,6):
        r=next(v for v in fr['selections'] if v['arm']=='M8_MACRO_AWARE' and v['fold']==fold)
        assert sha(runtime/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt')==r['checkpoint_sha256']
        d=torch.load(baseline/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        q=read(runtime/f'predictions_private/M8_MACRO_AWARE_fold{fold}.pkl')['modes']['full'];s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();y=d['y'][d['val']];pid=d['patient'][d['val']]
        for p in sorted(set(pid)):
            k=pid==p;z=y[k]==0;score=s[k];taus=np.unique(np.r_[0,.5,1,score,np.nextafter(score,np.float32(np.inf))]);pred=score[None,:]<taus[:,None];tp=(pred&z).sum(1);fp=(pred&~z).sum(1);fn=(~pred&z).sum(1);tn=(~pred&~z).sum(1);macro=.5*(np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp)),where=2*tp+fp+fn>0)+np.divide(2*tn,2*tn+fp+fn,out=np.zeros(len(tp)),where=2*tn+fp+fn>0))
            order=np.argsort(score,kind='stable');nz=np.cumsum(z[order]);nt=np.arange(1,len(z)+1);fcount=.5*(2*nz/(nt+z.sum())+2*(len(z)-nt-z.sum()+nz)/(2*len(z)-nt-z.sum()))
            from sklearn.metrics import f1_score
            actual=f1_score(z,score<r['threshold'],labels=[False,True],average='macro',zero_division=0)
            rank_top20=max(1,round(.2*len(z)));pr=np.zeros(len(z),bool);pr[order[:rank_top20]]=True
            out.append(dict(fold=fold,actual=actual,threshold_oracle=float(macro.max()),rank_oracle=float(fcount.max()),fixed_top20=f1_score(z,pr,labels=[False,True],average='macro',zero_division=0),oracle_threshold=float(taus[macro.argmax()]),actual_threshold=r['threshold'],EZ_fraction=float(z.mean())))
            grid=np.arange(0,1.0001,.005);pp=score[None,:]<grid[:,None];tp=(pp&z).sum(1);fp=(pp&~z).sum(1);fn=(~pp&z).sum(1);tn=(~pp&~z).sum(1);mf=.5*(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1));curves.append(mf)
    df=pd.DataFrame(out);curve=np.mean(curves,axis=0)
    print(json.dumps(dict(scope='repeated_Development_only_no_Outer',appearances=len(df),mean=df.mean().to_dict(),global_threshold_diagnostic=float(np.argmax(curve)*.005),global_threshold_oracle_macro=float(curve.max()),oracle_is_not_benchmark_performance=True,folds=df.groupby('fold')[['actual','threshold_oracle','fixed_top20']].mean().to_dict()),indent=2))

if __name__=='__main__':run()
