"""Read-only verification of every completed Phase4 cell, never Outer."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
CODE=Path(__file__).parents[1]/'code';sys.path.insert(0,str(CODE))
from bank import arguments,input_binding,read,sha,write,context
from models import Adapter,ARMS
from explore import original_modules

def main(a):
    binding=input_binding(a);fr=json.loads((a.runtime/'MODEL_FREEZE_PRIVATE.json').read_text());assert fr['binding']==binding and len(fr['selections'])==15
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);m,boot=original_modules();saved=pd.DataFrame(read(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl'));rows=[];audit=[]
    for fold in range(1,6):
        d=read(a.runtime/f'fold{fold}/BANK_PRIVATE.pkl');assert set(d['patient'][d['train']])==set(d['pre']['fit_patient_ids'])==banks[fold-1]['roles']['fit'] and set(d['patient'][d['val']])==banks[fold-1]['roles']['validation'] and set(d['patient']).isdisjoint(banks[fold-1]['roles']['test'])
        groups=[np.flatnonzero(d['patient']==p) for p in sorted(set(d['patient'][d['train']]))];f=d['features_raw'];mean=np.mean([f[ix].astype(float).mean(0) for ix in groups],axis=0);second=np.mean([(f[ix].astype(float)**2).mean(0) for ix in groups],axis=0);sd=np.sqrt(np.maximum(second-mean**2,0)).clip(1e-6)
        assert np.array_equal(d['pre']['mean'],mean) and np.array_equal(d['pre']['scale'],sd) and np.array_equal(d['x'],((f-mean)/sd).astype(np.float32))
        for arm in ARMS:
            ref=next(q for q in fr['selections'] if q['arm']==arm and q['fold']==fold);cp=a.runtime/arm/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==ref['checkpoint_sha256'] and sha(a.runtime/f'fold{fold}/BANK_PRIVATE.pkl')==ref['bank_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt') and ck['threshold']==ref['threshold'] and ck['epoch']==ref['epoch']
            model=Adapter(arm).cuda().eval();model.load_state_dict(ck['model'],strict=True)
            with torch.inference_mode():ll=model(torch.tensor(d['x'][d['val']],device='cuda'),torch.tensor(d['base'][d['val']],device='cuda')).cpu().numpy()
            q=read(a.runtime/f'predictions_private/{arm}_fold{fold}.pkl');assert q['binding']==binding and q['checkpoint_sha256']==sha(cp) and np.array_equal(q['indices'],d['val']);drift=float(np.abs(ll-q['logit']).max());assert drift<=1e-6
            score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
            for pid in sorted(set(d['patient'][d['val']])):
                k=np.flatnonzero(d['patient'][d['val']]==pid);k=k[np.argsort(d['channel'][d['val']][k].astype(str),kind='stable')];ii=d['val'][k];rows.append(dict(arm=arm,fold=fold,patient=str(pid),**m.independent(d['y'][ii],score[k],ref['threshold'])))
            audit.append(dict(arm=arm,fold=fold,selected_replay_max_logit_difference=drift,checkpoint_sha256=sha(cp),FIT_VAL_roles_exact=True,fit_only_scaler_recomputed=True));del model
    actual=pd.DataFrame(rows);keys=['arm','fold','patient'];l=actual.set_index(keys).sort_index();r=saved[saved.arm.isin(ARMS)].set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[m.METRICS],r[m.METRICS],rtol=0,atol=1e-12,equal_nan=True)
    summary=pd.read_csv(a.runtime/'results/DEVELOPMENT_SUMMARY.csv').set_index('arm');means=actual.groupby('arm')[m.METRICS].mean();assert np.allclose(summary.loc[means.index,m.METRICS],means,rtol=0,atol=1e-12,equal_nan=True)
    values=means.reset_index();values['parameters']=values.arm.map({ARMS[0]:289,ARMS[1]:6289,ARMS[2]:6289});best=values.sort_values(['macro_f1','ez_f1','parameters'],ascending=[False,False,True]).iloc[0];assert best.arm==fr['selected']
    bs=pd.read_csv(a.runtime/'results/PAIRED_BOOTSTRAP.csv')
    for arm in ARMS:
        for ref in ['D1','M8_MACRO_AWARE']:
            for row in boot.compare(c,saved[saved.arm==arm],saved[saved.arm==ref],m.METRICS):
                q=bs[(bs.comparison==f'{arm} minus {ref}')&(bs.metric==row['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[row['delta'],row['ci_low'],row['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
    write(a.runtime/'results/INDEPENDENT_VERIFICATION.json',dict(status='PASS',all15_actual_training_cells=True,all15_selected_checkpoint_replays=True,patient_metric_cells=195,all65_appearances47_IDs_per_model=True,own_fold_TEST_excluded=True,scaler_fit_roles_recomputed=True,independent_sklearn_metrics=True,bootstrap_independent_multiplicity=True,selection_confirmed=str(best.arm),folds=audit,source_sha256=sha(Path(__file__)),no_new_Outer=True))
    print('PHASE4_ALL15_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main(arguments(argparse.ArgumentParser()).parse_args())
