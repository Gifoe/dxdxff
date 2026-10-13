"""One frozen development evaluation; no outer inference implementation."""
import json
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint,patient_rows
from bootstrap import compare
from mechanism import frozen_audit


def gate_values(summary,folds,integrity=True):
 t=summary.set_index('arm');wins=int((folds.B3>folds.B0).sum())
 return dict(AUROC_B3_minus_B0_ge0_010=bool(t.loc['B3','ez_auroc']-t.loc['B0','ez_auroc']>=.010),
  AUROC_B3_minus_B1_ge0_005=bool(t.loc['B3','ez_auroc']-t.loc['B1','ez_auroc']>=.005),
  AUROC_B3_minus_B2_ge0_005=bool(t.loc['B3','ez_auroc']-t.loc['B2','ez_auroc']>=.005),
  AUPRC_not_decreased=bool(t.loc['B3','ez_auprc']>=t.loc['B0','ez_auprc']),
  MacroF1_not_decreased=bool(t.loc['B3','macro_f1']>=t.loc['B0','macro_f1']),
  EZF1_decline_at_most0_005=bool(t.loc['B3','ez_f1']-t.loc['B0','ez_f1']>=-.005),
  specificity_decline_at_most0_010=bool(t.loc['B3','specificity']-t.loc['B0','specificity']>=-.010),
  positive_AUROC_folds_ge4=wins>=4,no_fold_preprocessing_or_feature_mismatch=bool(integrity),all_identity_parameter_numerical_leakage_checks=bool(integrity)),wins


def interaction_tables(df,metrics):
 tables={arm:df[df.arm==arm].set_index(['fold','patient']).sort_index() for arm in ['B0','B1','B2','B3']}
 assert all(tables['B0'].index.equals(t.index) for t in tables.values())
 left=tables['B3'].copy();right=tables['B0'].copy()
 left[metrics]=tables['B3'][metrics]-tables['B2'][metrics]-tables['B1'][metrics]+tables['B0'][metrics]
 right[metrics]=0.
 return left.reset_index(),right.reset_index()


def main():
 a=args();lock,ref,c,core,original,trainer,models,fm,banks,binding=setup(a);pub=a.runtime/'public'
 assert not (pub/'RUN_STATUS.json').exists()
 pre=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());eng=json.loads((pub/'ENGINEERING_TESTS.json').read_text())
 assert pre['binding']==eng['binding']==binding and pre['status']==eng['status']=='PASS'
 frozen=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert frozen['binding']==binding
 fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json');rows=[];files={};metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN']
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
  d=torch.load(bp,map_location='cpu',weights_only=False);va=d['val'];tx=torch.tensor(d['x'][va],device='cuda')
  for arm in lock['arms']:
   rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold);cp=checkpoint(a,arm,fold)
   assert c.sha(cp)==rec['checkpoint_sha256'] and rec['bank_sha256']==c.sha(bp)
   ck=torch.load(cp,map_location='cpu',weights_only=False);tau=ck['threshold'];path=bp.parent/f'{arm}_VAL_PRIVATE.pt'
   if arm=='B0':cell=torch.load(path,map_location='cpu',weights_only=False);score=cell['scores']
   else:
    m=models.constructor(original.SourceA0,arm,d['mapping'])().cuda().eval();m.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode():ll=m(tx);score=torch.sigmoid(ll).cpu().numpy()
    assert not path.exists();core.torch_write(path,dict(binding=fs,indices=va,logits=ll.cpu().numpy(),scores=score,threshold=tau,checkpoint_sha256=c.sha(cp)))
   cells=patient_rows(c,d,va,score,tau,arm,fold);assert len(cells)==13
   assert abs(np.mean([r['macro_f1'] for r in cells])-ck['macro_f1'])<1e-12 and abs(np.mean([r['ez_f1'] for r in cells])-ck['ez_f1'])<1e-12
   for r in cells:r['channel_stratum']='<64' if r['channels']<64 else '64to127' if r['channels']<128 else '>=128'
   rows.extend(cells);files[f'{arm}_{fold}']=c.sha(path)
 assert len(rows)==260 and len(set(r['patient'] for r in rows))==47
 core.torch_write(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pt',rows)
 c.write_json(a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json',dict(binding=fs,files=files,new_outer_accessed=False))
 df=pd.DataFrame(rows);summary=df.groupby('arm')[metrics].mean().reset_index();summary['parameters']=8817
 summary.to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
 df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
 for key,name in [('fold','FOLD'),('center','CENTER'),('channel_stratum','CHANNEL_STRATA')]:
  out=[dict(arm=arm,**{key:v},appearances=len(part),unique_patients=part.patient.nunique(),**part[metrics].mean().to_dict()) for (arm,v),part in df.groupby(['arm',key])]
  pd.DataFrame(out).to_csv(pub/('DEVELOPMENT_BY_'+name+'.csv'),index=False)
 effects=[]
 for left,right in lock['comparisons']:
  effects.extend(dict(comparison=left+' minus '+right,**r) for r in compare(c,df[df.arm==left],df[df.arm==right],metrics))
 pd.DataFrame(effects).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
 ll,rr=interaction_tables(df,metrics);inter=[dict(comparison='(B3-B2)-(B1-B0)',**r) for r in compare(c,ll,rr,metrics)]
 pd.DataFrame(effects+inter).to_csv(pub/'FACTORIAL_COMPARISONS.csv',index=False)
 folds=df.groupby(['fold','arm']).ez_auroc.mean().unstack();criteria,wins=gate_values(summary,folds)
 auc=next(r for r in effects if r['comparison']=='B3 minus B0' and r['metric']=='ez_auroc')
 c.write_json(pub/'VALIDATION_GATE.json',dict(pass_gate=all(criteria.values()),criteria=criteria,positive_AUROC_folds=wins,
  AUROC_B3_minus_B0_CI_excludes_zero=auc['ci_low']>0 or auc['ci_high']<0,new_outer_authorized=False))
 # Descriptive diagnostics follow immutable model/threshold and metric freeze.
 frozen_audit(a,c,core,original,models,binding,fs)
 assert setup(a)[-1]==binding
 c.write_json(a.runtime/'MECHANISM_SEAL.json',dict(binding=fs,sha256=c.sha(a.runtime/'NORMALIZATION_MECHANISM_PRIVATE.pt')))
 terminal='PURN_DEVELOPMENT_PROMISING_NOT_INDEPENDENTLY_CONFIRMED' if all(criteria.values()) else 'PURN_DEVELOPMENT_GATE_FAILED'
 c.write_json(pub/'SCIENTIFIC_TERMINAL.json',dict(terminal=terminal,development_only=True,new_outer_accessed=False))
 c.write_json(pub/'RUN_STATUS.json',dict(status='ALL_DEVELOPMENT_AND_MECHANISM_COMPLETE',new_training_cells=15,reused_baseline_cells=5,
  validation_appearances=65,unique_validation_patients=47,registered_protocol_unchanged=True,new_outer_accessed=False,development_freeze_sha256=fs))
 print('PURN_DEVELOPMENT_COMPLETE',terminal,flush=True)


if __name__=='__main__':main()
