"""Independent sklearn metrics and multiplicity bootstrap; no model inference/outer."""
import json
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint,baseline
from conditional_ridge import fit_cross,apply,mapping,wrong_permutation
from cove_model import constructor
from metrics import independent
from bootstrap import independent as independent_bootstrap
from evaluate import gate

def recompute(b,cell,arm,fold):
 va=cell['indices'];out=[]
 for pid in sorted(set(b['patient'][va])):
  k=np.flatnonzero(b['patient'][va]==pid);k=k[np.argsort(b['channel'][va][k].astype(str),kind='stable')];ii=va[k];n=len(k)
  out.append(dict(arm=arm,fold=fold,patient=pid,center=str(b['center'][ii[0]]),channel_stratum='<=64' if n<=64 else ('65-128' if n<=128 else '>128'),**independent(b['y'][ii],cell['scores'][k],cell['threshold'])))
 return out

def main():
 a=args();o=setup(a);c,core=o.c,o.core;pub=o.pub;status=json.loads((pub/'RUN_STATUS.json').read_text());assert status['binding']==o.bind
 assert status['status']=='DEVELOPMENT_COMPLETE_PENDING_INDEPENDENT_VERIFICATION'
 fr=json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text());dfreeze=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());seal=json.loads((a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json').read_text())
 assert fr['binding']==dfreeze['binding']==o.bind and c.sha(a.runtime/'NUISANCE_FREEZE.json')==fr['nuisance_sha256']==dfreeze['nuisance_sha256']
 assert c.sha(a.runtime/'DEVELOPMENT_FREEZE.json')==status['freeze_sha256']==seal['binding'] and c.sha(a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json')==status['prediction_seal_sha256']
 mp=mapping(o.manifest['columns']);rs=[];ms=[];checked=0;coeff=[];epochs=0
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==fr['files'][f'bank{fold}'];b=torch.load(bp,map_location='cpu',weights_only=False);d=o.banks[fold-1];mask=np.array([q['view']!='zdelta' for q in o.manifest['columns']])
  assert np.array_equal(b['x'],o.helper.masked(d['x'],mask)) and b['mapping']==mp and np.array_equal(b['permutation'],wrong_permutation(mp,42+1009*fold))
  for k in ['y','g','train','val','patient','channel','features']:assert np.array_equal(b[k],d[k])
  for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(b['pre'][k],d['pre'][k])
  tr=b['train'];va=b['val'];stats,raw=fit_cross(b['x'][tr],b['patient'][tr],mp,42+1009*fold)
  assert np.array_equal(raw/stats['scale'],b['e'][tr]) and np.array_equal(apply(b['x'][va],stats,mp),b['e'][va]) and np.array_equal(stats['scale'],b['stats']['scale'])
  assert set(stats['fit_ids'])==set(b['patient'][tr]) and set(stats['fit_ids']).isdisjoint(set(b['patient'][va]))
  for h in range(4):
   assert set(stats['inner'][h]['fit_ids']).isdisjoint(set(stats['inner'][h]['heldout_ids']))
   for j in range(14):
    for k in ['Dcoef','Dintercept','Rcoef','Rintercept']:assert np.array_equal(np.asarray(stats['inner'][h]['fits'][j][k]),np.asarray(b['stats']['inner'][h]['fits'][j][k]))
  for j in range(14):
   for k in ['Dcoef','Dintercept','Rcoef','Rintercept']:assert np.array_equal(np.asarray(stats['full'][j][k]),np.asarray(b['stats']['full'][j][k]))
  oldbank,oldcp,oldpred,rr,hh=baseline(a,fold);assert c.sha(oldbank)==rr['bank_sha256'] and c.sha(oldcp)==rr['checkpoint_sha256'] and c.sha(oldpred)==hh[f'F0_validation{fold}']
  p=a.runtime/f'fold{fold}/B0_VAL_PRIVATE.pt';assert c.sha(p)==fr['files'][f'baseline_validation{fold}'];oldcell=torch.load(oldpred,map_location='cpu',weights_only=False);bc=torch.load(p,map_location='cpu',weights_only=False);assert np.array_equal(bc['scores'],oldcell['scores']) and bc['threshold']==rr['threshold']
  rs.extend(recompute(b,bc,'B0',fold))
  for arm in ['B1','B2','B3']:
   rec=next(r for r in dfreeze['selections'] if r['arm']==arm and r['fold']==fold);cp=checkpoint(a,arm,fold);assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256']
   ck=torch.load(cp,map_location='cpu',weights_only=False);last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());epochs+=last['epoch']
   assert c.sha(cp)==done['best_sha256'] and c.sha(cp.parent/'LAST_PRIVATE.pt')==done['last_sha256'] and ck['epoch']==rec['epoch'] and ck['threshold']==rec['threshold']
   core.seed_all(42+1009*fold);model=constructor(o.original.SourceA0,arm,mp,stats,b['permutation'],b)(42+1009*fold);assert core.state_hash(model.state_dict())==ck['binding']['initial_hash'] and sum(p.numel() for p in model.parameters())==8845
   assert ck['binding']['max_epochs']==30 and ck['binding']['penalty']==0 and ck['binding']['device']=='cuda'
   pp=a.runtime/f'fold{fold}/{arm}_VAL_PRIVATE.pt';assert c.sha(pp)==seal['files'][f'{arm}_{fold}'];cell=torch.load(pp,map_location='cpu',weights_only=False)
   assert cell['binding']==status['freeze_sha256'] and cell['checkpoint_sha256']==c.sha(cp) and cell['threshold']==rec['threshold'] and np.array_equal(cell['indices'],va)
   part=recompute(b,cell,arm,fold);rs.extend(part);assert abs(np.mean([r['macro_f1'] for r in part])-ck['macro_f1'])<1e-12 and abs(np.mean([r['ez_f1'] for r in part])-ck['ez_f1'])<1e-12
   if arm=='B3':
    pp=a.runtime/f'fold{fold}/MECHANISM_PRIVATE.pt';assert c.sha(pp)==seal['files'][f'mechanism{fold}'];mech=torch.load(pp,map_location='cpu',weights_only=False);assert mech['binding']==status['freeze_sha256'] and mech['checkpoint_sha256']==c.sha(cp)
    for tag,scores in mech['scores'].items():ms.extend(recompute(b,dict(indices=va,scores=scores,threshold=rec['threshold']),tag,fold))
    for j in range(14):coeff.append(dict(fold=fold,group=j,alpha=float(ck['model']['alpha'][j]),beta=float(ck['model']['beta'][j])))
  print('COVE_VERIFIED_FOLD',fold,flush=True)
 df=pd.DataFrame(rs);mi=pd.DataFrame(ms);metrics=o.lock['metrics'];checked=len(df)+len(mi)
 for dat,p in [(df,a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pt'),(mi,a.runtime/'INTERVENTION_METRICS_PRIVATE.pt')]:
  saved=pd.DataFrame(torch.load(p,weights_only=False)).set_index(['arm','fold','patient']).sort_index();actual=dat.set_index(['arm','fold','patient']).sort_index();assert saved.index.equals(actual.index) and np.allclose(saved[metrics],actual[metrics],atol=1e-12,rtol=0,equal_nan=True)
 for filename,v in [('DEVELOPMENT_SUMMARY.csv',df.groupby('arm')[metrics].mean()),('DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean())]:
  s=pd.read_csv(pub/filename).set_index('arm');assert np.allclose(s[metrics],v,atol=1e-12,rtol=0,equal_nan=True)
 for key in ['fold','center','channel_stratum']:
  s=pd.read_csv(pub/f'DEVELOPMENT_BY_{key.upper()}.csv').set_index(['arm',key]).sort_index();v=df.groupby(['arm',key])[metrics].mean();assert np.allclose(s[metrics],v,atol=1e-12,rtol=0,equal_nan=True)
 for file,contrasts,right_df in [('PAIRED_BOOTSTRAP.csv',o.lock['bootstrap']['comparisons'],df),('INTERVENTION_BOOTSTRAP.csv',[['B3',v] for v in o.lock['interventions']],mi)]:
  boot=pd.read_csv(pub/file)
  for l,r in contrasts:
   values=independent_bootstrap(df[df.arm==l],right_df[right_df.arm==r],o.lock['bootstrap']['metrics'])
   for m,v in values.items():q=boot[(boot.comparison==l+' minus '+r)&(boot.metric==m)].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(dtype=float),v,atol=1e-12,rtol=0,equal_nan=True)
 interventions=pd.read_csv(pub/'COVE_INTERVENTIONS.csv').set_index('arm');v=mi.groupby('arm')[metrics].mean();assert np.allclose(interventions[metrics],v,atol=1e-12,rtol=0,equal_nan=True)
 for m in metrics:assert np.allclose(interventions['full_minus_'+m],df[df.arm=='B3'][m].mean()-v[m],atol=1e-12,rtol=0,equal_nan=True)
 coefficients=pd.read_csv(pub/'COVE_COEFFICIENTS.csv')
 for r in coeff:
  q=coefficients[(coefficients.fold==r['fold'])&(coefficients.group==r['group'])];assert len(q)==2 and np.allclose(q.alpha,r['alpha'],atol=1e-12,rtol=0) and np.allclose(q.beta,r['beta'],atol=1e-12,rtol=0)
 cg=gate(o,df,verified=True);c.write_json(pub/'VALIDATION_GATE.json',cg)
 terminal='COVE_DEVELOPMENT_PROMISING_NOT_INDEPENDENTLY_CONFIRMED' if cg['status']=='PASS' else 'COVE_DEVELOPMENT_GATE_FAILED'
 c.write_json(pub/'SCIENTIFIC_TERMINAL.json',dict(status=terminal,no_new_outer=True,independent_confirmation=False,seed42_complete=True,no_automatic_v2=True))
 c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',binding=o.bind,model_cells=15,completed_epochs=epochs,patient_metric_cells=checked,nuisance_crossfit_and_scales_rebuilt=True,frozen_baseline_predictions_exactly_reused=True,
  metrics_independent_sklearn=True,bootstrap_independent_multiplicity=True,draws=10000,seed=42,no_model_inference=True,no_outer_access=True,aggregate_hashes={p.name:c.sha(p) for p in pub.glob('*.csv')}))
 c.write_json(pub/'RUN_STATUS.json',dict(status='ALL_DEVELOPMENT_AND_MECHANISM_AUDITS_COMPLETE',scientific_terminal=terminal,new_model_cells=15,validation_appearances=65,unique_patients=47,no_outer_access=True,binding=o.bind))
 print(terminal,flush=True)

if __name__=='__main__':main()
