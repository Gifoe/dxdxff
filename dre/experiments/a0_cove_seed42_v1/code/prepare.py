"""Baseline replay, mappings, frozen nuisance crossfit, tests, and nested FIT probes."""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from runtime import args,setup,baseline,patient_rows
from conditional_ridge import mapping,fit_cross,apply,residuals,wrong_permutation,coordinates,weights,mean,sd,diagnostics,weighted_quantiles
from dependencies import load

def probe(d,stats,mp,perm,fold):
 tr=d['train'];xp=d['x'][tr];pp=d['patient'][tr];yy=d['y'][tr];base=np.array([q[v] for q in mp['triples'] for v in ['abs','delta','ratio']]);out=[];private=[]
 for h in range(4):
  fit=stats['assignment']!=h;held=~fit;assert set(pp[fit]).isdisjoint(set(pp[held]))
  ns,raw=fit_cross(xp[fit],pp[fit],mp,42+1009*fold+10000+h);train_e=raw/ns['scale'];held_e=apply(xp[held],ns,mp)
  neq=max(int((yy[fit]==0).sum()),1)/max(int((yy[fit]==1).sum()),1);w=weights(pp[fit])*np.where(yy[fit]==1,neq,1.)
  for arm in ['P0','P1','P2']:
   xf=xp[fit][:,base].astype(float);xv=xp[held][:,base].astype(float)
   if arm!='P0':
    per=perm if arm=='P2' else None;xf=np.column_stack([xf,coordinates(train_e,per)]);xv=np.column_stack([xv,coordinates(held_e,per)])
   m=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=1000,tol=1e-8).fit(xf,yy[fit],sample_weight=w)
   assert m.n_iter_.max()<1000;score=m.predict_proba(xv)[:,list(m.classes_).index(1)];pz=1-score;patients=pp[held];labs=yy[held];vals=[]
   for p in sorted(set(patients)):
    k=patients==p;z=labs[k]==0
    vals.append(dict(patient=p,ez_auroc=roc_auc_score(z,pz[k]) if z.any() and (~z).any() else np.nan,ez_auprc=average_precision_score(z,pz[k]) if z.any() else 0.,log_loss=log_loss(labs[k],score[k],labels=[0,1])))
   q=pd.DataFrame(vals);out.append(dict(fold=fold,inner_group=h,probe=arm,patients=len(q),channels=int(held.sum()),coordinates=xf.shape[1],nuisance_heldout_exclusion=True,**q[['ez_auroc','ez_auprc','log_loss']].mean().to_dict()))
   private.extend(dict(fold=fold,inner_group=h,probe=arm,**r) for r in vals)
 return out,private

def main():
 a=args();o=setup(a);c,core=o.c,o.core;assert not (a.runtime/'EXPERIMENT_FREEZE.json').exists();mp=mapping(o.manifest['columns']);pub=o.pub
 mpdf=pd.DataFrame(mp['triples']);mpdf.to_csv(pub/'FEATURE_TRIPLE_MAPPING.csv',index=False)
 baseline_rows=[];recs=[];banks=[];ridge=[];stability=[];interaction=[];files={};allfinite=True
 mask=np.array([q['view']!='zdelta' for q in o.manifest['columns']]);assert mask.sum()==62
 for fold,d in enumerate(o.banks,1):
  bp,cp,vp,r,h=baseline(a,fold);assert c.sha(bp)==r['bank_sha256'] and c.sha(cp)==r['checkpoint_sha256'] and c.sha(vp)==h[f'F0_validation{fold}']
  b=torch.load(bp,map_location='cpu',weights_only=False);assert np.array_equal(b['x'],o.helper.masked(d['x'],mask))
  for k in ['patient','channel','features','y','g','train','val']:assert np.array_equal(b[k],d[k])
  for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(b['pre'][k],d['pre'][k])
  assert [q['name'] for q in o.manifest['columns']]==d['features'].tolist()
  assert np.all(b['x'][:,mp['zero']+mp['masked']]==0)
  for dest,src in mp['duplicates']:assert np.array_equal(b['x'][:,dest],b['x'][:,src]) and np.array_equal(d['raw_x'][:,dest],d['raw_x'][:,src],equal_nan=True)
  ck=torch.load(cp,map_location='cpu',weights_only=False);cell=torch.load(vp,map_location='cpu',weights_only=False);va=d['val'];seed=42+1009*fold
  assert ck['threshold']==r['threshold']==cell['threshold'] and ck['epoch']==r['epoch'] and np.array_equal(cell['indices'],va)
  core.seed_all(seed);m=o.helper.source_free_factory(o.original.SourceA0)(seed).cuda().eval();assert core.state_hash(m.state_dict())==ck['binding']['initial_hash'];m.load_state_dict(ck['model'],strict=True)
  with torch.inference_mode():ss=torch.sigmoid(m(torch.as_tensor(b['x'][va],device='cuda'),torch.as_tensor(b['g'][va],device='cuda'))).cpu().numpy()
  drift=float(np.max(np.abs(ss-cell['scores'])));assert drift<=o.lock['parity']['baseline_score_atol'] and np.array_equal(ss>=r['threshold'],cell['scores']>=r['threshold'])
  selected=core.select_threshold(d['y'][va],cell['scores'],d['patient'][va])
  for k in ['threshold','macro_f1','ez_f1']:assert abs(selected[k]-ck[k])<1e-12
  dest=a.runtime/f'fold{fold}/B0_VAL_PRIVATE.pt';assert not dest.exists();core.torch_write(dest,dict(cell,checkpoint_sha256=c.sha(cp)))
  baseline_rows.extend(patient_rows(c,b,va,cell['scores'],r['threshold'],'B0',fold));recs.append(dict(fold=fold,epoch=r['epoch'],threshold=r['threshold'],checkpoint_sha256=c.sha(cp),bank_sha256=c.sha(bp),frozen_validation_sha256=c.sha(vp),replay_max_score_drift=drift))
  tr=d['train'];x=b['x'];patient=b['patient'];stats,raw=fit_cross(x[tr],patient[tr],mp,seed);repeat,rr=fit_cross(x[tr],patient[tr],mp,seed)
  assert np.array_equal(raw,rr) and np.array_equal(stats['scale'],repeat['scale'])
  assert set(stats['fit_ids'])==set(patient[tr]) and set(stats['fit_ids']).isdisjoint(set(patient[va]))
  ev=apply(x[va],stats,mp);e=np.empty((len(x),14,2));e[tr]=raw/stats['scale'];e[va]=ev;assert sorted(np.r_[tr,va].tolist())==list(range(len(x)))
  assert np.isfinite(e).all() and np.isfinite(stats['scale']).all();perm=wrong_permutation(mp,seed);q=coordinates(e[tr]);qw=coordinates(e[tr],perm);assert np.max(np.abs(q))<=1 and np.max(np.abs(qw))<=1
  allfinite=allfinite and bool(np.any(np.abs(q)>1e-6));modelbank=dict(b,stats=stats,mapping=mp,permutation=perm,e=e)
  target=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert not target.exists();core.torch_write(target,modelbank);files[f'bank{fold}']=c.sha(target);files[f'baseline_validation{fold}']=c.sha(dest);banks.append(modelbank)
  dr=diagnostics(x[tr],patient[tr],b['center'][tr],raw,stats,mp,fold,'FIT_OOF');vr=diagnostics(x[va],patient[va],b['center'][va],ev*stats['scale'],stats,mp,fold,'VALIDATION_FULL_FIT')
  for row in vr:
   f=next(z for z in dr if z['group']==row['group'] and z['residual']==row['residual']);row['validation_to_OOF_std_ratio']=row['std']/max(f['std'],1e-6);row['mean_drift_in_OOF_SD']=(row['mean']-f['mean'])/max(f['std'],1e-6)
  ridge.extend(dr+vr)
  for ix,rs,phase in [(tr,raw,'FIT_OOF'),(va,ev*stats['scale'],'VALIDATION_FULL_FIT')]:
   for center in sorted(set(b['center'][ix])):
    k=b['center'][ix]==center;dd=diagnostics(x[ix][k],patient[ix][k],b['center'][ix][k],rs[k],stats,mp,fold,phase)
    stability.extend(dict(center=str(center),**r) for r in dd)
  for phase,ix in [('FIT_OOF',tr),('VALIDATION_FULL_FIT',va)]:
   for label,p in [('CORRECT',None),('WRONG',perm)]:
    cq=coordinates(e[ix],p);w=weights(patient[ix])
    for j,g in enumerate(mp['triples']):
     qq=weighted_quantiles(cq[:,j],w);interaction.append(dict(fold=fold,phase=phase,pairing=label,group=j,feature=g['feature'],aggregation=g['aggregation'],mean=float(mean(cq[:,j],w)),std=float(sd(cq[:,j],w)),mean_abs=float(mean(np.abs(cq[:,j]),w)),near_zero_fraction=float(mean((np.abs(cq[:,j])<=1e-6).astype(float),w)),saturation_fraction=float(mean((np.abs(cq[:,j])>=.99).astype(float),w)),**dict(zip(['q00','q01','q05','q50','q95','q99','q100'],qq.tolist()))))
  del m;print('F0_REPLAY_NUISANCE_FROZEN',fold,flush=True)
 actual=pd.DataFrame(baseline_rows)[['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr']].mean().to_numpy();expected=np.array([.7165694393593033,.5085590459690381,.6420669456105904,.43351598043908673,.6765905768221185]);assert np.allclose(actual,expected,atol=1e-12,rtol=0)
 core.torch_write(a.runtime/'BASELINE_METRICS_PRIVATE.pt',baseline_rows)
 c.write_json(pub/'BASELINE_REPRODUCTION.json',dict(status='PASS',binding=o.bind,selections=recs,metrics=dict(zip(['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr'],actual.tolist())),frozen_predictions_reused=True,no_outer_data_or_outcomes_loaded=True))
 c.write_json(a.runtime/'NUISANCE_FREEZE.json',dict(status='PASS',binding=o.bind,files=files,mapping=mp,permutations=[b['permutation'].tolist() for b in banks],no_new_development_outcomes=True))
 assert allfinite,'COVE_NO_IDENTIFIABLE_INTERACTION_COORDINATES'
 contract=load('cove_contract',a.root/'tests/contract.py');eng=contract.run(a,o,banks)
 c.write_json(pub/'ENGINEERING_TESTS.json',dict(status='PASS',binding=o.bind,**eng))
 prows=[];pprivate=[]
 for fold,b in enumerate(banks,1):
  rp,pp=probe(b,b['stats'],mp,b['permutation'],fold);prows.extend(rp);pprivate.extend(pp)
 core.torch_write(a.runtime/'FIT_PROBE_METRICS_PRIVATE.pt',pprivate)
 pd.DataFrame(prows).to_csv(pub/'FIT_INNER_PROBE.csv',index=False);pd.DataFrame(pprivate).groupby('probe')[['ez_auroc','ez_auprc','log_loss']].mean().reset_index().to_csv(pub/'FIT_INNER_PROBE_SUMMARY.csv',index=False)
 pd.DataFrame(ridge).to_csv(pub/'CONDITIONAL_RIDGE_AUDIT.csv',index=False);pd.DataFrame(stability).to_csv(pub/'RESIDUAL_STABILITY.csv',index=False);pd.DataFrame(interaction).to_csv(pub/'INTERACTION_COORDINATES.csv',index=False)
 c.write_json(pub/'STAGE_A_GATE.json',dict(status='PASS',binding=o.bind,triples=14,finite_Ridge_and_coordinates=True,patient_crossfit_reproducible=True,OOF_heldout_exclusion=True,normalizer_fit_only=True,max_normalization_amplification=1000000,interaction_bound=1,all_fold_interactions_nonzero=True,weak_probe_not_selection=True,nuisance_freeze_sha256=c.sha(a.runtime/'NUISANCE_FREEZE.json'),no_outer_access=True))
 c.write_json(pub/'PARAMETER_AUDIT.json',dict(status='PASS',parameters=o.lock['parameters'],new_trainable_per_arm=28,Ridge_trainable_torch_parameters=0))
 pd.DataFrame([dict(arm=arm,total=o.lock['parameters'][arm],backbone=8817,correction=0 if arm=='B0' else 28) for arm in o.lock['arms']]).to_csv(pub/'PARAMETER_AUDIT.csv',index=False)
 assert setup(a).bind==o.bind
 c.write_json(a.runtime/'EXPERIMENT_FREEZE.json',dict(status='PASS',binding=o.bind,nuisance_sha256=c.sha(a.runtime/'NUISANCE_FREEZE.json'),baseline_sha256=c.sha(pub/'BASELINE_REPRODUCTION.json'),engineering_sha256=c.sha(pub/'ENGINEERING_TESTS.json'),files=files))
 print('COVE_ALL_PRETRAIN_GATES_PASS',flush=True)

if __name__=='__main__':main()
