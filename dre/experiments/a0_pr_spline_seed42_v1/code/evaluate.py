"""Frozen development metrics, registered gates and curve interventions only."""
import json
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint,patient_rows
from bootstrap import compare


def rank_change(p,q):
 ij=np.triu_indices(len(p),1)
 return float(np.mean(np.sign(p[ij[0]]-p[ij[1]])!=np.sign(q[ij[0]]-q[ij[1]]))) if len(ij[0]) else 0.


def main():
 a=args();lock,ref,c,core,original,trainer,models,ks,banks,binding=setup(a);pub=a.runtime/'public'
 assert not (pub/'RUN_STATUS.json').exists()
 pre=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());eng=json.loads((pub/'ENGINEERING_TESTS.json').read_text())
 assert pre['binding']==eng['binding']==binding and eng['status']=='PASS'
 frozen=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert frozen['binding']==binding
 freeze_sha=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json');rows=[];interventions=[];responses=[];attribution=[];files={}
 metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN']
 permutation=np.random.default_rng(42).permutation(56)
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
  d=torch.load(bp,map_location='cpu',weights_only=False);va=d['val'];tx=torch.as_tensor(d['x'][va],device='cuda')
  for arm in lock['arms']:
   rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold);cp=checkpoint(a,arm,fold)
   assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256']
   ck=torch.load(cp,map_location='cpu',weights_only=False);tau=ck['threshold'];path=bp.parent/f'{arm}_VAL_PRIVATE.pt'
   if arm=='B0':cell=torch.load(path,map_location='cpu',weights_only=False);score=cell['scores']
   else:
    m=models.constructor(original.SourceA0,arm,d['stats'])().cuda().eval();m.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode():ll=m(tx);score=torch.sigmoid(ll).cpu().numpy()
    assert not path.exists();core.torch_write(path,dict(binding=freeze_sha,indices=va,logits=ll.cpu().numpy(),scores=score,threshold=tau,checkpoint_sha256=c.sha(cp)))
   cells=patient_rows(c,d,va,score,tau,arm,fold);assert len(cells)==13
   assert abs(np.mean([r['macro_f1'] for r in cells])-ck['macro_f1'])<1e-12 and abs(np.mean([r['ez_f1'] for r in cells])-ck['ez_f1'])<1e-12
   rows.extend(cells);files[f'{arm}_{fold}']=c.sha(path)
   if arm!='B3':continue
   assert (m.transform.theta.detach()[~m.transform.slope_mask]==0).all()
   with torch.inference_mode():
    identity=m(tx,identity=True);perm=m(tx,permutation=torch.as_tensor(permutation,device='cuda'))
    fit_x=torch.as_tensor(d['x'][d['train']],device='cuda');fit_new=m.transformed(fit_x).cpu().numpy();val_new=m.transformed(tx).cpu().numpy()
   iq=torch.sigmoid(identity).cpu().numpy();pq=torch.sigmoid(perm).cpu().numpy()
   mp=bp.parent/'MECHANISM_PRIVATE.pt';assert not mp.exists()
   core.torch_write(mp,dict(binding=freeze_sha,indices=va,permutation=permutation,full_scores=score,identity_scores=iq,permuted_scores=pq,threshold=tau))
   files[f'mechanism_{fold}']=c.sha(mp)
   for tag,s in [('B3_IDENTITY',iq),('B3_CURVE_PERMUTED',pq)]:interventions.extend(patient_rows(c,d,va,s,tau,tag,fold))
   with torch.inference_mode():slopes=m.transform.slopes().cpu().numpy();theta=m.transform.theta.cpu().numpy()
   for i,j in enumerate(d['stats']['representatives']):
    active=d['stats']['slope_mask'][i];bb=d['stats']['boundary_mask'][i];knot=d['stats']['knots'][i,bb];ss=slopes[i,active]
    xf=d['x'][d['train'],j];xv=d['x'][va,j]
    responses.append(dict(fold=fold,feature=str(d['features'][j]),representative_index=int(j),active_intervals=int(active.sum()),
     knots=json.dumps(knot.tolist()),slopes=json.dumps(ss.tolist()),theta=json.dumps(theta[i,active].tolist()),
     max_abs_slope_minus1=float(np.abs(ss-1).max()) if len(ss) else 0.,mean_abs_slope_minus1=float(np.abs(ss-1).mean()) if len(ss) else 0.,
     FIT_mean_abs_transform=float(np.mean(np.abs(fit_new[:,j]-xf))),validation_mean_abs_transform=float(np.mean(np.abs(val_new[:,j]-xv))),
     FIT_outer_interval_fraction=float(np.mean((xf<knot[0])|(xf>knot[-1]))) if len(knot) else np.nan,
     validation_outer_interval_fraction=float(np.mean((xv<knot[0])|(xv>knot[-1]))) if len(knot) else np.nan,
     FIT_min=float(xf.min()),FIT_max=float(xf.max()),validation_min=float(xv.min()),validation_max=float(xv.max()),
     validation_median=float(np.median(xv)),identity_degenerate=not bool(active.any())))
   for p in sorted(set(d['patient'][va])):
    k=np.flatnonzero(d['patient'][va]==p);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')]
    y=d['y'][va[k]];full=score[k]<tau;ident=iq[k]<tau;per=pq[k]<tau
    attribution.append(dict(patient=p,fold=fold,center=str(d['center'][va[k[0]]]),
     identity_pair_rank_change=rank_change(score[k],iq[k]),permuted_pair_rank_change=rank_change(score[k],pq[k]),
     identity_top1_change=float(np.argmin(score[k])!=np.argmin(iq[k])),permuted_top1_change=float(np.argmin(score[k])!=np.argmin(pq[k])),
     identity_EZ_TP_gained=int(((y==0)&full&~ident).sum()),identity_EZ_TP_lost=int(((y==0)&~full&ident).sum()),
     identity_EZ_FP_added=int(((y==1)&full&~ident).sum()),identity_EZ_FP_removed=int(((y==1)&~full&ident).sum()),
     mean_abs_input_transform=float(np.mean(np.abs(val_new[k]-d['x'][va[k]]))),
     permuted_EZ_TP_gained=int(((y==0)&full&~per).sum()),permuted_EZ_FP_added=int(((y==1)&full&~per).sum())))
 assert len(rows)==260 and len(interventions)==130 and len(responses)==280 and len(attribution)==65
 core.torch_write(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pt',rows);core.torch_write(a.runtime/'INTERVENTION_METRICS_PRIVATE.pt',interventions)
 core.torch_write(a.runtime/'ATTRIBUTION_PRIVATE.pt',attribution)
 c.write_json(a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json',dict(binding=freeze_sha,files=files,new_outer_accessed=False))
 df=pd.DataFrame(rows);mi=pd.DataFrame(interventions);at=pd.DataFrame(attribution)
 summary=df.groupby('arm')[metrics].mean().reset_index();summary['parameters']=summary.arm.map(lock['parameters'])
 summary.to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
 df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
 for key in ['fold','center']:
  out=[dict(arm=arm,**{key:v},appearances=len(part),unique_patients=part.patient.nunique(),**part[metrics].mean().to_dict()) for (arm,v),part in df.groupby(['arm',key])]
  pd.DataFrame(out).to_csv(pub/('DEVELOPMENT_BY_'+key.upper()+'.csv'),index=False)
 effects=[]
 for left,right in lock['comparisons']:
  for r in compare(c,df[df.arm==left],df[df.arm==right],metrics):effects.append(dict(comparison=left+' minus '+right,**r))
 pd.DataFrame(effects).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
 pd.concat([df[df.arm=='B3'],mi]).groupby('arm')[metrics].mean().reset_index().to_csv(pub/'SPLINE_INTERVENTIONS.csv',index=False)
 mech=[]
 for right in ['B3_IDENTITY','B3_CURVE_PERMUTED']:
  for r in compare(c,df[df.arm=='B3'],mi[mi.arm==right],metrics):mech.append(dict(comparison='B3 minus '+right,**r))
 pd.DataFrame(mech).to_csv(pub/'SPLINE_INTERVENTION_BOOTSTRAP.csv',index=False)
 pd.DataFrame(responses).to_csv(pub/'SPLINE_RESPONSE_AUDIT.csv',index=False)
 acols=[k for k in at if k not in ['patient','fold','center']]
 ats=[dict(group='ALL',appearances=65,**at[acols].mean().to_dict())]
 ats.extend(dict(group=str(v),grouping=key,appearances=len(part),**part[acols].mean().to_dict()) for key in ['fold','center'] for v,part in at.groupby(key))
 pd.DataFrame(ats).to_csv(pub/'RANKING_ATTRIBUTION.csv',index=False)
 t=summary.set_index('arm');f=df.groupby(['fold','arm']).ez_auroc.mean().unstack();wins=int((f.B3>f.B0).sum())
 criteria={
  'AUROC_B3_minus_B0_ge0.010':bool(t.loc['B3','ez_auroc']-t.loc['B0','ez_auroc']>=.010),
  'AUPRC_not_decreased':bool(t.loc['B3','ez_auprc']>=t.loc['B0','ez_auprc']),
  'MacroF1_not_decreased':bool(t.loc['B3','macro_f1']>=t.loc['B0','macro_f1']),
  'EZF1_decline_at_most0.005':bool(t.loc['B3','ez_f1']-t.loc['B0','ez_f1']>=-.005),
  'AUROC_B3_minus_B1_ge0.005':bool(t.loc['B3','ez_auroc']-t.loc['B1','ez_auroc']>=.005),
  'AUROC_B3_minus_B2_ge0.005':bool(t.loc['B3','ez_auroc']-t.loc['B2','ez_auroc']>=.005),
  'positive_folds_ge4':wins>=4,
  'specificity_decline_at_most0.010':bool(t.loc['B3','specificity']-t.loc['B0','specificity']>=-.010),
  'all_pretraining_integrity_checks':True}
 assert setup(a)[-1]==binding
 aucci=next(r for r in effects if r['comparison']=='B3 minus B0' and r['metric']=='ez_auroc')
 c.write_json(pub/'VALIDATION_GATE.json',dict(pass_gate=all(criteria.values()),criteria=criteria,positive_AUROC_folds=wins,
   AUROC_B3_minus_B0_CI_excludes_zero=aucci['ci_low']>0 or aucci['ci_high']<0,new_outer_authorized=False))
 terminal='PR_SPLINE_DEVELOPMENT_PROMISING_NOT_INDEPENDENTLY_CONFIRMED' if all(criteria.values()) else 'PR_SPLINE_DEVELOPMENT_GATE_FAILED'
 c.write_json(pub/'SCIENTIFIC_TERMINAL.json',dict(terminal=terminal,development_only=True,new_outer_accessed=False))
 c.write_json(pub/'RUN_STATUS.json',dict(status='ALL_DEVELOPMENT_AND_MECHANISM_COMPLETE',new_training_cells=15,reused_baseline_cells=5,
   validation_appearances=65,unique_validation_patients=47,new_outer_accessed=False,registered_protocol_unchanged=True,development_freeze_sha256=freeze_sha))
 print('PR_SPLINE_DEVELOPMENT_COMPLETE',terminal,flush=True)


if __name__=='__main__':main()
