"""Frozen label-free normalization diagnostics; anonymous distributions only."""
import numpy as np
import pandas as pd
import torch


def ratio(a,b):
 return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)


def statistics(x,mapping,eps=1e-5):
 u=np.asarray(x,np.float64);v=u[:,mapping['stat_indices']];dup=u[:,list(mapping['duplicates'])]
 mean88=u.mean(1);mean56=v.mean(1);var88=u.var(1);var56=v.var(1)
 sec88=(u*u).mean(1);sec56=(v*v).mean(1);r88=np.sqrt(sec88);r56=np.sqrt(sec56)
 full_ss=((u-mean88[:,None])**2).sum(1);dup_ss=((dup-mean88[:,None])**2).sum(1)
 secsum=(u*u).sum(1);dsecsum=(dup*dup).sum(1);nonzero=np.r_[mapping['stat_indices'],list(mapping['duplicates'])]
 a72=u[:,nonzero];mean72=a72.mean(1);var72=a72.var(1)
 return dict(mean88=mean88,mean56=mean56,var88=var88,var56=var56,RMS88=r88,RMS56=r56,
  RMS56_over_RMS88_raw=ratio(r56,r88),RMS56_over_RMS88_eps=np.sqrt((sec56+eps)/(sec88+eps)),
  mean56_minus_mean88=mean56-mean88,var56_over_var88_eps=(var56+eps)/(var88+eps),
  inverse_LN88=1/np.sqrt(var88+eps),inverse_LN56=1/np.sqrt(var56+eps),
  inverse_RMS88=1/np.sqrt(sec88+eps),inverse_RMS56=1/np.sqrt(sec56+eps),
  extreme_inverse_LN88=(1/np.sqrt(var88+eps)>10).astype(float),extreme_inverse_LN56=(1/np.sqrt(var56+eps)>10).astype(float),
  extreme_inverse_RMS88=(1/np.sqrt(sec88+eps)>10).astype(float),extreme_inverse_RMS56=(1/np.sqrt(sec56+eps)>10).astype(float),
  duplicate_second_moment_contribution88=dsecsum/88,duplicate_mean_contribution88=dup.sum(1)/88,
  duplicate_energy_fraction=ratio(dsecsum,secsum),duplicate_centered_LN_energy_fraction=ratio(dup_ss,full_ss),
  zero_centered_LN_energy_fraction=ratio(16*mean88**2,full_ss),zero_variance_contribution88=16*mean88**2/88,
  RMS72_over_RMS88_eps=np.sqrt((secsum/72+eps)/(sec88+eps)),var72_over_var88_eps=(var72+eps)/(var88+eps),
  mean72_minus_mean88=mean72-mean88)


def quantile_summary(values,weights):
 x=np.asarray(values,np.float64);w=np.asarray(weights,np.float64);good=np.isfinite(x)
 out=dict(finite_fraction=float(w[good].sum()/w.sum()))
 if not good.any():return {**out,**{k:np.nan for k in ['mean','std','minimum','q05','q25','median','q75','q95','maximum']}}
 x=x[good];w=w[good]/w[good].sum();mean=float(x@w)
 order=np.argsort(x,kind='stable');xx=x[order];ww=w[order];cdf=np.cumsum(ww);cdf[-1]=1.
 qs=xx[np.searchsorted(cdf,[.05,.25,.50,.75,.95],side='left')]
 return dict(**out,mean=mean,std=float(np.sqrt(((x-mean)**2)@w)),minimum=float(x.min()),maximum=float(x.max()),**dict(zip(['q05','q25','median','q75','q95'],map(float,qs))))


def aggregate_distributions(table,fields):
 grouping=[(['role'],table.groupby('role',sort=True)),(['role','fold'],table.groupby(['role','fold'],sort=True)),
  (['role','center'],table.groupby(['role','center'],sort=True))]
 out=[]
 for keys,groups in grouping:
  for group,part in groups:
   group=group if isinstance(group,tuple) else (group,);meta=dict(zip(keys,group))
   unit=part['unit'].to_numpy();units=np.unique(unit);weight=np.zeros(len(part))
   for p in units:
    take=unit==p;weight[take]=1/(len(units)*take.sum())
   for field in fields:out.append(dict(**meta,statistic=field,patient_fold_units=len(units),unique_patients=part.patient.nunique(),channels=len(part),weighting='equal_patient_fold_channel_mass',**quantile_summary(part[field],weight)))
 return pd.DataFrame(out)


def rank_change(p,q):
 i,j=np.triu_indices(len(p),1)
 return float(np.mean(np.sign(p[i]-p[j])!=np.sign(q[i]-q[j]))) if len(i) else 0.


def frozen_audit(a,c,core,original,models,binding,freeze_sha):
 stat_tables=[];output_tables=[];affine=[];ranks=[]
 for fold in range(1,6):
  d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
  x=d['x'];stats=statistics(x,d['mapping']);meta=pd.DataFrame(dict(patient=d['patient'],center=d['center'],fold=fold,unit=[str(fold)+':'+str(p) for p in d['patient']]))
  for role,ix in [('FIT',d['train']),('VAL',d['val'])]:
   table=meta.iloc[ix].reset_index(drop=True).assign(role=role)
   stat_tables.append(pd.concat([table,pd.DataFrame({k:v[ix] for k,v in stats.items()})],axis=1))
  b0=torch.load(a.runtime/f'fold{fold}/B0_VAL_PRIVATE.pt',map_location='cpu',weights_only=False)['scores']
  for arm in ['B0','B1','B2','B3']:
   cp=a.previous/f'no_source/fold{fold}/TRAIN/BEST_PRIVATE.pt' if arm=='B0' else a.runtime/f'fold{fold}/{arm}/BEST_PRIVATE.pt'
   ck=torch.load(cp,map_location='cpu',weights_only=False);m=models.constructor(original.SourceA0,arm,d['mapping'])().cuda().eval();m.load_state_dict(ck['model'],strict=True)
   tx=torch.tensor(x,device='cuda',requires_grad=True);norm=m.network[0](tx);linear=m.network[1](norm);logits=m.network(tx).squeeze(-1)
   grad=torch.autograd.grad(logits.sum(),tx)[0];assert torch.isfinite(grad).all()
   out={'pre_RMS':np.sqrt(np.mean(x.astype(np.float64)**2,axis=1)),
    'post_RMS':norm.detach().double().square().mean(1).sqrt().cpu().numpy(),
    'input_gradient_L2':grad.detach().double().norm(dim=1).cpu().numpy(),
    'linear_activation_L2':linear.detach().double().norm(dim=1).cpu().numpy(),
    'post_zero_abs_mean':norm[:,d['mapping']['zeros']].detach().double().abs().mean(1).cpu().numpy()}
   for role,ix in [('FIT',d['train']),('VAL',d['val'])]:
    table=meta.iloc[ix].reset_index(drop=True).assign(role=role,arm=arm)
    output_tables.append(pd.concat([table,pd.DataFrame({k:v[ix] for k,v in out.items()})],axis=1))
   for name in ['weight','bias']:
    values=getattr(m.network[0],name).detach().double().cpu().numpy();affine.append(dict(arm=arm,fold=fold,parameter='gamma' if name=='weight' else 'beta',**quantile_summary(values,np.ones(88)/88)))
   if arm=='B0':continue
   va=d['val'];cell=torch.load(a.runtime/f'fold{fold}/{arm}_VAL_PRIVATE.pt',map_location='cpu',weights_only=False);assert cell['binding']==freeze_sha
   for p in sorted(set(d['patient'][va])):
    k=np.flatnonzero(d['patient'][va]==p);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')];ix=va[k]
    full=c.metric(d['y'][ix],cell['scores'][k],cell['threshold']);base_tau=torch.load(a.runtime/f'fold{fold}/B0_VAL_PRIVATE.pt',map_location='cpu',weights_only=False)['threshold'];base=c.metric(d['y'][ix],b0[k],base_tau)
    rr=stats['RMS56_over_RMS88_eps'][ix]
    ranks.append(dict(arm=arm,patient=p,fold=fold,center=str(d['center'][ix[0]]),RMS_ratio_mean=float(rr.mean()),RMS_ratio_std=float(rr.std()),
     mean_abs_mean_shift=float(np.abs(stats['mean56_minus_mean88'][ix]).mean()),pair_rank_change=rank_change(cell['scores'][k],b0[k]),
     AUROC_delta=full['ez_auroc']-base['ez_auroc'],AUPRC_delta=full['ez_auprc']-base['ez_auprc']))
 stat=pd.concat(stat_tables,ignore_index=True);outputs=pd.concat(output_tables,ignore_index=True);rank=pd.DataFrame(ranks)
 core.torch_write(a.runtime/'NORMALIZATION_MECHANISM_PRIVATE.pt',dict(binding=freeze_sha,statistics=stat,outputs=outputs,rank=rank))
 fields=list(stats);all_summary=aggregate_distributions(stat,fields)
 duplicate=[k for k in fields if k.startswith(('duplicate_','zero_','RMS72','var72','mean72'))]
 all_summary[~all_summary.statistic.isin(duplicate)].to_csv(a.runtime/'public/NORMALIZATION_STATISTICS.csv',index=False)
 all_summary[all_summary.statistic.isin(duplicate)].to_csv(a.runtime/'public/DUPLICATE_INFLUENCE.csv',index=False)
 blocks=[]
 for arm,part in outputs.groupby('arm'):
  block=aggregate_distributions(part,list(out));block['arm']=arm;blocks.append(block)
 pd.concat(blocks,ignore_index=True).to_csv(a.runtime/'public/NORMALIZATION_OUTPUT_AUDIT.csv',index=False)
 pd.DataFrame(affine).to_csv(a.runtime/'public/AFFINE_PARAMETER_AUDIT.csv',index=False)
 corr=[]
 for arm,part in rank.groupby('arm'):
  for estimator,q in [('65_appearances',part),('47_unique_ID_means',part.groupby('patient')[['RMS_ratio_mean','RMS_ratio_std','mean_abs_mean_shift','pair_rank_change','AUROC_delta','AUPRC_delta']].mean())]:
   for predictor in ['RMS_ratio_mean','RMS_ratio_std','mean_abs_mean_shift']:
    for outcome in ['pair_rank_change','AUROC_delta','AUPRC_delta']:
     for method in ['pearson','spearman']:
      estimable=q[predictor].nunique()>1 and q[outcome].nunique()>1
      corr.append(dict(arm=arm,estimator=estimator,n=len(q),predictor=predictor,outcome=outcome,method=method,
       correlation=float(q[predictor].corr(q[outcome],method=method)) if estimable else np.nan,status='descriptive' if estimable else 'not_estimable'))
 pd.DataFrame(corr).to_csv(a.runtime/'public/RANKING_NORMALIZATION_CORRELATIONS.csv',index=False)
 rank.groupby('arm')[['RMS_ratio_mean','RMS_ratio_std','mean_abs_mean_shift','pair_rank_change','AUROC_delta','AUPRC_delta']].mean().reset_index().to_csv(a.runtime/'public/RANKING_CHANGE_SUMMARY.csv',index=False)
