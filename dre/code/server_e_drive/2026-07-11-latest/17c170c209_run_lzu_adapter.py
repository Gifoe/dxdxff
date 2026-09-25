#!/usr/bin/env python3
"""Train a bounded LZU-only residual adapter from frozen RCC fold outputs."""
from __future__ import annotations
import argparse, ast, json
from pathlib import Path
import sys
import numpy as np, pandas as pd, torch
from torch.optim import AdamW

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neuroez_c.lzu_bounded_adapter import LZUBoundedResidualAdapter, apply_lzu_adapter
from neuroez_c.lzu_adapter_loss import adapter_loss
from neuroez_c.lzu_adapter_protocol import deterministic_lzu_split, non_lzu_invariance_audit

def _embedding(value: object) -> list[float]:
    if not isinstance(value,str) or not value.strip(): raise ValueError('RCC fit ledger lacks contextual_embedding; rerun RCC after the embedding-output change')
    return list(map(float, ast.literal_eval(value)))

def _features(frame: pd.DataFrame) -> torch.Tensor:
    result=[]
    for _,group in frame.groupby('subject_id',sort=False):
        logits=group.base_nez_logit.to_numpy(float); med=np.median(logits); mad=1.4826*np.median(np.abs(logits-med)); z=(logits-med)/max(mad,1e-5)
        rank=pd.Series(-logits).rank(method='average',pct=True).to_numpy(float)
        anchor=group.get('negative_anchor_distance',pd.Series(np.zeros(len(group)))).to_numpy(float); az=(anchor-anchor.mean())/max(anchor.std(),1e-5)
        for i,(_,row) in enumerate(group.iterrows()): result.append(_embedding(row.contextual_embedding)+[float(logits[i]),float(z[i]),float(rank[i]),float(az[i])])
    return torch.tensor(result,dtype=torch.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--selection_manifest',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--window_cache_path',default='');p.add_argument('--allowed_subjects_ledger',default='');p.add_argument('--fixed_fold_manifest',default='');p.add_argument('--require_n_patients',type=int,default=80);p.add_argument('--random_seed',type=int,default=42);p.add_argument('--lzu_adapter_hidden_dim',type=int,default=16);p.add_argument('--lzu_adapter_dropout',type=float,default=.10);p.add_argument('--lzu_adapter_max_delta',type=float,default=.15);p.add_argument('--lzu_adapter_lr',type=float,default=1e-4);p.add_argument('--lzu_adapter_weight_decay',type=float,default=1e-4);p.add_argument('--lzu_adapter_epochs',type=int,default=20);p.add_argument('--lzu_adapter_patience',type=int,default=5);a=p.parse_args()
 manifest=json.loads(Path(a.selection_manifest).read_text()); shared=Path(a.selection_manifest).parent/manifest['selected_profile']; out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 if manifest.get('selection_source')!='validation_only':raise RuntimeError('Adapter requires a validation-only RCC selection manifest')
 all_rows=[]; audits=[]; residual=[]
 for fold in range(1,6):
  fit_path=shared/f'fit_channel_predictions_neuroez_v2_fold_{fold}.csv'; test_path=shared/f'test_channel_predictions_neuroez_v2_fold_{fold}.csv'
  if not fit_path.is_file() or not test_path.is_file(): raise FileNotFoundError(f'Missing frozen RCC fit/test ledger for fold {fold}')
  fit=pd.read_csv(fit_path);test=pd.read_csv(test_path); lzu_subjects=fit.loc[fit.center.astype(str).str.lower().eq('lzu'),'subject_id'].drop_duplicates().tolist();train_ids,val_ids=deterministic_lzu_split(lzu_subjects,a.random_seed+1009*fold)
  for sid in train_ids:audits.append({'outer_fold':fold,'subject_id':sid,'split':'adapter_fit','center':'lzu','seed':a.random_seed+1009*fold})
  for sid in val_ids:audits.append({'outer_fold':fold,'subject_id':sid,'split':'adapter_validation','center':'lzu','seed':a.random_seed+1009*fold})
  train=fit[fit.subject_id.isin(train_ids)].copy(); val=fit[fit.subject_id.isin(val_ids)].copy()
  x=_features(train); xv=_features(val); y=torch.tensor(train.true_ez.to_numpy(float)); yv=torch.tensor(val.true_ez.to_numpy(float)); pid=torch.tensor(pd.factorize(train.subject_id)[0]); pidv=torch.tensor(pd.factorize(val.subject_id)[0])
  model=LZUBoundedResidualAdapter(x.shape[1],a.lzu_adapter_hidden_dim,a.lzu_adapter_dropout,a.lzu_adapter_max_delta); opt=AdamW(model.parameters(),lr=a.lzu_adapter_lr,weight_decay=a.lzu_adapter_weight_decay); best=None;wait=0
  for epoch in range(1,a.lzu_adapter_epochs+1):
   model.train();delta=model(x,torch.ones(len(x),dtype=torch.bool));loss=adapter_loss(torch.tensor(train.base_nez_logit.to_numpy(float))+delta,y,pid,delta)['total_loss'];opt.zero_grad();loss.backward();opt.step()
   model.eval();
   with torch.no_grad(): vd=model(xv,torch.ones(len(xv),dtype=torch.bool));vl=adapter_loss(torch.tensor(val.base_nez_logit.to_numpy(float))+vd,yv,pidv,vd)['total_loss']
   score=-float(vl); 
   if best is None or score>best[0]:best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()});wait=0
   else:wait+=1
   if wait>=a.lzu_adapter_patience:break
  model.load_state_dict(best[1]); torch.save({'adapter_state_dict':model.state_dict(),'shared_threshold':manifest['per_fold_global_threshold'].get(str(fold)),'n_trainable_shared_parameters':0,'n_trainable_adapter_parameters':sum(p.numel() for p in model.parameters())},out/f'fold_{fold}_adapter.pt')
  # Infer all test rows. Hard gate makes every non-LZU value exactly unchanged.
  xt=_features(test);is_lzu=torch.tensor(test.center.astype(str).str.lower().eq('lzu').to_numpy());base=torch.tensor(test.base_nez_logit.to_numpy(float));
  with torch.no_grad():delta=model(xt,is_lzu);adapted=apply_lzu_adapter(base,delta,is_lzu);score=torch.sigmoid(adapted);thr=float(manifest['per_fold_global_threshold'][str(fold)]);pred=(score>=thr).numpy().astype(int)
  test['adapter_delta']=delta.numpy();test['adapted_nez_logit']=adapted.numpy();test['adapted_score_nez']=score.numpy();test['adapted_predicted_nez']=pred;test['outer_fold']=fold;all_rows.append(test)
  non=~is_lzu.numpy();aud=non_lzu_invariance_audit(base.numpy()[non],adapted.numpy()[non],torch.sigmoid(base).numpy()[non],score.numpy()[non],(torch.sigmoid(base).numpy()[non]>=thr),(pred[non]>0));aud['outer_fold']=fold;residual.append({'outer_fold':fold,'mean_abs_adapter_delta':float(delta.abs().mean()),'p95_abs_adapter_delta':float(torch.quantile(delta.abs(),.95)),'max_abs_adapter_delta':float(delta.abs().max()),'saturation_rate':float((delta.abs()>=.95*a.lzu_adapter_max_delta).float().mean())});audits.append(aud)
 ledger=pd.concat(all_rows,ignore_index=True);ledger.to_csv(out/'adapter_oof_channel_ledger.csv',index=False);pd.DataFrame(audits).to_csv(out/'lzu_adapter_split_audit.csv',index=False);pd.DataFrame(residual).to_csv(out/'adapter_residual_diagnostics.csv',index=False);non=[x for x in audits if 'passed'in x];Path(out/'non_lzu_invariance_audit.json').write_text(json.dumps({'folds':non,'passed':all(x['passed'] for x in non)},indent=2))
if __name__=='__main__':main()
