"""Additional score-only ranking/group attribution and nuisance-summary verification."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from runtime import args,setup
from conditional_ridge import coordinates,weights,mean,sd

def main():
 a=args();o=setup(a);c=o.c;pub=o.pub;status=__import__('json').loads((pub/'RUN_STATUS.json').read_text());assert status['status']=='ALL_DEVELOPMENT_AND_MECHANISM_AUDITS_COMPLETE' and status['binding']==o.bind
 ranking=[];summaries=[];coeff=pd.read_csv(pub/'COVE_COEFFICIENTS.csv');inter=pd.read_csv(pub/'INTERACTION_COORDINATES.csv');groups=pd.read_csv(pub/'GROUP_INTERACTION_EFFECTS.csv');ims=pd.read_csv(pub/'COVE_INTERVENTIONS.csv').set_index('arm');checked=0
 for fold in range(1,6):
  b=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',weights_only=False);mech=torch.load(a.runtime/f'fold{fold}/MECHANISM_PRIVATE.pt',weights_only=False);va=b['val'];full=mech['full_scores'];tau=mech['threshold']
  for tag,score in mech['scores'].items():
   for pid in sorted(set(b['patient'][va])):
    ix=np.flatnonzero(b['patient'][va]==pid);ix=ix[np.argsort(b['channel'][va][ix].astype(str),kind='stable')];p=full[ix];q=score[ix];y=b['y'][va[ix]];i,j=np.triu_indices(len(ix),1)
    changed=int(np.count_nonzero(np.sign(p[i]-p[j])!=np.sign(q[i]-q[j])));total=len(i);fp=p<tau;qp=q<tau
    ranking.append(dict(intervention=tag,patient=pid,changed_pairs=changed,total_pairs=total,pair_fraction=changed/total if total else 0.,mean_abs_score_drift=float(np.abs(p-q).mean()),top1_changed=float(np.argmin(p)!=np.argmin(q)),TP_gained=int(np.count_nonzero((y==0)&fp&~qp)),TP_lost=int(np.count_nonzero((y==0)&~fp&qp)),FP_added=int(np.count_nonzero((y==1)&fp&~qp)),FP_removed=int(np.count_nonzero((y==1)&~fp&qp))));checked+=1
  for phase,ix in [('FIT_OOF',b['train']),('VALIDATION_FULL_FIT',va)]:
   e=b['e'][ix];w=weights(b['patient'][ix])
   for label,perm in [('CORRECT',None),('WRONG',b['permutation'])]:
    q=coordinates(e,perm)
    for j in range(14):
     row=inter[(inter.fold==fold)&(inter.phase==phase)&(inter.pairing==label)&(inter.group==j)].iloc[0]
     for k,v in {'mean':mean(q[:,j],w),'std':sd(q[:,j],w),'mean_abs':mean(np.abs(q[:,j]),w),'near_zero_fraction':mean((np.abs(q[:,j])<=1e-6).astype(float),w),'saturation_fraction':mean((np.abs(q[:,j])>=.99).astype(float),w)}.items():assert np.isclose(row[k],v,atol=1e-12,rtol=0)
     cr=coeff[(coeff.fold==fold)&(coeff.phase==phase)&(coeff.group==j)].iloc[0]
     assert abs(cr.correct_q_variance-sd(coordinates(e)[:,j],w)**2)<1e-12 and abs(cr.wrong_q_variance-sd(coordinates(e,b['permutation'])[:,j],w)**2)<1e-12
 df=pd.DataFrame(ranking);saved=pd.read_csv(pub/'RANKING_ATTRIBUTION.csv').set_index('intervention')
 for tag,q in df.groupby('intervention'):
  row=saved.loc[tag];values=dict(mean_patient_pair_fraction=q.pair_fraction.mean(),pooled_within_patient_pair_fraction=q.changed_pairs.sum()/q.total_pairs.sum(),changed_pairs=q.changed_pairs.sum(),total_pairs=q.total_pairs.sum())
  values.update(q[['mean_abs_score_drift','top1_changed','TP_gained','TP_lost','FP_added','FP_removed']].mean().to_dict())
  for k,v in values.items():assert np.isclose(row[k],v,atol=1e-12,rtol=0)
 for _,row in groups.iterrows():
  tag=f'GROUP_{int(row.group):02d}_DISABLED'
  for m in o.lock['metrics']:assert np.isclose(row['full_minus_'+m],ims.loc[tag,'full_minus_'+m],atol=1e-12,rtol=0)
  assert np.isclose(row.mean_patient_pair_fraction,saved.loc[tag,'mean_patient_pair_fraction'],atol=1e-12,rtol=0)
 c.write_json(pub/'ATTRIBUTION_VERIFICATION.json',dict(status='PASS',ranking_patient_cells=checked,all14_group_effects=True,all280_pair_coordinate_summaries=True,no_model_inference=True,no_outer_access=True,script_sha256=c.sha(Path(__file__))))
 print('ATTRIBUTION_VERIFICATION_PASS',checked,flush=True)

if __name__=='__main__':main()
