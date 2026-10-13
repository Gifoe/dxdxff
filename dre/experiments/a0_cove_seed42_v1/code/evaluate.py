"""One fixed development pass; frozen interventions, original ID-cluster draws."""
import json
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint,patient_rows
from cove_model import constructor
from conditional_ridge import coordinates,weights,mean,sd
from bootstrap import compare

def rank_counts(a,b):
 i,j=np.triu_indices(len(a),1);return int(np.sum(np.sign(a[i]-a[j])!=np.sign(b[i]-b[j]))),len(i)

def gate(o,df,verified=False):
 metrics=o.lock['metrics'];s=df.groupby('arm')[metrics].mean();f=df.groupby(['arm','fold'])[metrics].mean()
 positive=sum(f.loc['B3','ez_auroc']>f.loc['B0','ez_auroc']);g=o.lock['gates']
 conditions=dict(B3_B0_AUROC=float(s.loc['B3','ez_auroc']-s.loc['B0','ez_auroc'])>=g['B3_B0_AUROC'],B3_B1_AUROC=float(s.loc['B3','ez_auroc']-s.loc['B1','ez_auroc'])>=g['B3_B1_AUROC'],
  B3_B2_AUROC=float(s.loc['B3','ez_auroc']-s.loc['B2','ez_auroc'])>=g['B3_B2_AUROC'],AUPRC_nondecrease=s.loc['B3','ez_auprc']>=s.loc['B0','ez_auprc'],MacroF1_nondecrease=s.loc['B3','macro_f1']>=s.loc['B0','macro_f1'],
  EZF1_preserved=s.loc['B3','ez_f1']>=s.loc['B0','ez_f1']-.005,specificity_preserved=s.loc['B3','specificity']>=s.loc['B0','specificity']-.010,positive_folds=int(positive)>=4,correct_beats_wrong=s.loc['B3','ez_auroc']>s.loc['B2','ez_auroc'],engineering_and_independent_verification=verified)
 return dict(status='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,positive_folds=int(positive),deltas={r:float(s.loc['B3','ez_auroc']-s.loc[r,'ez_auroc']) for r in ['B0','B1','B2']},no_new_outer=True)

def main():
 a=args();o=setup(a);c,core=o.c,o.core;pub=o.pub;assert not (pub/'RUN_STATUS.json').exists()
 pre=json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text());fr=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert pre['binding']==fr['binding']==o.bind and len(fr['selections'])==15
 assert c.sha(a.runtime/'NUISANCE_FREEZE.json')==pre['nuisance_sha256']==fr['nuisance_sha256'];fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json')
 rows=torch.load(a.runtime/'BASELINE_METRICS_PRIVATE.pt',weights_only=False);interventions=[];ranking=[];coeffs=[];files={};replay=[]
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['files'][f'bank{fold}'];b=torch.load(bp,map_location='cpu',weights_only=False);va=b['val'];tx=torch.as_tensor(b['x'][va],device='cuda')
  for arm in ['B1','B2','B3']:
   r=next(r for r in fr['selections'] if r['arm']==arm and r['fold']==fold);cp=checkpoint(a,arm,fold);assert c.sha(cp)==r['checkpoint_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);tau=r['threshold']
   m=constructor(o.original.SourceA0,arm,b['mapping'],b['stats'],b['permutation'],b)().cuda().eval();m.load_state_dict(ck['model'],strict=True)
   pp=a.runtime/f'fold{fold}/{arm}_VAL_PRIVATE.pt'
   if pp.exists():
    cell=torch.load(pp,map_location='cpu',weights_only=False);assert cell['binding']==fs and cell['checkpoint_sha256']==c.sha(cp) and json.loads(pp.with_suffix('.sha256.json').read_text())==dict(binding=fs,sha256=c.sha(pp));score=cell['scores']
   else:
    with torch.inference_mode():ll=m(tx);score=torch.sigmoid(ll).cpu().numpy();indexed=torch.sigmoid(m(torch.as_tensor(va,device='cuda',dtype=torch.float32)[:,None])).cpu().numpy()
    drift=float(np.abs(score-indexed).max());assert drift<=o.lock['parity']['baseline_score_atol'] and np.array_equal(score>=tau,indexed>=tau)
    selected=core.select_threshold(b['y'][va],score,b['patient'][va])
    for key in ['threshold','macro_f1','ez_f1']:assert abs(selected[key]-ck[key])<1e-12
    cell=dict(binding=fs,indices=va,scores=score,threshold=tau,checkpoint_sha256=c.sha(cp),adapter_replay_drift=drift);core.torch_write(pp,cell);c.write_json(pp.with_suffix('.sha256.json'),dict(binding=fs,sha256=c.sha(pp)))
   rows.extend(patient_rows(c,b,va,score,tau,arm,fold));files[f'{arm}_{fold}']=c.sha(pp);replay.append(dict(arm=arm,fold=fold,adapter_score_drift=cell['adapter_replay_drift'],selection_exact=True))
   if arm!='B3':del m;continue
   mp=a.runtime/f'fold{fold}/MECHANISM_PRIVATE.pt'
   if mp.exists():
    mech=torch.load(mp,map_location='cpu',weights_only=False);assert mech['binding']==fs and json.loads(mp.with_suffix('.sha256.json').read_text())==dict(binding=fs,sha256=c.sha(mp))
   else:
    ss={}
    with torch.inference_mode():
     for tag in o.lock['interventions']:ss[tag]=torch.sigmoid(m(tx,intervention=tag)).cpu().numpy()
     for j in range(14):ss[f'GROUP_{j:02d}_DISABLED']=torch.sigmoid(m(tx,group=j)).cpu().numpy()
    mech=dict(binding=fs,indices=va,scores=ss,threshold=tau,full_scores=score,checkpoint_sha256=c.sha(cp));core.torch_write(mp,mech);c.write_json(mp.with_suffix('.sha256.json'),dict(binding=fs,sha256=c.sha(mp)))
   files[f'mechanism{fold}']=c.sha(mp)
   for tag,ss in mech['scores'].items():
    interventions.extend(patient_rows(c,b,va,ss,tau,tag,fold))
    for pid in sorted(set(b['patient'][va])):
     k=np.flatnonzero(b['patient'][va]==pid);k=k[np.argsort(b['channel'][va][k].astype(str),kind='stable')];aa=score[k];bb=ss[k];y=b['y'][va[k]];n,total=rank_counts(aa,bb);full=aa<tau;disabled=bb<tau
     ranking.append(dict(fold=fold,patient=pid,center=str(b['center'][va[k[0]]]),intervention=tag,changed_pairs=n,total_pairs=total,pair_fraction=n/total if total else 0.,mean_abs_score_drift=float(np.abs(aa-bb).mean()),top1_changed=float(np.argmin(aa)!=np.argmin(bb)),TP_gained=int(((y==0)&full&~disabled).sum()),TP_lost=int(((y==0)&~full&disabled).sum()),FP_added=int(((y==1)&full&~disabled).sum()),FP_removed=int(((y==1)&~full&disabled).sum())))
   with torch.inference_mode():
    tf=torch.as_tensor(b['x'][b['train']],device='cuda');ef=torch.as_tensor(b['e'][b['train']],device='cuda',dtype=torch.float64)
    corrected_fit=m.transformed(tf,ef).cpu().numpy();corrected_val=m.transformed(tx).cpu().numpy()
   for j,q in enumerate(b['mapping']['triples']):
    for phase,ix,corr in [('FIT_OOF',b['train'],corrected_fit),('VALIDATION_FULL_FIT',va,corrected_val)]:
     w=weights(b['patient'][ix]);e=b['e'][ix];qc=coordinates(e);qw=coordinates(e,b['permutation'])
     coeffs.append(dict(fold=fold,phase=phase,group=j,feature=q['feature'],aggregation=q['aggregation'],alpha=float(m.alpha[j].detach().cpu()),beta=float(m.beta[j].detach().cpu()),
      alpha_bounded=float(.25*torch.tanh(m.alpha[j]).detach().cpu()),beta_bounded=float(.25*torch.tanh(m.beta[j]).detach().cpu()),delta_abs_correction=float(mean(np.abs(corr[:,q['delta']]-b['x'][ix,q['delta']]),w)),ratio_abs_correction=float(mean(np.abs(corr[:,q['ratio']]-b['x'][ix,q['ratio']]),w)),
      correct_q_variance=float(sd(qc[:,j],w)**2),wrong_q_variance=float(sd(qw[:,j],w)**2),correct_near_zero_fraction=float(mean((np.abs(qc[:,j])<=1e-6).astype(float),w)),wrong_near_zero_fraction=float(mean((np.abs(qw[:,j])<=1e-6).astype(float),w))))
   del m
  print('COVE_DEVELOPMENT_FOLD_SEALED',fold,flush=True)
 assert len(rows)==260 and len(interventions)==1170 and len(ranking)==1170
 core.torch_write(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pt',rows);core.torch_write(a.runtime/'INTERVENTION_METRICS_PRIVATE.pt',interventions);core.torch_write(a.runtime/'RANKING_PRIVATE.pt',ranking)
 c.write_json(a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json',dict(binding=fs,files=files,no_outer_access=True))
 df=pd.DataFrame(rows);mi=pd.DataFrame(interventions);at=pd.DataFrame(ranking);metrics=o.lock['metrics']
 s=df.groupby('arm')[metrics].mean().reset_index();s['parameters']=s.arm.map(o.lock['parameters']);s.to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
 df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
 for key in ['fold','center','channel_stratum']:
  pd.DataFrame([dict(arm=arm,**{key:v},appearances=len(q),unique_patients=q.patient.nunique(),channels=int(q.channels.sum()),**q[metrics].mean().to_dict()) for (arm,v),q in df.groupby(['arm',key])]).to_csv(pub/f'DEVELOPMENT_BY_{key.upper()}.csv',index=False)
 boot=[]
 for left,right in o.lock['bootstrap']['comparisons']:boot.extend(dict(comparison=left+' minus '+right,**q) for q in compare(c,df[df.arm==left],df[df.arm==right],o.lock['bootstrap']['metrics']))
 pd.DataFrame(boot).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
 ims=mi.groupby('arm')[metrics].mean().reset_index();base=df[df.arm=='B3'][metrics].mean()
 for metric in metrics:ims['full_minus_'+metric]=base[metric]-ims[metric]
 ims.to_csv(pub/'COVE_INTERVENTIONS.csv',index=False)
 pd.DataFrame([dict(intervention=tag,appearances=len(q),unique_patients=q.patient.nunique(),mean_patient_pair_fraction=q.pair_fraction.mean(),pooled_within_patient_pair_fraction=q.changed_pairs.sum()/q.total_pairs.sum(),changed_pairs=int(q.changed_pairs.sum()),total_pairs=int(q.total_pairs.sum()),**q[['mean_abs_score_drift','top1_changed','TP_gained','TP_lost','FP_added','FP_removed']].mean().to_dict()) for tag,q in at.groupby('intervention')]).to_csv(pub/'RANKING_ATTRIBUTION.csv',index=False)
 pd.DataFrame(coeffs).to_csv(pub/'COVE_COEFFICIENTS.csv',index=False)
 mech_boot=[]
 for tag in o.lock['interventions']:mech_boot.extend(dict(comparison='B3 minus '+tag,**q) for q in compare(c,df[df.arm=='B3'],mi[mi.arm==tag],o.lock['bootstrap']['metrics']))
 pd.DataFrame(mech_boot).to_csv(pub/'INTERVENTION_BOOTSTRAP.csv',index=False)
 groupout=[]
 for j,q in enumerate(b['mapping']['triples']):
  tag=f'GROUP_{j:02d}_DISABLED';effect=ims[ims.arm==tag].iloc[0];rank=pd.read_csv(pub/'RANKING_ATTRIBUTION.csv').set_index('intervention').loc[tag]
  groupout.append(dict(group=j,feature=q['feature'],aggregation=q['aggregation'],**{f'full_minus_{m}':float(effect[f'full_minus_{m}']) for m in metrics},mean_patient_pair_fraction=float(rank.mean_patient_pair_fraction),pooled_pair_fraction=float(rank.pooled_within_patient_pair_fraction),original_features_not_removed=True,no_feature_selection=True))
 pd.DataFrame(groupout).to_csv(pub/'GROUP_INTERACTION_EFFECTS.csv',index=False)
 cg=gate(o,df,verified=False);c.write_json(pub/'VALIDATION_GATE.json',cg)
 c.write_json(pub/'REPLAY_AUDIT.json',dict(status='PASS',rows=replay,binding=o.bind,all15_selected_metrics_exact=True,original88_interface=True,no_outer_access=True))
 c.write_json(pub/'RUN_STATUS.json',dict(status='DEVELOPMENT_COMPLETE_PENDING_INDEPENDENT_VERIFICATION',new_model_cells=15,validation_appearances=65,unique_patients=47,intervention_cells=1170,prediction_seal_sha256=c.sha(a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json'),freeze_sha256=fs,binding=o.bind,no_outer_access=True))
 print('COVE_DEVELOPMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
