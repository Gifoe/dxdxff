"""Gated matched compact retraining; never reads a held-out Outer label/output."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from io_utils import sha,write,save,read
from dependencies import context
from compact_delta_classifier import CompactDelta,inflate
from evaluate import METRICS,patient_rows
from bootstrap import compare

STAGE_B_FILES=['dependencies.py','compact_delta_classifier.py','train.py','evaluate.py','bootstrap.py']

def aggregate(runtime,c,rs,comparisons):
 pub=runtime/'results';df=pd.DataFrame(rs)
 for arm,q in df.groupby('arm'):assert (len(q),q.patient.nunique())==(65,47),arm
 df.groupby('arm')[METRICS].mean().reset_index().to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
 for key in ['fold','center']:
  pd.DataFrame([dict(arm=arm,**{key:value},appearances=len(q),unique_IDs=q.patient.nunique(),channels=int(q.channels.sum()),**q[METRICS].mean().to_dict()) for (arm,value),q in df.groupby(['arm',key])]).to_csv(pub/f'DEVELOPMENT_BY_{key.upper()}.csv',index=False)
 df.groupby(['arm','patient'])[METRICS].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
 boot=[]
 for left,right in comparisons:boot.extend(dict(comparison=f'{left} minus {right}',**r) for r in compare(c,df[df.arm==left],df[df.arm==right],METRICS))
 pd.DataFrame(boot).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);save(runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rs)
 return df.groupby('arm')[METRICS].mean()

def stage_b(a):
 runtime=a.runtime;pub=runtime/'results';root=Path(__file__).parents[1]
 c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,runtime)
 bind=dict(stage_A=sha(pub/'STAGE_A_GATE.json'),dependencies=deps,code={f:sha(root/'code'/f) for f in STAGE_B_FILES},test=sha(root/'tests/test_stage_b.py'))
 freeze=runtime/'STAGE_B_INPUT_FREEZE_PRIVATE.json'
 if freeze.exists():assert json.loads(freeze.read_text())==bind
 else:write(freeze,bind)
 if (pub/'STAGE_B_GATE.json').exists():assert json.loads((pub/'STAGE_B_GATE.json').read_text())['binding']==bind;print('STAGE_B_ALREADY_COMPLETE',flush=True);return
 rs=[];parities=[];selections=[];preaudit=[];histories=[];parameters=[]
 def factory(seed=42):return CompactDelta(fac(seed).network,records)
 original_factory=trainer.SourceA0;trainer.SourceA0=factory
 try:
  for fold,(d,(hd,cp,r)) in enumerate(zip(banks,hist),1):
   seed=42+1009*fold;va=d['val'];ck=torch.load(cp,map_location='cpu',weights_only=False);m=fac(seed).cuda().eval();m.load_state_dict(ck['model'],strict=True);compact=CompactDelta(m.network,records).cuda().eval();tx=torch.as_tensor(hd['x'],device='cuda');inds=[q['index'] for q in records]
   with torch.inference_mode():
    ll=m(tx,torch.as_tensor(d['g'],device='cuda'));cl=compact(tx[:,inds]);scores=torch.sigmoid(ll[va]).cpu().numpy();cs=torch.sigmoid(cl[va]).cpu().numpy()
   ld=float((ll-cl).abs().max().cpu());sd=float(np.abs(scores-cs).max());assert ld<=2e-6 and sd<=1e-6 and np.array_equal(scores>=ck['threshold'],cs>=ck['threshold'])
   md=copy.deepcopy(m.network).double().cpu();cd=CompactDelta(md,records).double().eval();rng=np.random.default_rng(seed);synthetic=torch.from_numpy(rng.normal(size=(65,14)))
   with torch.inference_mode():dd=float((md(inflate(synthetic,records)).squeeze(-1)-cd(synthetic)).abs().max())
   assert dd<=1e-12
   sel=core.select_threshold(d['y'][va],scores,d['patient'][va])
   for k in ['threshold','macro_f1','ez_f1']:assert abs(sel[k]-ck[k])<1e-12
   rs.extend(patient_rows(c,d,scores,ck['threshold'],'D0',fold));save(runtime/f'scores/D0_fold{fold}_PRIVATE.pkl',dict(score=scores,threshold=ck['threshold'],checkpoint_sha256=sha(cp),binding=bind))
   parities.append(dict(fold=fold,logit_fp32_max=ld,score_fp32_max=sd,logit_fp64_synthetic_max=dd,all_legal_bank_rows_checked=True,validation_decisions_exact=True,original_selection_exact=True,checkpoint_sha256=sha(cp)))
   del m,compact,md,cd,tx
   initial_hash=[]
   for arm,key in [('E1','E14'),('D1','R14')]:
    rr=rec[key][d['archive_indices']];x,pre=core.prepare(rr,d['patient'],d['train']);assert np.isfinite(x).all() and set(pre['fit_patient_ids'])==d['roles']['fit']
    if arm=='E1':assert np.array_equal(x,hd['x'][:,inds]),'E14 original PatientZ/imputer/scaler parity failed'
    # Independent reference for Patient Z; this never inspects clinical y.
    z=core.patient_z(rr,d['patient']);zz=np.empty_like(z)
    for pid in np.unique(d['patient']):
     ii=d['patient']==pid;v=rr[ii].astype(np.float64);sdv=np.nanstd(v,axis=0);zz[ii]=(v-np.nanmean(v,axis=0))/np.where(sdv>1e-8,sdv,1.)
    assert np.array_equal(z,zz,equal_nan=True)
    bp=runtime/f'{arm}/fold{fold}/BANK_PRIVATE.pt';new=dict(d,x=x,pre=pre,input_source=key,binding=bind)
    if bp.exists():
     b=torch.load(bp,map_location='cpu',weights_only=False);assert b['binding']==bind and np.array_equal(b['x'],x)
    else:core.torch_write(bp,new)
    core.seed_all(seed);initial_hash.append(core.state_hash(factory(seed).state_dict()));cell=bp.parent/'TRAIN';t=time.perf_counter()
    model,best,history=trainer.train(x,d['y'],d['patient'],d['g'],d['train'],va,cell,seed,c.digest(dict(binding=bind,bank=sha(bp),arm=arm)),device='cuda',penalty=0.)
    with torch.inference_mode():ss=torch.sigmoid(model(torch.as_tensor(x[va],device='cuda'),torch.as_tensor(d['g'][va],device='cuda'))).cpu().numpy()
    assert core.select_threshold(d['y'][va],ss,d['patient'][va])['threshold']==best['threshold']
    rs.extend(patient_rows(c,d,ss,best['threshold'],arm,fold));cpp=cell/'BEST_PRIVATE.pt';save(runtime/f'scores/{arm}_fold{fold}_PRIVATE.pkl',dict(score=ss,threshold=best['threshold'],checkpoint_sha256=sha(cpp),binding=bind))
    selections.append(dict(arm=arm,fold=fold,epoch=best['epoch'],threshold=best['threshold'],checkpoint_sha256=sha(cpp),bank_sha256=sha(bp),initial_state_sha256=initial_hash[-1],epochs=len(history),wall_seconds=time.perf_counter()-t))
    preaudit.append(dict(arm=arm,fold=fold,fit_patients=len(d['roles']['fit']),validation_patients=len(d['roles']['validation']),Outer_patients_loaded=0,PatientZ_independent_exact=True,scaler_fit_only=True,input_finite=True,engineered_values_in_D1_predictor=False))
    histories.extend(dict(arm=arm,fold=fold,**r) for r in history);parameters.append(dict(arm=arm,fold=fold,total=1537,trainable=1537));print('RAW_DELTA_STAGE_B_SELECTED',arm,fold,best['epoch'],flush=True)
    write(pub/'RUN_STATUS.json',dict(status='STAGE_B_TRAINING',cells_complete=len(selections),total=10,no_new_Outer=True));del model
   assert initial_hash[0]==initial_hash[1], 'E1/D1 initialization mismatch'
 finally:trainer.SourceA0=original_factory
 means=aggregate(runtime,c,rs,[('D1','E1'),('E1','D0'),('D1','D0')]);expected=[.716600,.504414,.642347,.443493];assert np.allclose(means.loc['D0',['ez_auroc','ez_auprc','macro_f1','ez_f1']],expected,atol=5.1e-7,rtol=0)
 diff=means.loc['D1']-means.loc['E1'];q=means.loc['D1'];conditions=dict(AUROC_gap=diff.ez_auroc>=-.015,AP_gap=diff.ez_auprc>=-.020,MacroF1_gap=diff.macro_f1>=-.015,EZF1_gap=diff.ez_f1>=-.020,D1_AUROC=q.ez_auroc>=.700,D1_MacroF1=q.macro_f1>=.620,integrity=True,Raw_derived_label_blind=True,all5folds=True)
 passed=all(conditions.values());terminal='STAGE_B_PASS' if passed else 'RAW_DELTA_PREDICTIVE_RECOVERY_FAILED'
 write(pub/'STAGE_B_GATE.json',dict(binding=bind,pass_gate=passed,terminal=terminal,conditions=conditions,D1_E1_difference=diff.to_dict(),D1=q.to_dict(),no_Outer=True));write(runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json',dict(binding=bind,selections=selections,stage_B_pass=passed))
 pd.DataFrame(selections).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False);pd.DataFrame(parameters).to_csv(pub/'PARAMETER_AUDIT.csv',index=False)
 write(pub/'COMPACT_PARITY.json',dict(status='PASS',folds=parities,parameters=1537,original_D0_replayed_not_retrained=True));write(pub/'PREPROCESSING_PARITY.json',dict(status='PASS',cells=preaudit,initializations_matched=True,clinical_label_scope='own FIT and validation only'))
 write(pub/'RUN_STATUS.json',dict(status='STAGE_B_COMPLETE',terminal=terminal,new_training_cells=10,no_new_Outer=True));print('RAW_DELTA_STAGE_B_GATE',terminal,flush=True)
 if not passed:
  write(pub/'SCIENTIFIC_GATE.json',dict(terminal=terminal,stage='B',D2_D3='NOT_RUN_STAGE_B_GATE_FAILED',no_rescue=True,no_Outer=True))
  pd.DataFrame([dict(model=m,status='NOT_RUN_STAGE_B_GATE_FAILED') for m in ['D2','D3']]).to_csv(pub/'RAW_BRANCH_INTERVENTIONS.csv',index=False)
  pd.DataFrame([dict(model=m,status='NOT_RUN_STAGE_B_GATE_FAILED') for m in ['D2','D3']]).to_csv(pub/'ERROR_AND_RANKING_AUDIT.csv',index=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args();stage_b(a)
