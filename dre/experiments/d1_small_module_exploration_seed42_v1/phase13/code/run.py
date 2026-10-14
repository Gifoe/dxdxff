"""Fixed equal-weight combinations, validation-only calibration."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'phase8/code'));sys.path.insert(0,str(ROOT/'phase11/code'))
from calibrate import fit_map,apply_map,patient_metrics
from dispersion import transform_scaled,cutoff
from combine import combine_logits
COMBINATIONS={'E1_M8_M17':['M8_MACRO_AWARE','M17_TIME_SPECTRAL'],'E2_M8_M18':['M8_MACRO_AWARE','M18_TIME_CONTROL'],'E3_M17_M18':['M17_TIME_SPECTRAL','M18_TIME_CONTROL'],'E4_M8_M17_M18':['M8_MACRO_AWARE','M17_TIME_SPECTRAL','M18_TIME_CONTROL']}
POLICIES=['GLOBAL','SOURCE_MIN1','SOURCE_SCALE_HALF']

def main(a):
    from explore import context,original_modules,read,save,write,sha
    a.runtime.mkdir(parents=True,exist_ok=True);pub=a.runtime/'results';pub.mkdir(exist_ok=True);assert json.loads((Path(r'D:\d1_macro70_calibration_audit')/'ALL_PHASES_VERIFIED.json').read_text())['status']=='PASS'
    binding=dict(protocol=sha(ROOT/'phase13/PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},M8freeze=sha(a.m8/'MODEL_FREEZE_PRIVATE.json'),spectralfreeze=sha(a.phase7/'MODEL_FREEZE_PRIVATE.json'),previous_calibration_verification=sha(Path(r'D:\d1_macro70_calibration_audit')/'ALL_PHASES_VERIFIED.json'),calibration_code={str(p.relative_to(ROOT)):sha(p) for folder in ['phase8/code','phase11/code'] for p in (ROOT/folder).glob('*.py')});fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert json.loads(fp.read_text())==binding
    else:write(fp,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);metrics,boot=original_modules();rows=read(a.phase1/'D1_METRICS_PRIVATE.pkl');sels=[];identity=[]
    for name,components in COMBINATIONS.items():
        for fold in range(1,6):
            d=torch.load(a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test']);values=[];baselines=[];references=[]
            for base in components:
                rt=a.m8 if base=='M8_MACRO_AWARE' else a.phase7;fr=json.loads((rt/'MODEL_FREEZE_PRIVATE.json').read_text());r=next(q for q in fr['selections'] if q['arm']==base and q['fold']==fold);cp=rt/base/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==r['checkpoint_sha256'];pp=rt/f'predictions_private/{base}_fold{fold}.pkl';q=read(pp);v=q['modes']['full'] if 'modes' in q else q;values.append(v['logit']);baselines.append(v['base_logit']);references.append(dict(base=base,checkpoint=str(cp),checkpoint_sha256=sha(cp),validation_predictions_sha256=sha(pp)))
            assert all(np.array_equal(baselines[0],z) for z in baselines[1:]);ll,score=combine_logits(values);assert len(score)==len(d['val']);default=float(core.select_threshold(d['y'][d['val']],score,d['patient'][d['val']])['threshold']);save(a.runtime/f'combination_predictions_private/{name}_fold{fold}.pkl',dict(binding=binding,logit=ll,score=score,base_logit=baselines[0],components=references));identity.append(dict(combination=name,fold=fold,shared_D1_exact=True,equal_weights_fixed=True,components=len(components)))
            for policy in POLICIES:
                arm=name+'__'+policy;mapping={};counts={}
                if policy=='GLOBAL':actual_tau=np.full(len(score),default)
                else:
                    if policy=='SOURCE_SCALE_HALF':shift,median,scale=transform_scaled(score,d['patient'][d['val']],0.,.5)
                    else:shift=score
                    mapping,counts=fit_map(core,d,shift,default,'SOURCE_MIN1');tau=apply_map(d['center'][d['val']],default,mapping);actual_tau=cutoff(tau,median,scale,0.) if policy=='SOURCE_SCALE_HALF' else tau;assert np.array_equal(score>=actual_tau,shift>=tau)
                rows.extend(patient_metrics(c,metrics,d,score,actual_tau,arm,fold));target=a.runtime/f'calibration_private/{arm}_fold{fold}.pkl';cell=dict(binding=binding,combination=name,components=references,weights=[1/len(components)]*len(components),policy=policy,fold=fold,default=default,mapping=mapping,counts=counts,D1_bank_sha256=sha(a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt'),fit_source_coverage=sorted(set(d['center'][d['train']].astype(str))))
                if target.exists():assert read(target)==cell
                else:save(target,cell)
                sels.append(dict(arm=arm,combination=name,components=len(components),policy=policy,fold=fold,calibration_sha256=sha(target)));print('FIXED_COMBINATION_CELL',name,policy,fold,flush=True)
    df=pd.DataFrame(rows)
    for arm,g in df.groupby('arm'):assert len(g)==65 and g.patient.nunique()==47
    save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows);means=df.groupby('arm')[metrics.METRICS].mean();means.reset_index().to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:df.groupby(['arm',field])[metrics.METRICS].mean().reset_index().to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    pd.DataFrame(sels).to_csv(pub/'CALIBRATION_SELECTION.csv',index=False);write(pub/'SHARED_BASE_IDENTITY_AUDIT.json',dict(status='PASS',folds=identity));comparisons=[]
    for arm in means.index:
        if arm=='D1':continue
        comparisons.extend(dict(comparison=arm+' minus D1',**q) for q in boot.compare(c,df[df.arm==arm],df[df.arm=='D1'],metrics.METRICS))
    pd.DataFrame(comparisons).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);table=means.drop('D1').reset_index();table['components']=table.arm.map(pd.DataFrame(sels).groupby('arm').components.first());chosen=table.sort_values(['macro_f1','ez_f1','components'],ascending=[False,False,True]).iloc[0];quality=all(chosen[k]>=means.loc['D1',k] for k in ['ez_auroc','ez_auprc','ez_f1']);write(a.runtime/'CALIBRATION_FREEZE_PRIVATE.json',dict(binding=binding,selections=sels,selected=chosen.arm));write(pub/'RUN_STATUS.json',dict(status='ALL60_FIXED_COMBINATION_CELLS_COMPLETE',selected=chosen.arm,MacroF1=float(chosen.macro_f1),target_over_070=bool(chosen.macro_f1>.7),AUROC_AP_EZF1_vs_D1_quality=bool(quality),no_TEST_access=True,multiple_branch_model=True,calibration_fitted_validation_not_independent=True));print('PHASE13_FIXED_COMBINATION_COMPLETE',chosen.arm,float(chosen.macro_f1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_macro70_phase13_runtime'));p.add_argument('--m8',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase7',type=Path,default=Path(r'D:\d1_macro70_phase7_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
