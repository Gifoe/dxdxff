"""Fixed, label-free patient location/dispersion decoder matrix."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'phase8/code'))
from calibrate import fit_map,apply_map,patient_metrics
from dispersion import transform_scaled,cutoff
BASES=['M8_MACRO_AWARE','M18_TIME_CONTROL'];COEFFICIENTS=[0.,.25,.5,1.];EXPONENTS=[.5,1.]

def main(a):
    from explore import context,original_modules,read,save,write,sha
    a.runtime.mkdir(parents=True,exist_ok=True);pub=a.runtime/'results';pub.mkdir(exist_ok=True);assert json.loads((a.phase10/'results/RUN_STATUS.json').read_text())['status']=='ALL75_CROSS_RECIPE_CALIBRATION_CELLS_COMPLETE'
    binding=dict(protocol=sha(ROOT/'phase11/PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},phase10freeze=sha(a.phase10/'CALIBRATION_FREEZE_PRIVATE.json'),M8freeze=sha(a.m8/'MODEL_FREEZE_PRIVATE.json'),M18freeze=sha(a.phase7/'MODEL_FREEZE_PRIVATE.json'),phase8_code={p.name:sha(p) for p in (ROOT/'phase8/code').glob('*.py')})
    fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert json.loads(fp.read_text())==binding
    else:write(fp,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);metrics,boot=original_modules();rows=read(a.phase1/'D1_METRICS_PRIVATE.pkl');sels=[]
    for base in BASES:
        rt=a.m8 if base==BASES[0] else a.phase7;fr=json.loads((rt/'MODEL_FREEZE_PRIVATE.json').read_text())
        for fold in range(1,6):
            d=torch.load(a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test']);r=next(q for q in fr['selections'] if q['arm']==base and q['fold']==fold);cp=rt/base/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==r['checkpoint_sha256'];pp=rt/f'predictions_private/{base}_fold{fold}.pkl';q=read(pp);ll=q['modes']['full']['logit'] if base==BASES[0] else q['logit'];score=torch.sigmoid(torch.from_numpy(ll)).numpy();default=float(r['threshold']);rows.extend(patient_metrics(c,metrics,d,score,np.full(len(score),default),base,fold))
            for coefficient in COEFFICIENTS:
                for exponent in EXPONENTS:
                    arm=base+f'__SOURCE_LOCATION_{coefficient}_SCALE_{exponent}';shift,median,scale=transform_scaled(score,d['patient'][d['val']],coefficient,exponent);mapping,counts=fit_map(core,d,shift,default,'SOURCE_MIN1');tau=apply_map(d['center'][d['val']],default,mapping);actual_tau=cutoff(tau,median,scale,coefficient);assert np.array_equal(score>=actual_tau,shift>=tau),'Refuse numerical threshold mismatch';rows.extend(patient_metrics(c,metrics,d,score,actual_tau,arm,fold));target=a.runtime/f'calibration_private/{arm}_fold{fold}.pkl';cell=dict(binding=binding,base=base,coefficient=coefficient,exponent=exponent,fold=fold,base_checkpoint_sha256=sha(cp),base_checkpoint=str(cp),validation_predictions_sha256=sha(pp),D1_bank_sha256=sha(a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt'),default=default,mapping=mapping,counts=counts,fit_source_coverage=sorted(set(d['center'][d['train']].astype(str))))
                    if target.exists():assert read(target)==cell
                    else:save(target,cell)
                    sels.append(dict(arm=arm,base=base,coefficient=coefficient,exponent=exponent,fold=fold,calibration_sha256=sha(target),base_checkpoint_sha256=sha(cp)));print('DISPERSION_CALIBRATION_CELL',base,coefficient,exponent,fold,flush=True)
    df=pd.DataFrame(rows)
    for arm,g in df.groupby('arm'):assert len(g)==65 and g.patient.nunique()==47
    save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows);means=df.groupby('arm')[metrics.METRICS].mean();means.reset_index().to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:df.groupby(['arm',field])[metrics.METRICS].mean().reset_index().to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    pd.DataFrame(sels).to_csv(pub/'CALIBRATION_SELECTION.csv',index=False);comparisons=[]
    for arm in means.index:
        if '__' not in arm:continue
        base=arm.split('__')[0]
        for ref in ['D1',base]:comparisons.extend(dict(comparison=arm+' minus '+ref,**q) for q in boot.compare(c,df[df.arm==arm],df[df.arm==ref],metrics.METRICS))
    pd.DataFrame(comparisons).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);table=means.loc[[q for q in means.index if '__' in q]].reset_index();ss=pd.DataFrame(sels).groupby('arm').first();table['coefficient']=table.arm.map(ss.coefficient);table['exponent']=table.arm.map(ss.exponent);chosen=table.sort_values(['macro_f1','ez_f1','coefficient','exponent'],ascending=[False,False,True,True]).iloc[0];quality=all(chosen[k]>=means.loc['D1',k] for k in ['ez_auroc','ez_auprc','ez_f1']);write(a.runtime/'CALIBRATION_FREEZE_PRIVATE.json',dict(binding=binding,selections=sels,selected=chosen.arm));write(pub/'RUN_STATUS.json',dict(status='ALL80_DISPERSION_CALIBRATION_CELLS_COMPLETE',selected=chosen.arm,MacroF1=float(chosen.macro_f1),target_over_070=bool(chosen.macro_f1>.7),AUROC_AP_EZF1_vs_D1_quality=bool(quality),no_TEST_access=True,ranking_scores_exact_unchanged=True,calibration_fitted_validation_not_independent=True));print('PHASE11_DISPERSION_COMPLETE',chosen.arm,float(chosen.macro_f1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_macro70_phase11_runtime'));p.add_argument('--phase10',type=Path,default=Path(r'D:\d1_macro70_phase10_runtime'));p.add_argument('--m8',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase7',type=Path,default=Path(r'D:\d1_macro70_phase7_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
