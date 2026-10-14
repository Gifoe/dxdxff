"""Prospective source-defined validation decoder; no TEST data loader."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'code'))
BASES=['M8_MACRO_AWARE','M18_TIME_CONTROL'];POLICIES=['SOURCE_MIN3','SOURCE_MIN1','SOURCE_SHRINK5']

def apply_map(center,default,mapping):return np.asarray([mapping.get(str(c),default) for c in center],float)

def fit_map(core,d,score,default,policy):
    va=d['val'];known=set(d['center'][d['train']].astype(str));mapping={};counts={}
    for source in sorted(set(d['center'][va].astype(str))):
        k=np.flatnonzero(d['center'][va].astype(str)==source);patients=len(set(d['patient'][va][k]));minimum=3 if policy=='SOURCE_MIN3' else 1
        if source not in known or patients<minimum:continue
        sel=core.select_threshold(d['y'][va][k],score[k],d['patient'][va][k]);tau=sel['threshold']
        if policy=='SOURCE_SHRINK5':tau=(patients*tau+5*default)/(patients+5)
        mapping[source]=float(tau);counts[source]=patients
    return mapping,counts

def patient_metrics(c,metrics,d,score,tau,arm,fold):
    out=[];va=d['val']
    for pid in sorted(set(d['patient'][va])):
        k=np.flatnonzero(d['patient'][va]==pid);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')];ii=va[k];assert np.ptp(tau[k])==0 and len(set(d['center'][ii].astype(str)))==1
        a=c.metric(d['y'][ii],score[k],tau[k][0]);b=metrics.independent(d['y'][ii],score[k],tau[k][0])
        for key in b:assert np.isclose(a[key],b[key],atol=1e-12,rtol=0,equal_nan=True)
        out.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(d['center'][ii[0]]),**a))
    return out

def main(a):
    from explore import context,original_modules,read,save,write,sha
    a.runtime.mkdir(parents=True,exist_ok=True);pub=a.runtime/'results';pub.mkdir(exist_ok=True);binding=dict(protocol=sha(ROOT/'phase8/PROTOCOL_LOCK.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},M8freeze=sha(a.m8/'MODEL_FREEZE_PRIVATE.json'),M18freeze=sha(a.phase7/'MODEL_FREEZE_PRIVATE.json'),originalD1freeze=sha(a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json'))
    fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert json.loads(fp.read_text())==binding
    else:write(fp,binding)
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);metrics,boot=original_modules();rows=read(a.phase1/'D1_METRICS_PRIVATE.pkl');selects=[];audits=[]
    for base in BASES:
        rt=a.m8 if base==BASES[0] else a.phase7;fr=json.loads((rt/'MODEL_FREEZE_PRIVATE.json').read_text())
        for fold in range(1,6):
            d=torch.load(a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test']);ref=next(q for q in fr['selections'] if q['arm']==base and q['fold']==fold);cp=rt/base/f'fold{fold}/TRAIN/BEST_PRIVATE.pt';assert sha(cp)==ref['checkpoint_sha256'];pp=rt/f'predictions_private/{base}_fold{fold}.pkl';q=read(pp);ll=q['modes']['full']['logit'] if base==BASES[0] else q['logit'];score=torch.sigmoid(torch.from_numpy(ll)).numpy();default=float(ref['threshold']);assert len(score)==len(d['val']) and np.isfinite(score).all()
            rows.extend(patient_metrics(c,metrics,d,score,np.full(len(score),default),base,fold))
            for policy in POLICIES:
                arm=base+'__'+policy;mapping,counts=fit_map(core,d,score,default,policy);tau=apply_map(d['center'][d['val']],default,mapping);rows.extend(patient_metrics(c,metrics,d,score,tau,arm,fold));target=a.runtime/f'calibration_private/{arm}_fold{fold}.pkl';cell=dict(binding=binding,base=base,policy=policy,fold=fold,base_checkpoint=str(cp),base_checkpoint_sha256=sha(cp),D1_bank_sha256=sha(a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt'),validation_predictions_sha256=sha(pp),default=default,mapping=mapping,counts=counts,fit_source_coverage=sorted(set(d['center'][d['train']].astype(str))))
                if target.exists():assert read(target)==cell
                else:save(target,cell)
                selects.append(dict(arm=arm,base=base,policy=policy,fold=fold,calibrated_groups=len(mapping),calibration_sha256=sha(target),base_checkpoint_sha256=sha(cp)));audits.append(dict(arm=arm,fold=fold,validation_patients=len(set(d['patient'][d['val']])),calibrated_sources=len(mapping),sources_below3_patients=sum(v<3 for v in counts.values()),unknown_source_fallback_global=True,ownfold_test_excluded=True));print('SOURCE_CALIBRATION_CELL_COMPLETE',base,policy,fold,flush=True)
    df=pd.DataFrame(rows)
    for arm,g in df.groupby('arm'):assert len(g)==65 and g.patient.nunique()==47
    save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows);means=df.groupby('arm')[metrics.METRICS].mean();means.reset_index().to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    for field in ['fold','center']:df.groupby(['arm',field])[metrics.METRICS].mean().reset_index().to_csv(pub/f'DEVELOPMENT_BY_{field.upper()}.csv',index=False)
    pd.DataFrame(selects).to_csv(pub/'CALIBRATION_SELECTION.csv',index=False);pd.DataFrame(audits).to_csv(pub/'CALIBRATION_PROVENANCE.csv',index=False);comparisons=[]
    for arm in means.index:
        if '__' not in arm:continue
        base=arm.split('__')[0]
        for ref in ['D1',base]:comparisons.extend(dict(comparison=arm+' minus '+ref,**q) for q in boot.compare(c,df[df.arm==arm],df[df.arm==ref],metrics.METRICS))
    pd.DataFrame(comparisons).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False);table=means.loc[[q for q in means.index if '__' in q]].reset_index();table['groups']=table.arm.map(pd.DataFrame(selects).groupby('arm').calibrated_groups.sum());chosen=table.sort_values(['macro_f1','ez_f1','groups'],ascending=[False,False,True]).iloc[0];quality=all(chosen[k]>=means.loc['D1',k] for k in ['ez_auroc','ez_auprc','ez_f1']);write(a.runtime/'CALIBRATION_FREEZE_PRIVATE.json',dict(binding=binding,selections=selects,selected=chosen.arm));write(pub/'RUN_STATUS.json',dict(status='ALL30_SOURCE_CALIBRATION_CELLS_COMPLETE',selected=chosen.arm,MacroF1=float(chosen.macro_f1),target_over_070=bool(chosen.macro_f1>.7),AUROC_AP_EZF1_vs_D1_quality=bool(quality),no_TEST_access=True,ranking_scores_exact_unchanged=True,validation_calibration_fitted_not_independent=True));print('SOURCE_CALIBRATION_PHASE8_COMPLETE',chosen.arm,float(chosen.macro_f1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_macro70_phase8_runtime'));p.add_argument('--m8',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase7',type=Path,default=Path(r'D:\d1_macro70_phase7_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
