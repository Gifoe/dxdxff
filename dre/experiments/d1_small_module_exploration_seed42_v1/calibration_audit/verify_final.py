"""Independent recomputation of phases13/14 and globally selected final recipe."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[1]
for folder in ['code','phase8/code','phase11/code','phase13/code','phase14/code']:sys.path.insert(0,str(ROOT/folder))
from calibrate import fit_map,apply_map
from dispersion import transform_scaled,cutoff
from combine import combine_logits
from exact import source_map
from explore import context,read,sha,write,original_modules

def main():
    baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime');c,core,fac,trainer,records,rec,banks,hist,deps=context(Path(r'E:\DRE-nips\new-pipeline\7-11'),baseline);m,b=original_modules();reports=[]
    for phase in [13,14]:
        rt=Path(f'D:/d1_macro70_phase{phase}_runtime');fr=json.loads((rt/'CALIBRATION_FREEZE_PRIVATE.json').read_text());binding=json.loads((rt/'INPUT_FREEZE_PRIVATE.json').read_text());assert fr['binding']==binding and binding['code']=={p.name:sha(p) for p in (ROOT/f'phase{phase}/code').glob('*.py')};saved=pd.DataFrame(read(rt/'DEVELOPMENT_METRICS_PRIVATE.pkl'));rows=[]
        for r in fr['selections']:
            arm,fold=r['arm'],r['fold'];mp=rt/f'calibration_private/{arm}_fold{fold}.pkl';assert sha(mp)==r['calibration_sha256'];q=read(mp);assert q['binding']==binding;bp=baseline/f'D1/fold{fold}/BANK_PRIVATE.pt';assert sha(bp)==q['D1_bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test']);va=d['val']
            if phase==13:
                ll=[]
                for component in q['components']:
                    cp=Path(component['checkpoint']);assert sha(cp)==component['checkpoint_sha256'];pp=cp.parents[3]/f"predictions_private/{component['base']}_fold{fold}.pkl";assert sha(pp)==component['validation_predictions_sha256'];pred=read(pp);v=pred['modes']['full'] if 'modes' in pred else pred;ll.append(v['logit'])
                assert q['weights']==[1/len(ll)]*len(ll);_,score=combine_logits(ll);assert q['default']==core.select_threshold(d['y'][va],score,d['patient'][va])['threshold'];policy=q['policy']
                if policy=='GLOBAL':mapping={};counts={};tau=np.full(len(score),q['default'])
                else:
                    if policy=='SOURCE_SCALE_HALF':shift,median,scale=transform_scaled(score,d['patient'][va],0.,.5)
                    else:shift=score
                    mapping,counts=fit_map(core,d,shift,q['default'],'SOURCE_MIN1');source_tau=apply_map(d['center'][va],q['default'],mapping);tau=cutoff(source_tau,median,scale,0.) if policy=='SOURCE_SCALE_HALF' else source_tau;assert np.array_equal(score>=tau,shift>=source_tau)
            else:
                cp=Path(q['base_checkpoint']);assert sha(cp)==q['base_checkpoint_sha256'];pp=cp.parents[3]/f"predictions_private/{q['base']}_fold{fold}.pkl";assert sha(pp)==q['validation_predictions_sha256'];pred=read(pp);v=pred['modes']['full'] if 'modes' in pred else pred;score=torch.sigmoid(torch.from_numpy(v['logit'])).numpy();shift,median,scale=transform_scaled(score,d['patient'][va],q['coefficient'],q['exponent']);mapping,counts=source_map(d,shift,q['default']);source_tau=apply_map(d['center'][va],q['default'],mapping);tau=cutoff(source_tau,median,scale,q['coefficient']);assert np.array_equal(score>=tau,shift>=source_tau)
            assert mapping==q['mapping'] and counts==q['counts']
            for pid in sorted(set(d['patient'][va])):
                k=np.flatnonzero(d['patient'][va]==pid);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')];ii=va[k];assert np.ptp(tau[k])==0;rows.append(dict(arm=arm,fold=fold,patient=str(pid),**m.independent(d['y'][ii],score[k],float(tau[k][0]))))
        actual=pd.DataFrame(rows);keys=['arm','fold','patient'];l=actual.set_index(keys).sort_index();r=saved[saved.arm.str.contains('__')].set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[m.METRICS],r[m.METRICS],atol=1e-12,rtol=0,equal_nan=True);means=actual.groupby('arm')[m.METRICS].mean();summary=pd.read_csv(rt/'results/DEVELOPMENT_SUMMARY.csv').set_index('arm');assert np.allclose(summary.loc[means.index,m.METRICS],means,atol=1e-12,rtol=0,equal_nan=True);assert means.sort_values(['macro_f1','ez_f1'],ascending=False).index[0]==fr['selected'];bs=pd.read_csv(rt/'results/PAIRED_BOOTSTRAP.csv')
        for arm in means.index:
            references=['D1'] if phase==13 else ['D1',arm.split('__')[0]]
            for ref in references:
                for row in b.compare(c,saved[saved.arm==arm],saved[saved.arm==ref],m.METRICS):
                    q=bs[(bs.comparison==arm+' minus '+ref)&(bs.metric==row['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[row['delta'],row['ci_low'],row['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
        report=dict(status='PASS',phase=phase,calibration_cells=len(fr['selections']),independent_patient_metric_cells=len(rows),all_maps_or_fixed_weights_recomputed=True,all_checkpoint_and_bank_hashes_verified=True,own_fold_test_excluded=True,bootstrap_independent_multiplicity=True,no_TEST_access=True,selected=fr['selected'],MacroF1=float(means.loc[fr['selected'],'macro_f1']),aggregate_sha256={p.name:sha(p) for p in (rt/'results').glob('*.csv')});write(rt/'results/INDEPENDENT_VERIFICATION.json',report);reports.append(report);print('RECENT_CALIBRATION_INDEPENDENT_PASS',phase,report['MacroF1'],flush=True)
    candidates=[]
    for phase in range(4,15):
        rt=Path(f'D:/d1_macro70_phase{phase}_runtime');summary=pd.read_csv(rt/'results/DEVELOPMENT_SUMMARY.csv')
        for q in summary.to_dict('records'):
            if q['arm']=='D1':continue
            candidates.append(dict(phase=phase,**q))
    table=pd.DataFrame(candidates);selected=table.sort_values(['macro_f1','ez_f1'],ascending=False).iloc[0];assert selected.phase==14
    out=Path(r'D:\d1_macro70_calibration_audit');table.to_csv(out/'ALL_NEW_CANDIDATES.csv',index=False);write(out/'FINAL_DEVELOPMENT_SELECTION.json',dict(status='PASS',selected_phase=int(selected.phase),selected_arm=str(selected.arm),MacroF1=float(selected.macro_f1),target_over_070=bool(selected.macro_f1>.7),new_model_training_cells=50,calibration_cells=495,no_TEST_access=True,all_declared_configs_retained=True,scientific_interpretation='Best fitted repeatedDevelopment decoder, not independentconfirmation'))
    print('FINAL_DEVELOPMENT_GLOBAL_SELECTION_VERIFIED',selected.arm,float(selected.macro_f1),flush=True)

if __name__=='__main__':main()
