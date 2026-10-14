"""Independent saved-score/source-map audit for completed phases8..12."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'phase8/code'));sys.path.insert(0,str(ROOT/'phase9/code'));sys.path.insert(0,str(ROOT/'phase11/code'))
from calibrate import fit_map,apply_map
from offset import transform,effective
from dispersion import transform_scaled,cutoff
from explore import context,read,sha,write,original_modules

def main():
    baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime');c,core,fac,trainer,records,rec,banks,hist,deps=context(Path(r'E:\DRE-nips\new-pipeline\7-11'),baseline);metrics,boot=original_modules();reports=[]
    for phase in range(8,13):
        rt=Path(f'D:/d1_macro70_phase{phase}_runtime');fr=json.loads((rt/'CALIBRATION_FREEZE_PRIVATE.json').read_text());binding=json.loads((rt/'INPUT_FREEZE_PRIVATE.json').read_text());assert fr['binding']==binding and binding['code']=={p.name:sha(p) for p in (ROOT/f'phase{phase}/code').glob('*.py')}
        saved=pd.DataFrame(read(rt/'DEVELOPMENT_METRICS_PRIVATE.pkl'));rows=[];cells=[]
        for r in fr['selections']:
            arm,base,fold=r['arm'],r['base'],r['fold'];cpmap=rt/f'calibration_private/{arm}_fold{fold}.pkl';assert sha(cpmap)==r['calibration_sha256'];q=read(cpmap);assert q['binding']==binding;cp=Path(q['base_checkpoint']);assert sha(cp)==q['base_checkpoint_sha256']==r['base_checkpoint_sha256'];bp=baseline/f'D1/fold{fold}/BANK_PRIVATE.pt';assert sha(bp)==q['D1_bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False);assert set(d['patient']).isdisjoint(banks[fold-1]['roles']['test']) and set(d['pre']['fit_patient_ids'])==banks[fold-1]['roles']['fit'];prediction=cp.parents[3]/f'predictions_private/{base}_fold{fold}.pkl';assert sha(prediction)==q['validation_predictions_sha256'];pred=read(prediction);ll=pred['modes']['full']['logit'] if 'modes' in pred else pred['logit'];score=torch.sigmoid(torch.from_numpy(ll)).numpy();va=d['val'];coefficient=q.get('coefficient',0.)
            if 'exponent' in q:shift,median,scale=transform_scaled(score,d['patient'][va],coefficient,q['exponent'])
            elif 'coefficient' in q:shift,median=transform(score,d['patient'][va],coefficient)
            else:shift=score
            mapping,counts=fit_map(core,d,shift,q['default'],q.get('policy','SOURCE_MIN1'));assert mapping==q['mapping'] and counts==q['counts'];source_tau=apply_map(d['center'][va],q['default'],mapping)
            actual_tau=cutoff(source_tau,median,scale,coefficient) if 'exponent' in q else effective(source_tau,median,coefficient) if 'coefficient' in q else source_tau
            assert np.array_equal(score>=actual_tau,shift>=source_tau)
            for pid in sorted(set(d['patient'][va])):
                k=np.flatnonzero(d['patient'][va]==pid);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')];ii=va[k];assert np.ptp(actual_tau[k])==0;rows.append(dict(arm=arm,fold=fold,patient=str(pid),**metrics.independent(d['y'][ii],score[k],float(actual_tau[k][0]))))
            cells.append(dict(arm=arm,fold=fold,base_checkpoint_sha256=sha(cp),calibration_sha256=sha(cpmap),source_map_recomputed_from_validation=True,original_ranking_scores_unchanged=True,own_fold_test_excluded=True))
        actual=pd.DataFrame(rows);keys=['arm','fold','patient'];l=actual.set_index(keys).sort_index();r=saved[saved.arm.str.contains('__')].set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[metrics.METRICS],r[metrics.METRICS],atol=1e-12,rtol=0,equal_nan=True)
        means=actual.groupby('arm')[metrics.METRICS].mean();summary=pd.read_csv(rt/'results/DEVELOPMENT_SUMMARY.csv').set_index('arm');assert np.allclose(summary.loc[means.index,metrics.METRICS],means,rtol=0,atol=1e-12,equal_nan=True)
        for arm in means.index:
            assert len(actual[actual.arm==arm])==65 and actual[actual.arm==arm].patient.nunique()==47
            base=arm.split('__')[0];left=saved[saved.arm==arm].set_index(['fold','patient']).sort_index();right=saved[saved.arm==base].set_index(['fold','patient']).sort_index()
            assert np.array_equal(left[['ez_auroc','ez_auprc','ez_mrr','top1_is_ez']].to_numpy(),right[['ez_auroc','ez_auprc','ez_mrr','top1_is_ez']].to_numpy(),equal_nan=True)
        bs=pd.read_csv(rt/'results/PAIRED_BOOTSTRAP.csv')
        for arm in means.index:
            base=arm.split('__')[0]
            for reference in ['D1',base]:
                for row in boot.compare(c,saved[saved.arm==arm],saved[saved.arm==reference],metrics.METRICS):
                    q=bs[(bs.comparison==arm+' minus '+reference)&(bs.metric==row['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[row['delta'],row['ci_low'],row['ci_high']],rtol=0,atol=1e-12,equal_nan=True)
        chosen=means.sort_values(['macro_f1','ez_f1'],ascending=False).index[0];assert chosen==fr['selected'];report=dict(status='PASS',phase=phase,calibration_cells=len(cells),configurations=len(means),independent_patient_metric_cells=len(rows),original_ranking_exact_unchanged=True,all_source_maps_recomputed_validation_only=True,own_fold_test_excluded=True,bootstrap_independent_multiplicity=True,selected=chosen,MacroF1=float(means.loc[chosen,'macro_f1']),no_TEST_access=True,source_sha256=sha(Path(__file__)),aggregate_sha256={p.name:sha(p) for p in (rt/'results').glob('*.csv')});write(rt/'results/INDEPENDENT_VERIFICATION.json',report);reports.append(report);print('CALIBRATION_PHASE_INDEPENDENT_PASS',phase,len(cells),float(means.loc[chosen,'macro_f1']),flush=True)
    write(Path(r'D:\d1_macro70_calibration_audit')/'ALL_PHASES_VERIFIED.json',dict(status='PASS',phases=reports,total_calibration_cells=sum(r['calibration_cells'] for r in reports),no_training_or_TEST_access=True));print('ALL_CALIBRATION_AUDITS_COMPLETE',flush=True)

if __name__=='__main__':main()
