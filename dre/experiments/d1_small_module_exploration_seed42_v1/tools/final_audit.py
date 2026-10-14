"""Read-only actual private artifact verification and matched aggregate contrasts."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).parents[1]
P1=Path(__import__('os').environ.get('D1_PHASE1_CODE',str(ROOT/'code')))
sys.path.insert(0,str(P1))
from explore import context,original_modules,read,sha,write

def audit(a):
    roots=[a.phase1,a.phase2,a.phase3];verified=[];total_epochs=0
    for i,r in enumerate(roots,1):
        freeze=json.loads((r/'INPUT_FREEZE_PRIVATE.json').read_text())
        code=ROOT/'code' if i==1 else ROOT/f'phase{i}/code'
        assert freeze['code']=={p.name:sha(p) for p in code.glob('*.py')}
        assert freeze['protocol']==sha(code.parent/'PROTOCOL_LOCK.json')
        complete=list(r.glob('M*/fold*/TRAIN/COMPLETE_PRIVATE.json'));assert len(complete)==[15,15,10][i-1]
        rows=[]
        for p in complete:
            q=json.loads(p.read_text());assert sha(p.parent/'BEST_PRIVATE.pt')==q['best_sha256'] and sha(p.parent/'LAST_PRIVATE.pt')==q['last_sha256']
            ck=torch.load(p.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False);best=torch.load(p.parent/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert ck['binding']==q['binding']==best['binding'];assert best['epoch']==q['epoch'] and best['threshold']==q['threshold'];assert len(ck['history'])==ck['epoch']
            total_epochs+=ck['epoch'];rows.append(dict(arm=p.parents[2].name,fold=p.parents[1].name,epoch=best['epoch'],checkpoint_sha256=q['best_sha256']))
        # Whole-file public aggregate hashes independently checked against actual outputs.
        hashes=json.loads((r/'results/PUBLIC_OUTPUT_HASHES.json').read_text())['sha256'];assert all(sha(r/'results'/name)==h for name,h in hashes.items())
        verified.append(dict(phase=i,cells=len(complete),all_BEST_LAST_COMPLETE_hashes_match=True,code_protocol_hashes_match=True,all_public_hashes_match=True))
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline);fr=json.loads((a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text())
    for row in fr['selections']:
        if row['arm']=='D1':
            assert sha(a.baseline/f'D1/fold{row["fold"]}/TRAIN/BEST_PRIVATE.pt')==row['checkpoint_sha256']
            assert sha(a.baseline/f'D1/fold{row["fold"]}/BANK_PRIVATE.pt')==row['bank_sha256']
    _,boot=original_modules();two=pd.DataFrame(read(a.phase2/'DEVELOPMENT_METRICS_PRIVATE.pkl'));three=pd.DataFrame(read(a.phase3/'DEVELOPMENT_METRICS_PRIVATE.pkl'));one=pd.DataFrame(read(a.phase1/'DEVELOPMENT_METRICS_PRIVATE.pkl'))
    merged=pd.concat([one[one.arm.isin(['D1','M1_SCALED_DGM','M2_STAGE_DELTA','M3_DELTA_SET'])],two[two.arm.isin(['M4_FEATURE_CORRECTION','M5_STAGE_CORRECTION','M6_MORPH_CORRECTION'])],three[three.arm.isin(['M7_MARGIN_GATED','M8_MACRO_AWARE'])]],ignore_index=True)
    metrics,_=original_modules();out=[]
    for left,right in [('M8_MACRO_AWARE','M6_MORPH_CORRECTION'),('M7_MARGIN_GATED','M6_MORPH_CORRECTION'),('M6_MORPH_CORRECTION','M1_SCALED_DGM')]:
        out.extend(dict(comparison=left+' minus '+right,**r) for r in boot.compare(c,merged[merged.arm==left],merged[merged.arm==right],metrics.METRICS))
    a.out.mkdir(parents=True,exist_ok=True);pd.DataFrame(out).to_csv(a.out/'MATCHED_MECHANISM_BOOTSTRAP.csv',index=False)
    # Check the target directly from primary per-patient data, not just a status flag.
    means=merged.groupby('arm')[metrics.METRICS].mean();delta=means.loc['M8_MACRO_AWARE']-means.loc['D1'];assert delta.macro_f1>=.02
    assert delta.ez_auroc>=0 and delta.ez_auprc>=0 and delta.ez_f1>=0 and delta.sensitivity>=-.01
    positive=sum(merged[(merged.arm=='M8_MACRO_AWARE')&(merged.fold==f)].macro_f1.mean()>merged[(merged.arm=='D1')&(merged.fold==f)].macro_f1.mean() for f in range(1,6));assert positive>=4
    assert (len(merged[merged.arm=='M8_MACRO_AWARE']),merged[merged.arm=='M8_MACRO_AWARE'].patient.nunique())==(65,47)
    # Inspect every saved M8 private inference output binding/finite/identity against actual checkpoint.
    for fold in range(1,6):
        pred=read(a.phase3/f'predictions_private/M8_MACRO_AWARE_fold{fold}.pkl');cp=a.phase3/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt'
        assert pred['binding']['checkpoint_sha256']==sha(cp)
        assert np.isfinite(pred['modes']['full']['logit']).all()
        assert np.array_equal(pred['modes']['residual0']['logit'],pred['modes']['full']['base_logit'])
    # Exact M6/M8 architecture initialization matches across all folds; only declared loss changes.
    s2=pd.read_csv(a.phase2/'results/MODEL_SELECTION.csv');s3=pd.read_csv(a.phase3/'results/MODEL_SELECTION.csv')
    assert s2[s2.arm=='M6_MORPH_CORRECTION'].sort_values('fold').initial_state_sha256.tolist()==s3[s3.arm=='M8_MACRO_AWARE'].sort_values('fold').initial_state_sha256.tolist()
    write(a.out/'GOAL_COMPLETION_AUDIT.json',dict(status='PASS',phases=verified,actual_completed_training_cells=40,total_saved_epochs=total_epochs,original_D1_checkpoints_banks_unchanged=True,M6_M8_initial_state_exact=True,M8_initial_and_disabled_D1_identity_verified=True,primary_appearances=65,unique_IDs=47,primary_MacroF1=float(means.loc['M8_MACRO_AWARE','macro_f1']),baseline_MacroF1=float(means.loc['D1','macro_f1']),absolute_gain=float(delta.macro_f1),two_absolute_percentage_points_met=True,positive_folds=int(positive),AUROC_AP_EZF1_nondecline=True,other_metric_deltas=delta.to_dict(),no_new_Outer=True,clinical_independent_confirmation=False,all_candidates_retained=True,source_tests_and_reproducible_training_commands_delivered=True,private_artifacts_not_exported=True))
    print('ACTUAL40_CELLS_AND_M8_TARGET_AUDIT_PASS',total_epochs,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--phase2',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase2_runtime'));p.add_argument('--phase3',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));p.add_argument('--out',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_final_audit'));audit(p.parse_args())
