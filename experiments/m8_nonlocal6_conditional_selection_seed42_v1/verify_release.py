"""Local public-byte, report, figure and protected-artifact verification."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jr(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    data=HERE/'results';manifest=jr(data/'PUBLIC_TRANSFER_MANIFEST.json')
    assert manifest['status']=='PASS' and len(manifest['files'])==21
    for name,value in manifest['files'].items():
        assert (data/name).stat().st_size==value['bytes'] and sha(data/name)==value['sha256']
        if name.endswith('.csv'):assert not set(pd.read_csv(data/name,nrows=1).columns)&{'patient','channel','probability','prediction','y','path'}
    for section in ['source','additional_source']:
        for name,digest in manifest[section].items():assert sha(HERE/name)==digest
    report=jr(data/'REPORT_STATUS.json');verification=jr(data/'INDEPENDENT_EXPERIMENT_VERIFICATION.json')
    assert verification['status']=='PASS' and report['actual_search_fits']==verification['search_fits_replayed']==5120
    assert report['actual_assessment_refits']==verification['assessment_refits_replayed']==574
    assert verification['maximum_choice_score_error']==verification['maximum_assessment_probability_error']==0
    assert report['logical_patient_metric_rows']==verification['independent_patient_metric_rows']==13260
    assert report['patient_appearances_per_arm']==255 and report['unique_patients']==79
    assert report['new_encoder_training']==0 and not report['new_Outer_inference']
    assert not report['goal_complete_proven']
    plan=pd.read_csv(data/'FROZEN_SELECTION_PLAN.csv');assert len(plan)==1040 and not plan.chosen_using_HELD.any()
    assert len(pd.read_csv(data/'EXHAUSTIVE_CHOICE_RECORD.csv'))==5120
    eligible=pd.read_csv(data/'OWN_TRAIN_ELIGIBILITY.csv');assert len(eligible)==120 and eligible.eligible.all()
    main=pd.read_csv(data/'MAIN_RESULTS.csv').set_index(['context','arm']);assert len(main)==52
    pairs=pd.read_csv(data/'PAIRED_INTERVALS.csv');assert len(pairs)==171
    for row in pairs.itertuples(index=False):
        delta=main.loc[(row.context,row.arm),row.metric]-main.loc[(row.reference_context,row.reference_arm),row.metric]
        assert abs(delta-row.delta)<=1e-12
    stability=pd.read_csv(data/'SELECTION_STABILITY.csv');freq=pd.read_csv(data/'SELECTION_FREQUENCY.csv')
    selected=plan[plan.arm=='SELECTED']
    for row in stability.itertuples(index=False):
        own=selected[(selected.context==row.context)&((selected.screen==0) if row.scope=='final_paths' else (selected.screen>0))]
        assert len(own)==row.paths and (own.k>0).sum()==row.nonempty_paths and abs(own.k.mean()-row.mean_k)<1e-12
    utility=pd.read_csv(data/'UTILITY_INTERVALS.csv');gate=jr(data/'ALLOCATION_GATE.json')
    u=utility[(utility.context=='RAW_OLD7')&(utility.reference_context=='RAW_OLD7')&(utility.reference_arm=='BASE')].iloc[0]
    s=stability[(stability.context=='RAW_OLD7')&(stability.scope=='final_paths')].iloc[0]
    checks=dict(utility_interval_lower_positive=bool(u.ci_low>0),
        all_four_clinical_deltas_positive=bool(all(main.loc[('RAW_OLD7','SELECTED'),m]>main.loc[('RAW_OLD7','BASE'),m] for m in ['auroc','auprc','macro_f1','f1'])),
        all_known_source_AUROC_deltas_at_least_minus002=all(v>=-.02 for v in gate['source_AUROC_deltas'].values()),
        final_pairwise_Jaccard_at_least_point5=bool(s.mean_pairwise_jaccard>=.5),at_least_four_nonempty_final_sets=bool(s.nonempty_paths>=4))
    assert checks==gate['checks'] and gate['status']==('PASS' if all(checks.values()) else 'FAIL')=='FAIL'
    assert gate['previous32_group_nonlinear_gate_still_FAIL'] and not gate['automatic_full_M8_or_Outer_authorized']
    aliases={'Selected new only':('PHYS','SELECTED'),'Original7':('PHYS_OLD7','BASE'),
        'Original7 + selected new':('PHYS_OLD7','SELECTED'),'Frozen Raw':('RAW','BASE'),
        'Frozen Raw + selected new':('RAW','SELECTED'),'Frozen Raw + original7':('RAW_OLD7','BASE'),
        'Frozen Raw + original7 + selected new':('RAW_OLD7','SELECTED')}
    metrics=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1'];checked=matched=0
    contexts={'New only':'PHYS','Original7 + new':'PHYS_OLD7','Raw + new':'RAW','Raw + original7 + new':'RAW_OLD7'}
    for line in (HERE/'REPORT.md').read_text(encoding='utf-8').splitlines():
        if not line.startswith('|'):continue
        columns=[c.strip() for c in line.strip('|').split('|')]
        if columns[0] in aliases and len(columns)==9:
            for m,value in zip(metrics,columns[1:]):assert abs(float(value)-main.loc[aliases[columns[0]],m])<=5.1e-7;checked+=1
        if columns[0] in contexts and len(columns)==4 and '[' in columns[1]:
            c=contexts[columns[0]]
            for ref,value in zip(['BEST_ALTERNATIVE','RANDOM_A','RANDOM_B'],columns[1:]):
                numbers=np.array([float(v) for v in re.findall(r'-?\d+\.\d+',value.replace('−','-'))])
                row=pairs[(pairs.context==c)&(pairs.reference_context==c)&(pairs.reference_arm==ref)&(pairs.metric=='auroc')].iloc[0]
                np.testing.assert_allclose(numbers,[row.delta,row.ci_low,row.ci_high],atol=5.1e-7,rtol=0);matched+=3
    assert checked==56 and matched==36
    curve=pd.read_csv(HERE/'figures/SOURCE_CARDINALITY.csv');assert len(curve)==28
    for row in curve.itertuples(index=False):
        for m in ['auroc','macro_f1']:assert abs(getattr(row,m)-main.loc[(row.context,f'K{row.k}'),m])<=1e-12
    source=pd.read_csv(HERE/'figures/SOURCE_FINAL_FREQUENCY.csv');assert len(source)==24
    assert len(pd.read_csv(HERE/'figures/SOURCE_INCREMENT.csv'))==8
    qa=jr(HERE/'FIGURE_QA.json');assert qa['status']=='PASS' and qa['final_size_visual_inspection'].startswith('PASS')
    for ext in ['svg','pdf','png','tiff']:assert (HERE/f'figures/01_nonlocal6_cardinality_stability.{ext}').stat().st_size>1000
    old=HERE.parent/'m8_physiology_discovery_selection_seed42_v2';previous=jr(old/'RELEASE_MANIFEST.json')
    for name,value in previous['files'].items():assert sha(old/name)==value['sha256']
    protected=0
    for folder in ['m8_rawconditional_feature_selection_seed42_v1/results/dense_reference',
                   'm8_conditional_readout_validation_seed42_v1/results/readout',
                   'm8_nonlocal8_eligibility_audit_seed42_v1/results',
                   'm8_nonlocal6_measurement_qualification_seed42_v1/results/qualified']:
        root=HERE.parent/folder
        for name,value in jr(root/'PUBLIC_TRANSFER_MANIFEST.json')['files'].items():assert sha(root/name)==value['sha256'];protected+=1
    result=dict(status='PASS',anonymous_public_files=21,actual_search_fits=5120,actual_assessment_refits=574,
        complete_search_and_head_replay=True,logical_patient_rows_dual_checked=13260,main_report_metric_cells=checked,
        equal_k_report_numeric_cells=matched,figure_cardinality_rows=28,figure_final_frequency_cells=24,
        previous_v2_files_unchanged=len(previous['files']),previous_supplement_public_files_unchanged=protected,
        allocation_gate='FAIL',new_encoder_or_Outer_runs=0,new_stable_minimum_or_neural_gain_proven=False)
    (HERE/'RELEASE_STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
