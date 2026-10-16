"""Anonymous export, report and evidence-ledger integrity; no private reader."""
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
    assert manifest['status']=='PASS' and len(manifest['files'])==7
    for name,value in manifest['files'].items():
        assert (data/name).stat().st_size==value['bytes'] and sha(data/name)==value['sha256']
        if name.endswith('.csv'):assert not set(pd.read_csv(data/name,nrows=1).columns)&{'patient','channel','seizure','y','probability','prediction','path'}
    for section in ['source','admission_source','additional_source']:
        for name,digest in manifest[section].items():assert sha(HERE/name)==digest
    old=jr(data/'OLD_EFFECT_STATISTICAL_AUDIT.json');status=jr(data/'ATLAS_STATUS.json');verified=jr(data/'INDEPENDENT_ATLAS_VERIFICATION.json')
    assert old['status']==verified['status']=='PASS' and old['effect_rows_reaggregated']==800 and old['source_rows_reaggregated']==3200
    assert old['numeric_or_flag_cells_checked']==10400 and old['maximum_old_numeric_error']<=1e-12
    assert verified['new_patient_effects_independently_replayed']==612 and verified['new_record_effects_independently_replayed']==1296
    assert verified['maximum_effect_error']<=1e-12 and status['model_fits']==0 and not status['new_Outer_access']
    effects=pd.read_csv(data/'NEW12_PATIENT_ASSOCIATIONS.csv').set_index('coordinate');repeat=pd.read_csv(data/'NEW12_REPEAT_SEIZURE_SUPPORT.csv').set_index('coordinate')
    assert len(effects)==12 and effects.association_support.sum()==10 and effects.patients.min()==50 and effects.patients.max()==51
    assert repeat.record_analogue_applicable.sum()==8 and repeat.repeated_direction_support.sum()==3
    assert len(pd.read_csv(data/'NEW12_SOURCE_ASSOCIATIONS.csv'))==48 and len(pd.read_csv(data/'PRIMARY172_ASSOCIATION_ATLAS.csv'))==172
    aliases={}
    for group,name in [('early_partial_HG_80_125_slope','HG slope'),('early_LL_slope','LL slope'),
        ('baseline_crossing_partial_HG_80_125','HG crossing'),('baseline_crossing_LL','LL crossing'),
        ('post_top20_partial_HG_80_125','HG top20%'),('post_top20_LL','LL top20%')]:
        for suffix,label in ([('observed_position','position'),('crossing_rate','rate')] if 'crossing' in group else [('seizure_mean','mean'),('seizure_SD','SD')]):aliases[name+' '+label]=group+'__'+suffix
    cells=repetition_cells=0
    for line in (HERE/'REPORT.md').read_text(encoding='utf-8').splitlines():
        if not line.startswith('|'):continue
        col=[c.strip() for c in line.strip('|').split('|')]
        if col[0] not in aliases:continue
        name=aliases[col[0]]
        if '[' in col[3]:
            row=effects.loc[name];assert int(col[1])==row.patients
            ci=[float(n) for n in re.findall(r'-?\d+\.\d+',col[3].replace('−','-'))]
            np.testing.assert_allclose([float(col[2].replace('−','-')),*ci],[row.mean_effect,row.ci_low,row.ci_high],atol=5.1e-7,rtol=0)
            np.testing.assert_allclose(float(col[4]),row.BH_q_new12,rtol=5e-6,atol=0)
            assert (col[5]=='Yes')==row.association_support;cells+=6
        else:
            row=repeat.loc[name];assert row.record_analogue_applicable
            assert list(map(int,col[1:4]))==[row.repeated_finite_seizure_patients,row.finite_record_effects,row.zero_record_effects]
            assert abs(float(col[4])-row.median_inclusive_directed_agreement)<=5.1e-7
            assert (col[5]=='Yes')==row.repeated_direction_support;repetition_cells+=5
    assert cells==72 and repetition_cells==40
    ledger=pd.read_csv(data/'FEATURE38_EVIDENCE_LEDGER.csv');resolution=pd.read_csv(data/'HISTORICAL28_CURRENT_RESOLUTION.csv');ledgerstatus=jr(data/'FEATURE_EVIDENCE_LEDGER_STATUS.json')
    assert len(ledger)==ledger.feature.nunique()==38 and len(resolution)==28
    assert not ledger.joint38_group_search_executed.any() and not ledger.source_failure_means_biological_ineffectiveness.any()
    assert (ledger.stable_predictive_adoption=='NOT_ESTABLISHED').all() and not resolution.biological_ineffectiveness_proven.any()
    for entry in ledgerstatus['source_files'].values():assert sha(HERE.parent/entry['path'])==entry['sha256']
    assert len(resolution[resolution.current_status=='NOT_AVAILABLE_AS_TRUSTWORTHY_LF_MEASUREMENT'])==2
    assert len(resolution[resolution.current_status=='UNCHANGED_OPERATOR_EXCLUDED'])==2
    assert ledgerstatus['joint38_or76_coordinate_model_trained']==False and ledgerstatus['model_or_new_Outer_evaluations']==0
    assert ledgerstatus['original_primary_operational_support_endpoints']==109
    parent=HERE.parent/'m8_physiology_discovery_selection_seed42_v2'
    catalogue=pd.read_csv(parent/'FEATURE_CATALOGUE.csv').set_index('feature')
    biology=pd.read_csv(parent/'results/PHYSIOLOGICAL_EFFECTS.csv');biology=biology[biology.outer_fold==1]
    selected=pd.read_csv(parent/'results/FROZEN_FEATURE_SELECTION.csv');selected=selected[selected.arm=='SELECTED']
    cond=pd.read_csv(HERE.parent/'m8_rawconditional_feature_selection_seed42_v1/results/dense_reference/FROZEN_INTERNAL_SELECTION.csv');cond=cond[cond.recipe=='SELECTED']
    red=pd.read_csv(parent/'results/FIT_REDUNDANCY.csv');red=red[red.outer_fold==1]
    freq=pd.read_csv(HERE.parent/'m8_nonlocal6_conditional_selection_seed42_v1/results/SELECTION_FREQUENCY.csv')
    for row in ledger.itertuples(index=False):
        if row.feature in catalogue.index:
            assert row.definition==catalogue.loc[row.feature,'definition']
            own=biology[biology.feature==row.feature];assert row.primary_association_endpoints==5
            assert row.association_support_endpoints==own.operationally_reproducible.sum()
            last=own[own.phase=='late_minus_pre'].iloc[0]
            np.testing.assert_allclose([row.fixed_late_minus_pre_effect,row.fixed_late_minus_pre_ci_low,row.fixed_late_minus_pre_ci_high],
                [last.mean_effect,last.ci_low,last.ci_high],atol=1e-12,rtol=0)
            assert row.D1_final_selected_paths==sum(row.feature in str(v).split(';') for v in selected.features.fillna(''))
            assert row.Raw32_conditional_final_selected_paths==sum(row.feature in str(v).split(';') for v in cond.features.fillna(''))
            correlation=red[(red.feature_a==row.feature)|(red.feature_b==row.feature)]
            assert abs(row.primary_FIT_max_abs_patient_median_spearman_to_other_window_group-correlation.median_spearman.abs().max())<1e-12
        else:
            own=effects[effects.group==row.feature];rr=repeat.loc[own.index]
            assert row.primary_association_endpoints==2 and row.association_support_endpoints==own.association_support.sum()
            assert row.record_repetition_applicable_endpoints==rr.record_analogue_applicable.sum()
            assert row.record_repeated_direction_supported_endpoints==rr.repeated_direction_support.sum()
            for context in ['PHYS','PHYS_OLD7','RAW','RAW_OLD7']:
                ff=freq[(freq.context==context)&(freq.scope=='final_paths')&(freq.feature==row.feature)].iloc[0]
                assert getattr(row,context+'_final_selected_paths')==ff.selected_paths and getattr(row,context+'_final_paths')==ff.paths
    figure=pd.read_csv(HERE/'figures/SOURCE_PATIENT_EFFECTS.csv').set_index('coordinate')
    for name,row in figure.iterrows():
        for field in ['patients','mean_effect','ci_low','ci_high']:assert abs(row[field]-effects.loc[name,field])<1e-12
    assert len(pd.read_csv(HERE/'figures/SOURCE_CENTER_EFFECTS.csv'))==48
    qa=jr(HERE/'FIGURE_QA.json');assert qa['status']=='PASS' and qa['final_size_visual_inspection'].startswith('PASS')
    for ext in ['svg','pdf','png','tiff']:assert (HERE/f'figures/01_record_physiological_associations.{ext}').stat().st_size>1000
    prior=jr(parent/'RELEASE_MANIFEST.json')
    for name,entry in prior['files'].items():assert sha(parent/name)==entry['sha256']
    protected=0
    for folder in ['m8_rawconditional_feature_selection_seed42_v1/results/dense_reference',
        'm8_conditional_readout_validation_seed42_v1/results/readout','m8_nonlocal8_eligibility_audit_seed42_v1/results',
        'm8_nonlocal6_measurement_qualification_seed42_v1/results/qualified','m8_nonlocal6_conditional_selection_seed42_v1/results']:
        root=HERE.parent/folder
        for name,entry in jr(root/'PUBLIC_TRANSFER_MANIFEST.json')['files'].items():assert sha(root/name)==entry['sha256'];protected+=1
    for folder in ['m8_conditional_readout_validation_seed42_v1/results/readout','m8_nonlocal6_conditional_selection_seed42_v1/results']:
        assert jr(HERE.parent/folder/'ALLOCATION_GATE.json')['status']=='FAIL'
    result=dict(status='PASS',verified_server_aggregate_files=7,reported_association_numeric_flag_cells=cells,
        reported_recurrence_numeric_flag_cells=repetition_cells,group_evidence_ledger_rows=38,historical_definition_resolution_rows=28,
        new_patient_effects_replayed=612,new_record_effects_replayed=1296,old_effect_rows_audited=800,old_source_rows_audited=3200,
        previous_v2_files_unchanged=len(prior['files']),previous_supplement_public_files_unchanged=protected,
        failed_predictive_gates_unchanged=True,new_classifier_or_Outer_runs=0,stable_minimum_or_Raw_gain_proven=False)
    (HERE/'RELEASE_STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
