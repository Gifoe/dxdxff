"""Join existing public evidence; no re-selection or individual-data access."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results'
OLD=ROOT/'m8_physiology_discovery_selection_seed42_v2'
COND=ROOT/'m8_rawconditional_feature_selection_seed42_v1/results/dense_reference'
NEW=ROOT/'m8_nonlocal6_conditional_selection_seed42_v1/results'
SIX=['early_partial_HG_80_125_slope','early_LL_slope','baseline_crossing_partial_HG_80_125',
     'baseline_crossing_LL','post_top20_partial_HG_80_125','post_top20_LL']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def counts(frame,column):
    return {name:sum(name in str(v).split(';') for v in frame[column].fillna('')) for name in catalog_names}


def main():
    sources={
        'catalogue':OLD/'FEATURE_CATALOGUE.csv','attributes':ROOT/'m8_conditional_readout_validation_seed42_v1/results/ATTRIBUTE_MAP.csv',
        'old_biology':OLD/'results/PHYSIOLOGICAL_EFFECTS.csv','old_selection':OLD/'results/FROZEN_FEATURE_SELECTION.csv',
        'conditional_selection':COND/'FROZEN_INTERNAL_SELECTION.csv','redundancy':OLD/'results/FIT_REDUNDANCY.csv',
        'new_effects':OUT/'NEW12_PATIENT_ASSOCIATIONS.csv','new_recurrence':OUT/'NEW12_REPEAT_SEIZURE_SUPPORT.csv',
        'new_selection_frequency':NEW/'SELECTION_FREQUENCY.csv','historical28':OLD/'results/HISTORICAL28_ELIGIBILITY.csv'}
    catalogue=pd.read_csv(sources['catalogue']);attributes=pd.read_csv(sources['attributes']).set_index('feature')
    global catalog_names
    catalog_names=catalogue.feature.tolist();assert len(catalog_names)==32
    biology=pd.read_csv(sources['old_biology']);biology=biology[biology.outer_fold==1]
    selected=pd.read_csv(sources['old_selection']);selected=selected[selected.arm=='SELECTED']
    conditional=pd.read_csv(sources['conditional_selection']);conditional=conditional[conditional.recipe=='SELECTED']
    a=counts(selected,'features');b=counts(conditional,'features');red=pd.read_csv(sources['redundancy']);red=red[red.outer_fold==1]
    new_effect=pd.read_csv(sources['new_effects']);recurrence=pd.read_csv(sources['new_recurrence']);frequency=pd.read_csv(sources['new_selection_frequency'])
    rows=[]
    for row in catalogue.itertuples(index=False):
        g=biology[biology.feature==row.feature];last=g[g.phase=='late_minus_pre'].iloc[0]
        correlation=red[(red.feature_a==row.feature)|(red.feature_b==row.feature)]
        rows.append(dict(feature=row.feature,measurement_kind='WINDOW_BASELINE_SUBTRACTED_SEIZURE_SUMMARY',
            definition=row.definition,reason_to_consider=row.rationale_and_boundary,
            attribute=attributes.loc[row.feature,'attribute'],coarse_family=attributes.loc[row.feature,'coarse_family'],
            source_available=True,coordinates_per_group=2,primary_association_endpoints=5,
            association_support_endpoints=int(g.operationally_reproducible.sum()),association_support_rule='original160-test_operational_rule',
            fixed_late_minus_pre_effect=last.mean_effect,fixed_late_minus_pre_ci_low=last.ci_low,fixed_late_minus_pre_ci_high=last.ci_high,
            D1_final_selected_paths=a[row.feature],D1_final_paths=5,Raw32_conditional_final_selected_paths=b[row.feature],Raw32_conditional_final_paths=5,
            primary_FIT_max_abs_patient_median_spearman_to_other_window_group=float(correlation.median_spearman.abs().max()),
            individual_conditional_irreplaceability='NOT_ESTABLISHED',stable_predictive_adoption='NOT_ESTABLISHED',
            actual_predictive_scope='prior32-group_LR_and_matched_neural_comparisons;separate_Raw32_screen_and_failed_readout_transfer',
            joint38_group_search_executed=False,source_failure_means_biological_ineffectiveness=False))
    reasons={
        'early_partial_HG_80_125_slope':'Retained-HG early recruitment trend; not a clinical onset estimate',
        'early_LL_slope':'Early roughness/amplitude recruitment trend',
        'baseline_crossing_partial_HG_80_125':'Observed retained-HG baseline-threshold crossing position and event rate; censoring explicit',
        'baseline_crossing_LL':'Observed LL baseline-threshold crossing position and event rate; not clinical onset',
        'post_top20_partial_HG_80_125':'Upper-quintile post-event retained-HG concentration',
        'post_top20_LL':'Upper-quintile post-event roughness/amplitude concentration'}
    for name in SIX:
        g=new_effect[new_effect.group==name];r=recurrence[recurrence.coordinate.isin(g.coordinate)]
        row=dict(feature=name,measurement_kind='QUALIFIED_RECORD_LEVEL_CANONICAL_CHANNEL_AGGREGATE',
            definition='Observed position+crossing rate' if name.startswith('baseline_crossing') else 'equal-seizure mean+population SD',
            reason_to_consider=reasons[name],attribute=name,coarse_family='record_temporal_recruitment' if 'slope' in name or 'crossing' in name else 'record_upper_tail_concentration',
            source_available=True,coordinates_per_group=2,primary_association_endpoints=2,association_support_endpoints=int(g.association_support.sum()),
            association_support_rule='new12-test_association_rule;not_directly_equivalent_to_old_seizure_reproducibility_rule',
            record_repetition_applicable_endpoints=int(r.record_analogue_applicable.sum()),record_repeated_direction_supported_endpoints=int(r.repeated_direction_support.sum()),
            primary_FIT_max_abs_patient_median_spearman_to_other_window_group=np.nan,
            individual_conditional_irreplaceability='NOT_ESTABLISHED',stable_predictive_adoption='NOT_ESTABLISHED',
            actual_predictive_scope='qualified6-group_exhaustive_ridge_screen;failed_allocation_gate;no_new_neural_or_Outer',
            joint38_group_search_executed=False,source_failure_means_biological_ineffectiveness=False)
        for context in ['PHYS','PHYS_OLD7','RAW','RAW_OLD7']:
            q=frequency[(frequency.feature==name)&(frequency.context==context)&(frequency.scope=='final_paths')].iloc[0]
            row[context+'_final_selected_paths']=int(q.selected_paths);row[context+'_final_paths']=int(q.paths)
        rows.append(row)
    ledger=pd.DataFrame(rows);assert len(ledger)==38 and ledger.feature.nunique()==38
    ledger.to_csv(OUT/'FEATURE38_EVIDENCE_LEDGER.csv',index=False)
    mapping={'log_bp_alpha':'partial_alpha_10_13_legacy','log_bp_beta':'log_bp_beta','log_bp_low_gamma':'log_bp_low_gamma',
        'log_bp_high_gamma':'partial_high_gamma_80_125','log_total_power':'log_total_power_10_125_legacy','rms':'rms_legacy',
        'variance':'variance','line_length_per_sec':'line_length_per_sec','spectral_entropy':'spectral_entropy_10_125',
        'hjorth_mobility':'hjorth_mobility_legacy','hjorth_complexity':'hjorth_complexity_legacy',
        'early_high_gamma_slope':SIX[0],'early_line_length_slope':SIX[1],'onset_latency_high_gamma':SIX[2],
        'onset_latency_line_length':SIX[3],'high_gamma_top20pct_mean':SIX[4],'line_length_top20pct_mean':SIX[5]}
    graphs=['degree_norm','strength_norm','clustering_coeff','eigenvector_centrality','pagerank','kcore_norm','local_efficiency']
    historical=[]
    for h in pd.read_csv(sources['historical28']).itertuples(index=False):
        name=h.feature
        if name in ['log_bp_delta','log_bp_theta']:
            status='NOT_AVAILABLE_AS_TRUSTWORTHY_LF_MEASUREMENT';candidate='';reason='all1471965_valid_cached_values_zero;LF_source_gate_failed'
        elif name.startswith('onset_rank_'):
            status='UNCHANGED_OPERATOR_EXCLUDED';candidate='';reason='tied_latencies_get_channel-order-dependent_ranks;not_biological_ineffectiveness'
        elif name in graphs:
            status='LEGACY_SOURCE_NOT_EQUIVALENT;DISTINCT_RAW_REBUILT_OPERATOR_TESTED';candidate='rebuilt_'+name;reason='source-mismatched_cached_graph_not_reused;verified_rebuilt_graph_in32'
        elif name in ['early_high_gamma_slope','early_line_length_slope','onset_latency_high_gamma','onset_latency_line_length','high_gamma_top20pct_mean','line_length_top20pct_mean']:
            status='DISTINCT_RAW250_RECORD_OPERATOR_QUALIFIED_AND_SCREENED';candidate=mapping[name];reason='not_window_DELTA;Raw_mask_and_censoring_definition_explicit;screen_failed_increment_gate'
        else:status='MATCHED_RETAINED_BAND_WINDOW_OPERATOR_TESTED';candidate=mapping[name];reason='preserve_band_and_numeric_definition;not_full_alpha_or_full_high_gamma'
        historical.append(dict(historical_name=name,current_status=status,distinct_tested_candidate=candidate,reason=reason,
            unchanged_cached_source_claimed_equivalent=False,biological_ineffectiveness_proven=False))
    pd.DataFrame(historical).to_csv(OUT/'HISTORICAL28_CURRENT_RESOLUTION.csv',index=False)
    summary=dict(status='COMPLETE_PUBLIC_EVIDENCE_JOIN',qualified_groups=38,numerical_coordinates_if_concatenated=76,
        joint38_or76_coordinate_model_trained=False,original_window_groups=32,distinct_record_groups=6,historical_definitions_resolved=28,
        original_primary_operational_support_endpoints=int(biology.operationally_reproducible.sum()),
        new_association_support_endpoints=int(new_effect.association_support.sum()),new_repeated_direction_support_endpoints=int(recurrence.repeated_direction_support.sum()),
        association_rules_differ_and_counts_are_not_a_cross_family_ranking=True,model_or_new_Outer_evaluations=0,
        stable_minimum_or_Raw_gain_proven=False,source_files={k:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for k,p in sources.items()})
    (OUT/'FEATURE_EVIDENCE_LEDGER_STATUS.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_files'},indent=2))


if __name__=='__main__':main()
