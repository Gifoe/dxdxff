"""Public-byte/numeric audit; no individual data or model training required."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data=HERE/'results/readout';manifest=json.loads((data/'PUBLIC_TRANSFER_MANIFEST.json').read_text())
    assert manifest['status']=='PASS' and len(manifest['files'])==17
    for name,entry in manifest['files'].items():
        assert (data/name).stat().st_size==entry['bytes'] and sha(data/name)==entry['sha256']
    assert manifest['code_sha256']==sha(HERE/'export_public.py')
    parent=HERE.parent/'m8_physiology_discovery_selection_seed42_v2'
    old=json.loads((parent/'RELEASE_MANIFEST.json').read_text())
    for name,entry in old['files'].items():assert sha(parent/name)==entry['sha256']
    cond=HERE.parent/'m8_rawconditional_feature_selection_seed42_v1/results/dense_reference'
    previous=json.loads((cond/'PUBLIC_TRANSFER_MANIFEST.json').read_text())
    for name,entry in previous['files'].items():assert sha(cond/name)==entry['sha256']
    plan=pd.read_csv(data/'FROZEN_READOUT_PLAN.csv');assert len(plan)==15 and not plan.assessment_used_for_this_choice.any()
    assert plan.k.between(5,9).all()
    maps=pd.read_csv(HERE/'results/ATTRIBUTE_MAP.csv').set_index('feature')
    semantics=[]
    for r in plan.itertuples(index=False):
        names=r.features.split(';');assert len(names)==r.k
        semantics.append(dict(fold=r.fold,screen=r.screen,numeric_group_k=r.k,
                              measurement_attributes=len(set(maps.loc[names].attribute)),
                              coarse_families=len(set(maps.loc[names].coarse_family)),
                              no_input_or_selection_change=True))
    pd.DataFrame(semantics).to_csv(HERE/'results/READOUT_PLAN_ATTRIBUTE_COUNTS.csv',index=False)
    overlap=pd.read_csv(HERE/'results/ATTRIBUTE_OVERLAP_PAIRS.csv')
    nonempty=overlap[(overlap.study=='Raw_conditional') & (overlap.fold_a!=2) & (overlap.fold_b!=2)]
    assert len(nonempty)==18
    nonempty.groupby('level',as_index=False).jaccard.mean().to_csv(HERE/'results/ATTRIBUTE_NONEMPTY_OVERLAP.csv',index=False)
    selection=pd.read_csv(data/'READOUT_SELECTION.csv');assert len(selection)==90 and not selection.alias.any()
    assert selection.epochs.sum()==928
    assert (selection.groupby(['fold','screen','head_seed']).initial.nunique()==1).all()
    replay=pd.read_csv(data/'READOUT_REPLAY_VERIFICATION.csv');assert len(replay)==90 and (replay.logit_replay_max_error==0).all()
    main=pd.read_csv(data/'READOUT_MAIN_RESULTS.csv').set_index('arm')
    seeds=pd.read_csv(data/'READOUT_SEED_RESULTS.csv').set_index(['arm','head_seed'])
    pairs=pd.read_csv(data/'READOUT_PAIRED_INTERVALS.csv')
    metrics=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
    for arm in main.index:
        for m in metrics:assert abs(main.loc[arm,m]-(seeds.loc[(arm,42),m]+seeds.loc[(arm,43),m])/2)<1e-12
    for r in pairs.itertuples(index=False):assert abs(main.loc['SELECTED',r.metric]-main.loc[r.reference,r.metric]-r.delta)<1e-12
    aliases={'Raw-only':'RAW_ONLY','Raw + selected physiology':'SELECTED','Raw + original7':'OLD7'}
    table_metrics=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1'];cells=0
    for line in (HERE/'REPORT.md').read_text(encoding='utf-8').splitlines():
        if not line.startswith('|'):continue
        columns=[v.strip() for v in line.strip('|').split('|')]
        if columns[0] in aliases:
            for metric,value in zip(table_metrics,columns[1:]):
                assert abs(float(value)-main.loc[aliases[columns[0]],metric])<=5.1e-7;cells+=1
    assert cells==24
    utility=pd.read_csv(data/'READOUT_UTILITY_INTERVALS.csv').set_index('reference')
    gate=json.loads((data/'ALLOCATION_GATE.json').read_text())
    seed_pass=all(seeds.loc[('SELECTED',s),m]>seeds.loc[('RAW_ONLY',s),m] for s in [42,43] for m in ['auroc','auprc','macro_f1','f1'])
    assert seed_pass==gate['each_primary_mean_positive_in_both_head_seeds']
    assert (utility.loc['RAW_ONLY','ci_low']>0)==gate['four_endpoint_utility_CI_lower_positive']
    source=pd.read_csv(data/'READOUT_SOURCE_RESULTS.csv').set_index(['arm','center'])
    source_pass=all(source.loc[('SELECTED',c),'auroc']-source.loc[('RAW_ONLY',c),'auroc']>=-.02 for c in ['hup','lzu','multicenter','pediatric'])
    assert source_pass==gate['all_source_AUROC_deltas_at_least_minus002']
    passed=seed_pass and source_pass and utility.loc['RAW_ONLY','ci_low']>0
    assert gate['status']==('PASS' if passed else 'FAIL')
    status=dict(status='PASS',transferred_public_files=17,previous_v2_release_files_unchanged=len(old['files']),
        previous_conditional_public_files_unchanged=len(previous['files']),readout_fits=90,readout_epochs=928,
        exact_sealed_logit_replays=90,independent_patient_metric_rows=1530,
        reported_main_metric_cells_checked=24,allocation_gate=gate['status'],
        new_full_Raw_or_M8_training=0,new_Outer_inference=False,
        stable_minimum_or_reliable_neural_increment_proven=False)
    (HERE/'RELEASE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
