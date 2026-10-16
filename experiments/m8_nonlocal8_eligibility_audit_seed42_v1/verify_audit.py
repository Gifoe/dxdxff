"""Reaggregate private audit arrays independently; never load outcomes/models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit
from nonlocal_helpers import NAMES


def main():
    root=audit.ROOT;pub=audit.PUB
    status=json.loads((pub/'AUDIT_STATUS.json').read_text())
    assert status['status']=='COMPLETE_UNLABELED_NONLOCAL8_ELIGIBILITY_AUDIT'
    value=audit.read(root/'RECORD_DIAGNOSTICS_PRIVATE.pkl')
    assert audit.sha(root/'RECORD_DIAGNOSTICS_PRIVATE.pkl')==status['private_diagnostic_sha256']
    assert value['binding']==json.loads((root/'INPUT_FREEZE_PRIVATE.json').read_text())
    for name,h in value['binding']['source'].items():assert audit.sha(audit.HERE/name)==h
    records=value['records'];assert len(records)==2048
    coverage=value['coverage'];assert len(coverage)==256
    assert len(set(r['patient'] for r in records))==80
    frame=pd.DataFrame(coverage)
    assert frame.record_channel_units.sum()==24995 and frame.valid_Raw_channel_windows.sum()==1471965
    assert frame.valid_Raw_record_windows.sum()==15074 and frame.native_geometry_available.all() and frame.uniform_channel_mask.all()
    formula=pd.read_csv(pub/'FORMULA_MASK_SUMMARY.csv').query("scope == 'overall'").set_index('feature')
    checked=0
    for name in NAMES:
        own=[r for r in records if r['feature']==name];assert len(own)==256
        assert sum(r['units'] for r in own)==24995
        exact=sum(np.array_equal(r['stored'],r['cached_formula']) for r in own)
        maximum=max(float(np.max(np.abs(r['stored'].astype(float)-r['cached_formula'].astype(float)))) for r in own)
        assert exact==formula.loc[name,'formula_exact_records']==256 and maximum==formula.loc[name,'formula_max_abs']==0
        assert all(r['broadcast_exact'] for r in own)
        for mode,column in [('CACHE_STORED','stored'),('CACHE_RAW_VALID_MASK','cached_Raw_mask'),
                            ('CACHE_NATIVE_VALID_MASK','cached_native_mask'),('RAW250_VALID','Raw250_mask')]:
            a=np.concatenate([r[column] for r in own]);assert a.size==24995
            public=pd.read_csv(pub/'DESCRIPTOR_DISTRIBUTIONS.csv')
            row=public[(public['mode']==mode)&(public.center=='ALL')&(public.feature==name)].iloc[0]
            finite=a[np.isfinite(a)]
            assert len(finite)==row.finite_units==24995
            for key,actual in [('min',finite.min()),('q25',np.quantile(finite,.25)),('median',np.median(finite)),
                               ('q75',np.quantile(finite,.75)),('max',finite.max()),('mean',finite.mean())]:
                assert np.isclose(actual,row[key],rtol=1e-12,atol=1e-30), (name,mode,key)
            checked+=6
    perm=pd.DataFrame(value['permutation']);assert len(perm)==4096
    public_perm=pd.read_csv(pub/'PERMUTATION_EQUIVARIANCE.csv').query("scope == 'overall'").set_index(['mode','feature'])
    for (mode,name),g in perm.groupby(['mode','feature']):
        public=public_perm.loc[(mode,name)]
        assert g.units.sum()==public.units==24995
        assert g.permutation_violating_units.sum()==public.permutation_violating_units
        assert g.permutation_max_abs.max()==public.permutation_max_abs
    decisions=pd.read_csv(pub/'ELIGIBILITY_DECISIONS.csv').set_index('feature')
    for name in NAMES:
        raw=[r['Raw250_mask'] for r in records if r['feature']==name]
        computable=all(np.isfinite(a).all() for a in raw)
        invariant=public_perm.loc[('RAW250_VALID',name),'permutation_violating_units']==0
        assert decisions.loc[name,'Raw250_valid_mask_scalar_computable']==computable
        assert decisions.loc[name,'Raw250_channel_order_equivariant']==invariant
        assert decisions.loc[name,'eligible_for_later_distinct_candidate_admission']==(computable and invariant)
    assert decisions.eligible_for_later_distinct_candidate_admission.sum()==6
    assert not decisions.biological_ineffectiveness_claim.any() and not decisions.training_or_feature_pool_change_authorized.any()
    # Byte identities and patient-role arrays, not inferred from an old status.
    assert audit.sha(audit.CACHE)==value['binding']['feature_cache_sha256']
    assert audit.sha(audit.BASE/'RECONSTRUCTION_PRIVATE.pkl')==value['binding']['reconstruction_sha256']
    rec=audit.read(audit.BASE/'RECONSTRUCTION_PRIVATE.pkl')
    for name,h in value['binding']['split_array_sha256'].items():assert audit.array_sha(rec['meta'][name])==h
    audit.write(pub/'INDEPENDENT_AUDIT_VERIFICATION.json',dict(status='PASS',
        private_feature_record_rows_reaggregated=2048,permutation_rows_reaggregated=4096,
        distribution_summary_cells_checked=checked,all256_record_identities_accounted=True,
        all80_patients_and7635_original_channels_retained=True,original_split_arrays_and_source_cache_unchanged=True,
        six_distinct_Raw250_scalar_operators_computable=True,two_rank_operators_fail_order_equivariance=True,
        source_replay_not_cache_creation_or_clinical_provenance=True,
        prediction_or_classification_metrics_read=False,model_fits=0,
        source_sha256=audit.sha(Path(__file__))))
    print('INDEPENDENT_NONLOCAL_AUDIT_REAGGREGATION_PASS',checked,flush=True)


if __name__=='__main__':
    main()
