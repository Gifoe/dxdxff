"""Aggregate role-union diagnostic; no labels returned or predictions made."""
import common as c


def main():
    c.setup();meta,banks=c.io.identity_and_banks(labels=False)
    cohort=set(meta['patient']);unions={k:set() for k in ['fit','validation','test']};rows=[]
    for fold,d in enumerate(banks,1):
        roles={k:set(d['roles'][k]) for k in unions}
        assert all(roles[a].isdisjoint(roles[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
        assert set.union(*roles.values())==cohort
        assert set(d['patient'][d['train']])==roles['fit'] and set(d['patient'][d['val']])==roles['validation']
        for k in unions:unions[k].update(roles[k])
        rows.append(dict(fold=fold,**{k:len(v) for k,v in roles.items()}))
    assert len(cohort)==80 and len(meta['patient'])==7635 and len(unions['test'])==80
    result=dict(status='ROLE_DIAGNOSTIC_COMPLETE',cohort_patients=len(cohort),canonical_channels=len(meta['patient']),
        role_union_counts={k:len(v) for k,v in unions.items()},patients_never_in_any_FIT=len(cohort-unions['fit']),fold_counts=rows,
        FIT_union_identity_sha256=c.digest(sorted(map(str,unions['fit']))),cohort_identity_sha256=c.digest(sorted(map(str,cohort))),
        original_role_registry_sha256=c.io.EXPECTED_ENCODERS,raw_registry_sha256=c.io.EXPECTED_REG,reconstruction_sha256=c.io.EXPECTED_REC,
        all_original_role_identities_unchanged=True,no_labels_returned=True,no_training_prediction_or_threshold_optimization=True,
        legacy_mixed_metadata_physically_unpickled=True)
    destination=c.ROOT/'final_stage_v1/public/FINAL_ROLE_ADMISSION_DIAGNOSTIC.json';c.write(destination,result)
    print('JOINT_RAW_ORIGINAL_ROLE_DIAGNOSTIC',result,flush=True)


if __name__=='__main__':main()
