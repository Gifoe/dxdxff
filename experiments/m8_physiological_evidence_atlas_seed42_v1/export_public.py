"""Only audited anonymous feature/source statistics may be transferred."""
import atlas as q

FILES=['ATLAS_STATUS.json','OLD_EFFECT_STATISTICAL_AUDIT.json','NEW12_PATIENT_ASSOCIATIONS.csv',
       'NEW12_SOURCE_ASSOCIATIONS.csv','NEW12_REPEAT_SEIZURE_SUPPORT.csv','PRIMARY172_ASSOCIATION_ATLAS.csv',
       'INDEPENDENT_ATLAS_VERIFICATION.json']


if __name__=='__main__':
    root=q.ROOT/'rolecount_checked';pub=root/'public'
    assert q.jr(pub/'INDEPENDENT_ATLAS_VERIFICATION.json')['status']=='PASS'
    for name in FILES:
        if name.endswith('.csv'):assert not set(q.pd.read_csv(pub/name,nrows=1).columns)&{'patient','channel','seizure','y','path','probability','prediction'}
    binding=q.jr(root/'INPUT_FREEZE_PRIVATE.json')
    q.write(pub/'PUBLIC_TRANSFER_MANIFEST.json',dict(status='PASS',files={name:dict(bytes=(pub/name).stat().st_size,sha256=q.sha(pub/name)) for name in FILES},
        source=binding['code'],admission_source=binding['admission_guard_amendment'],
        additional_source={n:q.sha(q.HERE/n) for n in ['verify_atlas.py','export_public.py']},
        no_private_identity_or_effect_rows_transferred=True))
    print('ANONYMOUS_EVIDENCE_ATLAS_EXPORT_READY',len(FILES),flush=True)
