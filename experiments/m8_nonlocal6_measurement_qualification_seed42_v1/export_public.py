"""Only seven whitelisted anonymous aggregate files leave the server."""
import qualify as q

FILES=['QUALIFICATION_STATUS.json','SOURCE_COVERAGE.csv','INNER_NUMERIC_QUALIFICATION.csv',
       'UNIT_STRESS.csv','QUALIFICATION_DECISIONS.csv','INDEPENDENT_QUALIFICATION_VERIFICATION.json']


if __name__=='__main__':
    pub=q.ROOT/'constant_safe/public'
    assert q.jr(pub/'INDEPENDENT_QUALIFICATION_VERIFICATION.json')['status']=='PASS'
    q.write(pub/'PUBLIC_TRANSFER_MANIFEST.json',dict(status='PASS',files={n:dict(bytes=(pub/n).stat().st_size,sha256=q.sha(pub/n)) for n in FILES},
        exporter_sha256=q.sha(q.HERE/'export_public.py'),verifier_sha256=q.sha(q.HERE/'verify_qualification.py'),
        failed_original_gate_sha256=q.sha(q.PUB/'UNIT_GATE_FAILURE_DIAGNOSIS.json'),
        no_private_identity_or_array_outputs=True))
    print('ANONYMOUS_AGGREGATE_EXPORT_READY',len(FILES),flush=True)
