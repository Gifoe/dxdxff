"""Explicit anonymous source-audit allowlist; no raw/candidate/private archive."""
from pathlib import Path
import audit

FILES=['AUDIT_STATUS.json','CACHE_CONFIG_METADATA.json','FORMULA_MASK_SUMMARY.csv',
       'DESCRIPTOR_DISTRIBUTIONS.csv','PERMUTATION_EQUIVARIANCE.csv','LATENCY_TIE_DIAGNOSTICS.csv',
       'ALIGNMENT_COVERAGE.csv','ELIGIBILITY_DECISIONS.csv','INDEPENDENT_AUDIT_VERIFICATION.json']


if __name__=='__main__':
    import json
    pub=audit.PUB
    assert json.loads((pub/'INDEPENDENT_AUDIT_VERIFICATION.json').read_text())['status']=='PASS'
    audit.write(pub/'PUBLIC_TRANSFER_MANIFEST.json',dict(status='PASS',source_sha256=audit.sha(Path(__file__)),
        private_arrays_identifiers_and_logs_exported=False,
        files={n:dict(bytes=(pub/n).stat().st_size,sha256=audit.sha(pub/n)) for n in FILES}))
    print('NONLOCAL8_PUBLIC_ALLOWLIST_READY',len(FILES),flush=True)
