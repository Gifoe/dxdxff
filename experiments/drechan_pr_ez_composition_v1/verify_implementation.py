"""Read-only source QA; this does NOT attest real experiment completion."""
import json
import io_contract as io

def main():
    proof=io.read(io.HERE/'FROZEN_SOURCE.json')
    assert proof['source_commit']=='b12952d9b8abce3b280dda7fa704671bb78e60fc'
    assert proof['sources']==io.sources(),'Frozen scientific/parent source changed'
    assert proof['legacy_anomaly_sha256']==io.sha(io.PRREF/'results/ANOMALY.csv')
    assert len(io.protocol()['primary_family'])==2 and len(io.protocol()['secondary_family'])==11
    from release_check import COUNTS
    assert sum(COUNTS.values())==28818 and len(io.PAYLOADS)==21
    print(json.dumps(dict(status='IMPLEMENTATION_SOURCE_PARITY_PASS',real_experiment_status='NOT_ATTESTED_BY_THIS_CHECK',scientific_files=len(io.SCIENCE),parent_files=len(io.parent.sources()),source_commit=proof['source_commit'],planned_public_table_rows=sum(COUNTS.values())),indent=2))
if __name__=='__main__':main()
