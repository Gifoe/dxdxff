"""Authenticate anonymous package and schemas BEFORE allowlist extraction."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import contract

def check(path,expected):
    assert contract.sha(path)==expected,'Whole package SHA mismatch'
    with zipfile.ZipFile(path) as z:
        names=z.namelist();assert len(names)==len(set(names))==len(contract.PAYLOADS)+1
        assert set(names)==set(contract.PAYLOADS+['MANIFEST.json'])
        assert all('/' not in n and '\\' not in n for n in names)
        assert sum(p.file_size for p in z.infolist())<10000000
        payload={n:z.read(n) for n in names};manifest=json.loads(payload['MANIFEST.json'])
        assert manifest['schema']=='PR_REFERENCE_TRANSFER_PUBLIC_V1' and manifest['sources']==contract.sources()
        assert set(manifest['files'])==set(contract.PAYLOADS)
        for n,h in manifest['files'].items():assert hashlib.sha256(payload[n]).hexdigest()==h
        status=json.loads(payload['STATUS.json']);ver=json.loads(payload['INDEPENDENT_VERIFICATION.json']);coverage=json.loads(payload['COVERAGE.json'])
        assert status['status']=='COMPLETE_VERIFIED_PR_REFERENCE_TRANSFER_80' and ver['status']=='PASS'
        assert coverage['patients']==ver['patients']==80 and coverage['channels']==ver['channels']==7635
        assert ver['family_intervals_checked']==6 and ver['crps_patient_arm_coordinate_values']==4800
        from math_core import SOURCES,COORDS,ARMS
        schemas={
            'BACKGROUND':['scope','coordinate','arm','mean','median','ci_low','ci_high','patients','positive_fraction'],
            'CONTRASTS':['scope','coordinate','contrast','primary','mean','median','ci_low','ci_high','patients','positive_fraction','family_low','family_high'],
            'ANOMALY':['scope','coordinate','endpoint','mean','median','ci_low','ci_high','patients','positive_fraction','family_low','family_high'],
            'SUPPORT':['scope','coordinate','endpoint','mean','median','ci_low','ci_high','patients','positive_fraction'],
            'PERMUTATION':['scope','permutation','gap'],
            'PERMUTATION_SUMMARY':['scope','observed','null_mean','null_low','null_high','conditional_plus_one_upper','permutations','patients']}
        for name,columns in schemas.items():
            reader=csv.DictReader(io.StringIO(payload[name+'.csv'].decode('utf-8')));assert reader.fieldnames==columns
            rows=list(reader);assert len(rows)==status['row_counts'][name]
            for row in rows:
                assert row['scope'] in ['ALL']+SOURCES
                if 'coordinate' in row:assert row['coordinate'] in ['ALL12']+COORDS
                if 'arm' in row:assert row['arm'] in ARMS
                for key,value in row.items():
                    if key not in ['scope','coordinate','arm','contrast','primary','endpoint'] and value:
                        import math
                        assert math.isfinite(float(value))
        # Fixed aggregate JSON fields; free-form individual payloads are rejected.
        allowed={
          'COVERAGE':{'input_hashes','bank_hashes','patients','channels','EZ','NEZ','seizures','valid_windows','ASSESS_unique','ASSESS_appearances','legacy_internal_unique','legacy_internal_appearances','strict_original_FIT_assessed','original_validation_only_assessed','new_assignment_sha256','source_patients','single_seizure_patients','primary_patients','primary_channels','reference_repeats'},
          'ADMISSION':{'status','full14_operational_parity','all80_channel_support','all_sources_in_each_TRAIN','source_TRAIN_patient_counts','no_new_outcomes_computed','sources','source_commit'},
          'FREEZE':{'status','references_sha256','patients','partitions','directional_reference_sets','queried_channel_appearances','source_commit','sources'},
          'SEAL':{'status','sha256','patients','crps_patient_arm_coordinate_values','permutation_patient_values','sources'},
          'INDEPENDENT_VERIFICATION':{'status','patients','channels','partitions','crps_patient_arm_coordinate_values','endpoint_values_checked','aggregate_interval_rows_checked','family_intervals_checked','permutation_aggregate_values_checked','max_endpoint_absolute_error','absolute_tolerance','relative_tolerance','frozen_sources_unchanged','source_commit'},
          'STATUS':{'status','patients','channels','repetitions','permutations','secondary_family','row_counts','source_commit'}}
        for name,keys in allowed.items():assert set(json.loads(payload[name+'.json']))==keys,name
    return payload

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--extract',type=Path);a=p.parse_args()
    payload=check(a.package,a.sha256)
    if a.extract:
        assert not a.extract.exists();a.extract.mkdir(parents=True)
        for name,value in payload.items():(a.extract/name).write_bytes(value)
    print('PUBLIC_PACKAGE_PASS',len(payload)-1,'anonymous_payloads',a.sha256)
if __name__=='__main__':main()
