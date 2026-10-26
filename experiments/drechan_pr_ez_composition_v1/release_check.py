"""Authenticate exact anonymous package before allowlist extraction."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import io_contract as contract
from core import ARMS,COORDS,SOURCES,FRACTIONS

COUNTS={'E1_CLASS_LOSS':650,'E1_GAIN':650,'E1_H':325,'E2_LOSS':3900,'E2_GAIN':300,'E2_H':1300,'E2_ANOMALY':1040,'E2_SUPPORT':520,'REFERENCE_COMPOSITION':80,'COMPOSITION_SLOPE':20,'PRIMARY':2,'SECONDARY':11,'H_PERMUTATION':20000,'H_PERMUTATION_SUMMARY':20}
TAGS={'scope','coordinate','fraction','arm','group','endpoint','identity','unit','status'}

def check(package,expected):
    assert contract.sha(package)==expected,'Whole-archive SHA mismatch'
    with zipfile.ZipFile(package) as z:
        assert len(z.namelist())==len(set(z.namelist()))==len(contract.PAYLOADS)+1
        assert set(z.namelist())==set(contract.PAYLOADS+['MANIFEST.json'])
        assert all('/' not in n and '\\' not in n for n in z.namelist())
        assert sum(p.file_size for p in z.infolist())<15000000
        payload={n:z.read(n) for n in z.namelist()}
    manifest=json.loads(payload['MANIFEST.json']);assert manifest['schema']=='PR_EZ_COMPOSITION_PUBLIC_V1'
    assert manifest['sources']==contract.sources() and set(manifest['files'])==set(contract.PAYLOADS)
    for n,h in manifest['files'].items():assert hashlib.sha256(payload[n]).hexdigest()==h,n
    status=json.loads(payload['STATUS.json']);verification=json.loads(payload['INDEPENDENT_VERIFICATION.json']);coverage=json.loads(payload['COVERAGE.json'])
    assert status['status']=='COMPLETE_VERIFIED_PR_EZ_COMPOSITION_80' and verification['status']=='PASS'
    assert status['patients']==coverage['patients']==verification['patients']==80 and status['channels']==coverage['channels']==verification['channels']==7635
    assert status['primary_family']==2 and status['secondary_family']==11 and status['row_counts']==COUNTS
    legacy=contract.PRREF/'results/ANOMALY.csv';assert payload['LEGACY_ANOMALY.csv']==legacy.read_bytes()
    assert status['legacy_anomaly_sha256']==hashlib.sha256(payload['LEGACY_ANOMALY.csv']).hexdigest()
    allowed_numeric={'mean','median','ci_low','ci_high','family_low','family_high','patients','assessed_patients','empty_bootstrap_draws','permutation','H','observed_H','null_mean','null_low','null_high','conditional_plus_one_upper','conditional_plus_one_lower','permutations'}
    for name,n in COUNTS.items():
        reader=csv.DictReader(io.StringIO(payload[name+'.csv'].decode('utf-8')));assert set(reader.fieldnames)<=TAGS|allowed_numeric
        rows=list(reader);assert len(rows)==n
        for r in rows:
            assert r['scope'] in ['ALL']+SOURCES
            if 'coordinate' in r:assert r['coordinate'] in ['ALL12']+COORDS
            if 'arm' in r:assert r['arm'] in ARMS
            if 'fraction' in r:assert float(r['fraction']) in FRACTIONS
            if 'group' in r:assert r['group'] in ['EZ','NEZ','ALL']
            if 'status' in r:assert r['status'] in ['estimable','not_estimable','single_patient_numerics_withheld']
            if 'unit' in r:assert r['unit']=='CRPS_per10pp_EZ'
            for key,value in r.items():
                if key not in TAGS and value:assert math.isfinite(float(value))
            if r.get('status')=='single_patient_numerics_withheld':assert r['mean']==r['ci_low']==r['ci_high']==''
    # No arbitrary private payload may appear;all public fields are source-bound.
    return payload

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--extract',type=Path);a=p.parse_args();payload=check(a.package,a.sha256)
    if a.extract:
        assert not a.extract.exists();a.extract.mkdir(parents=True)
        for n,b in payload.items():(a.extract/n).write_bytes(b)
    print('PUBLIC_PACKAGE_PASS',len(payload)-1,'anonymous_payloads',a.sha256)
if __name__=='__main__':main()
