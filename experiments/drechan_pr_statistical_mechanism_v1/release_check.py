"""Authenticate the aggregate allowlist and schemas BEFORE any extraction."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
COMMIT='49db6da4084635bade1cf64e3da10a5ed890e8fb'
SCIENCE=['PROTOCOL_LOCK.json','PROTOCOL_LOCK.md','mechanism_math.py','trusted_io.py',
         'run_mechanism.py','statistics_report.py','verify_mechanism.py','tests/test_mechanism.py']
ROWS={'representation_alignment.csv':42,'transferable_snr.csv':42,'boundary_drift.csv':42,
 'factorial_contrasts.csv':72,'paired_intervals.csv':168,'negative_controls.csv':2200,
 'negative_control_summary.csv':136,'source_results.csv':168,'patient_distribution_bins.csv':1146,
 'support_and_donor_audit.csv':5,'fold_coverage.csv':5}
JSONS=['INPUT_AUDIT.json','FREEZE_STATUS.json','ASSESS_STATUS.json','REPORT_STATUS.json',
       'INDEPENDENT_VERIFICATION.json','STATUS.json','PUBLIC_PRIVACY_REVIEW.json']


def h(data):return hashlib.sha256(data).hexdigest()


def check(package,expected,source_tree=None):
    package=Path(package);assert re.fullmatch('[a-f0-9]{64}',expected)
    assert package.stat().st_size<20_000_000 and h(package.read_bytes())==expected
    allowed=set(ROWS)|set(JSONS)|{'PACKAGE_MANIFEST.json'}
    with zipfile.ZipFile(package) as z:
        assert len(z.namelist())==len(allowed) and set(z.namelist())==allowed
        assert all(q.file_size<10_000_000 and not q.is_dir() for q in z.infolist())
        data={n:z.read(n) for n in allowed}
    manifest=json.loads(data['PACKAGE_MANIFEST.json'])
    assert manifest['schema']=='PR_STATISTICAL_MECHANISM_V1' and not manifest['private_exported']
    assert manifest['publication_privacy_filter'] and not manifest['strict79_table_published']
    assert set(manifest['files'])==allowed-{'PACKAGE_MANIFEST.json'}
    assert all(h(data[n])==v for n,v in manifest['files'].items())
    assert set(manifest['source'])==set(SCIENCE)
    for n,v in manifest['source'].items():
        if source_tree is None:
            b=subprocess.check_output(['git','show',COMMIT+':experiments/drechan_pr_statistical_mechanism_v1/'+n],cwd=HERE)
        else:b=(Path(source_tree)/n).read_bytes()
        assert h(b)==v,('Source does not match pre-outcome commit',n)
    for n,b in data.items():
        text=b.decode('utf-8')
        assert not re.search(r'[A-Za-z]:[\\/]|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',text),('Private path/contact',n)
    objects={n:json.loads(data[n]) for n in JSONS}
    assert objects['INPUT_AUDIT.json']['ASSESS_unique']==80
    assert objects['INPUT_AUDIT.json']['strict_original_FIT_assessed']==79
    assert objects['INPUT_AUDIT.json']['original_validation_only_assessed']==1
    assert objects['FREEZE_STATUS.json']['source_commit']==COMMIT
    assert objects['FREEZE_STATUS.json']['states']==7070
    assert objects['FREEZE_STATUS.json']['source']==manifest['source']
    assert objects['ASSESS_STATUS.json']['rows']==179360
    proof=objects['INDEPENDENT_VERIFICATION.json']
    assert proof['status']=='PASS' and proof['unique_assessed_patients']==80
    assert proof['directions_reconstructed']==7070 and proof['endpoint_values']==717440
    assert proof['nominal_intervals_verified']==1112 and proof['secondary_family_intervals']==25
    assert objects['STATUS.json']['status']=='COMPLETE_VERIFIED_PR_STATISTICAL_MECHANISM_80'
    privacy=objects['PUBLIC_PRIVACY_REVIEW.json']
    assert privacy['status']=='PUBLIC_COMPLEMENT_PRIVACY_FILTER_PASS'
    assert privacy['all80_primary_unchanged'] and privacy['strict79_remains_trusted_only']
    assert not privacy['exact_strict79_values_or_intervals_exported']
    assert not privacy['scientific_source_changed'] and not privacy['scientific_results_changed']
    tables={}
    forbidden={'patient','patient_id','channel','channel_id','score','prediction','identity','donor_id'}
    for n,count in ROWS.items():
        reader=csv.DictReader(io.StringIO(data[n].decode('utf-8')))
        assert not forbidden&set(reader.fieldnames)
        rows=list(reader);assert len(rows)==count,(n,len(rows),count)
        assert all(None not in r for r in rows);tables[n]=rows
    for n in ['representation_alignment.csv','transferable_snr.csv','boundary_drift.csv','paired_intervals.csv']:
        assert all(int(r['patients'])==80 for r in tables[n])
    cov=tables['fold_coverage.csv']
    assert sum(int(r['ASSESS_patients']) for r in cov)==80
    assert sum(int(r['ASSESS_channels']) for r in cov)==7635
    assert sum(int(r['EZ_channels']) for r in cov)==1743
    assert sum(int(r['NEZ_channels']) for r in cov)==5892
    assert sum(int(r['original_validation_only_assessment']) for r in cov)==1
    primary=[r for r in tables['paired_intervals.csv'] if r['primary']=='True']
    assert len(primary)==1 and primary[0]['target']=='G3_FULL' and primary[0]['metric']=='T'
    family=sum(r['family_size']=='25' for n in ['paired_intervals.csv','factorial_contrasts.csv'] for r in tables[n])
    assert family==25
    result=dict(status='PASS',package_sha256=expected,payloads=len(allowed)-1,source_files=8,
        unique_patients=80,channels=7635,strict_original_FIT=79,original_validation_exception=1,
        source_commit=COMMIT,row_counts=ROWS,verified_before_extraction=True,
        strict79_complementary_means_private_only=True)
    return data,result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--extract',type=Path)
    p.add_argument('--source-tree',type=Path);p.add_argument('--proof',type=Path);a=p.parse_args()
    data,proof=check(a.package,a.sha256,a.source_tree)
    if a.extract:
        assert not a.extract.exists(),'Do not overwrite released evidence'
        a.extract.mkdir(parents=True)
        for n,b in data.items():(a.extract/n).write_bytes(b)
    if a.proof:
        assert not a.proof.exists();a.proof.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,indent=2))
