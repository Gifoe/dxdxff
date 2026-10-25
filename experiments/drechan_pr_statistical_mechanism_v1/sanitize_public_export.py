"""Publication-only privacy filter; NEVER changes frozen scientific calculations.

Publishing exact80 and79 means exposes the omitted person's endpoint by subtraction.
The strict79 calculation remains trusted-side; only its qualitative conclusion and
verification commitment are exported. All80 primary and negative-control rows stay.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

OMIT='strict_fit_sensitivity.csv'
REVIEW='PUBLIC_PRIVACY_REVIEW.json'


def sha(b):return hashlib.sha256(b).hexdigest()


def sanitize(source,destination):
    source=Path(source);destination=Path(destination);assert not destination.exists()
    with zipfile.ZipFile(source) as z:
        names=z.namelist();assert len(names)==len(set(names))==19
        data={n:z.read(n) for n in names}
    prior=json.loads(data['PACKAGE_MANIFEST.json'])
    assert prior['schema']=='PR_STATISTICAL_MECHANISM_V1' and not prior['private_exported']
    assert set(data)==set(prior['files'])|{'PACKAGE_MANIFEST.json'}
    assert all(sha(data[n])==h for n,h in prior['files'].items())
    checks=json.loads(data['INDEPENDENT_VERIFICATION.json'])
    assert checks['status']=='PASS' and checks['strict79_sensitivity_intervals']==32
    rows=list(csv.DictReader(io.StringIO(data[OMIT].decode('utf-8'))));assert len(rows)==32
    result=next(r for r in rows if r['target']=='G3_FULL' and r['control']=='G0_GLOBAL' and r['metric']=='T')
    assert float(result['effect'])<0 and float(result['ci_high'])<0
    review=dict(status='PUBLIC_COMPLEMENT_PRIVACY_FILTER_PASS',all80_primary_unchanged=True,
        strict79_remains_trusted_only=True,strict79_intervals_independently_verified=32,
        strict79_qualitative_conclusion='NO_SUPPORTED_IMPROVEMENT_IN_TRANSFERABLE_T',
        exact_strict79_values_or_intervals_exported=False,
        reason='Exact80 and79 means allow reconstruction of one patient endpoint by subtraction',
        removed_public_file=OMIT,private_table_sha256=prior['files'][OMIT],
        original_private_intermediate_archive_sha256=sha(source.read_bytes()),
        scientific_source_changed=False,scientific_results_changed=False)
    original_manifest_sha=sha(data.pop('PACKAGE_MANIFEST.json'));data.pop(OMIT)
    data[REVIEW]=(json.dumps(review,indent=2)+'\n').encode('utf-8')
    final=dict(schema=prior['schema'],source=prior['source'],files={n:sha(b) for n,b in sorted(data.items())},
        private_exported=False,publication_privacy_filter=True,original_manifest_sha256=original_manifest_sha,
        strict79_table_published=False)
    data['PACKAGE_MANIFEST.json']=(json.dumps(final,indent=2)+'\n').encode('utf-8')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(data.items()):
            entry=zipfile.ZipInfo(n,date_time=(2026,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(entry,b)
    print('PRIVACY_SAFE_ALL80_PUBLIC_PACKAGE',sha(destination.read_bytes()),destination.stat().st_size,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();sanitize(a.input,a.output)
