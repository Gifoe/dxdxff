"""Authenticate the exact user-authorized deidentified selected-case export."""
import argparse,csv,hashlib,io,json,re,zipfile
from pathlib import Path
FILES={'ADMISSION.json','CASE_METADATA.csv','CASE_METRICS.csv','CHANNEL_SCORES_ANONYMOUS.csv','RECALL_AT_K.csv','SELECTION_AUDIT.json','CASE_SELECTION_REASON.json','ALL80_CHANGE_DISTRIBUTION.csv','COHORT_AGGREGATES.csv','INDEPENDENT_VERIFICATION.json','PROVENANCE.json','STATUS.json'}
CASES={'CASE-P01','CASE-N01'}
def check(path):
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())=={'public/'+x for x in FILES}|{'PUBLIC_MANIFEST.json'} and len(z.namelist())==len(FILES)+1
        manifest=json.loads(z.read('PUBLIC_MANIFEST.json'));assert set(manifest['files'])==FILES
        assert manifest['real_patient_identifiers_included'] is False and manifest['original_contact_names_included'] is False and manifest['user_authorized_deidentified_channel_scores'] is True
        for name,meta in manifest['files'].items():
            b=z.read('public/'+name);assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256']
            if name.endswith('.csv'):
                reader=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
                assert not {'patient','patient_id','pid','private_patient_id','original_channel_name','channel_name','birth_date','name','identity'}&set(reader.fieldnames)
                for row in reader:
                    if 'case_id' in row:assert row['case_id'] in CASES
                    if 'channel_id' in row:assert re.fullmatch(r'CH\d{4}',row['channel_id'])
                    if 'center' in row:assert row['center'] in ['HUP','LZU','Multicenter','Pediatric']
        a=json.loads(z.read('public/ADMISSION.json'));assert a['status']=='PASS' and a['score_cells']==40 and a['new_model_fits']==a['new_model_forward_passes']==0
        iv=json.loads(z.read('public/INDEPENDENT_VERIFICATION.json'))
        assert iv['status']=='PASS' and iv['patient_condition_rows']==640 and iv['patient_change_rows']==400 and iv['endpoint_values']==3200
        assert iv['original_parent_patient_metric_replays']==640 and iv['selected_case_count']==2 and iv['case_metric_rows']==50
        assert iv['independent_metric_formulas'] and iv['independent_selector_agreement'] and iv['parent_hashes_unchanged']
        assert json.loads(z.read('public/STATUS.json'))['status']=='COMPLETE_VERIFIED_FROZEN_OUTER_REPRESENTATIVE_CASES'
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--extract',type=Path);a=p.parse_args();m=check(a.package)
    print('DEIDENTIFIED_CASE_PACKAGE_PASS',hashlib.sha256(a.package.read_bytes()).hexdigest())
    if a.extract:
        assert not a.extract.exists();a.extract.mkdir(parents=True)
        with zipfile.ZipFile(a.package) as z:
            for n in FILES:(a.extract/n).write_bytes(z.read('public/'+n))
            (a.extract/'PUBLIC_MANIFEST.json').write_bytes(z.read('PUBLIC_MANIFEST.json'))
