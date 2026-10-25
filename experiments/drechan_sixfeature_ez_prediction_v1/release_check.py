"""Validate the explicit anonymous archive and immutable sources before extraction."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent


def literal_list(file,name):
    for n in ast.parse((HERE/file).read_text(encoding='utf-8')).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name:return ast.literal_eval(n.value)
    raise AssertionError(name)


def verify(path):
    allow=literal_list('report.py','ALLOW');science=literal_list('core.py','SCIENCE')
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())==set(allow)|{'PACKAGE_MANIFEST.json'}
        assert len(z.namelist())==len(set(z.namelist()))
        manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
        assert manifest['schema']=='SIXFEATURE_EZ_PREDICTION_V1' and not manifest['private_exported']
        assert set(manifest['files'])==set(allow) and set(manifest['source'])==set(science)
        for n,h in manifest['files'].items():
            assert hashlib.sha256(z.read(n)).hexdigest()==h,n
            if n.endswith('.csv'):
                rows=list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8'))))
                assert len(rows)>0
                assert not set(rows[0])&{'patient','patient_id','channel','channel_id','subject_id','label','y','probability','path'}
        for n,h in manifest['source'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,('FROZEN_SOURCE_CHANGED',n)
        proof=json.loads(z.read('INDEPENDENT_VERIFICATION.json'))
        assert proof['status']=='PASS' and proof['patient_rows']==1120 and proof['null_patient_rows']==8000
        assert proof['patient_endpoint_values']==72960 and proof['score_cells']==570 and proof['formal_unique_LR_fits']==1410
        assert proof['additional_28family_CIs_verified']==28 and proof['paired_CIs_verified']==112
        assert proof['absolute_and_source_CIs_verified']==560 and not proof['private_exported']
        assert json.loads(z.read('STATUS.json'))['status']=='COMPLETE_VERIFIED_SIXFEATURE_EZ_PREDICTION'
        counts={'main_results.csv':8,'individual_feature_results.csv':12,'all_results.csv':14,
          'pr_improvement.csv':56,'paired_comparisons.csv':56,'fold_results.csv':560,'source_results.csv':448,
          'negative_control.csv':100,'negative_control_summary.csv':8,'coordinate_variation.csv':60,
          'coverage.csv':14,'model_selection.csv':70,'inner_cv.csv':280,'fit_audit.csv':1410}
        for n,k in counts.items():assert len(list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8')))))==k,n
    print('ANONYMOUS_PREDICTION_PACKAGE_PASS',hashlib.sha256(Path(path).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);verify(p.parse_args().package)
