"""Check authenticated anonymous package before explicit allowlist extraction."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent


def verify(path):
    tree=ast.parse((HERE/'run_analysis.py').read_text())
    vals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
          and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['SCIENCE','ALLOW']}
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())==set(vals['ALLOW'])|{'PACKAGE_MANIFEST.json'}
        m=json.loads(z.read('PACKAGE_MANIFEST.json'));assert m['schema']=='SIXFEATURE_EZ_ASSOCIATION_V1'
        assert not m['private_artifacts_exported'] and set(m['files'])==set(vals['ALLOW'])
        for n,h in m['files'].items():
            assert hashlib.sha256(z.read(n)).hexdigest()==h
            if n.endswith('.csv'):
                rows=list(csv.DictReader(io.StringIO(z.read(n).decode())))
                assert not set(rows[0])&{'patient','patient_id','channel','channel_id','subject_id','probability','y','label'}
        assert set(m['source'])==set(vals['SCIENCE'])
        for n,h in m['source'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,('SOURCE_CHANGED',n)
        q=json.loads(z.read('INDEPENDENT_VERIFICATION.json'))
        assert q['status']=='PASS' and q['patient_effects_independently_verified']==960
        assert q['source_coordinate_rows']==48 and q['independently_checked_permuted_effects']==9600000
        assert q['primary_Wilcoxon_and_BH_tests_checked']==12 and q['patient_bootstrap_intervals_checked']==60
        assert q['models_trained']==0 and not q['individual_effects_exported']
        assert json.loads(z.read('STATUS.json'))['status']=='COMPLETE_INDEPENDENTLY_VERIFIED_SIXFEATURE_EZ_ASSOCIATION'
        for n,k in {'feature_association.csv':12,'source_association.csv':48,'patient_effect_summary.csv':12,
                    'statistical_tests.csv':12,'patient_effect_histogram.csv':480,'permutation_control.csv':12}.items():
            assert len(list(csv.DictReader(io.StringIO(z.read(n).decode()))))==k,n
    print('ANONYMOUS_ASSOCIATION_PACKAGE_PASS',hashlib.sha256(Path(path).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True,type=Path)
    verify(parser.parse_args().package)
