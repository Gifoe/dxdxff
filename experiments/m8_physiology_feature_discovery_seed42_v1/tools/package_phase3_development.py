"""Only finished anonymous old7 development/admission artifacts, not model data."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
runtime=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
old=runtime/'old7_matched_M8/public';new=runtime/'integrated_M8_v2/public'
assert json.loads((old/'STATUS.json').read_text())['status']=='ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE'
assert json.loads((new/'REAL_FIT_ADMISSION.json').read_text())['status']=='PASS'
names=['STATUS.json','MODEL_SELECTION.csv','TRAINING_HISTORY.csv','ADMISSION.json','DEVELOPMENT_SUMMARY.csv',
    'DEVELOPMENT_BY_FOLD.csv','DEVELOPMENT_BY_CENTER.csv','METRIC_COVERAGE.csv','PAIRED_CLUSTER_INTERVALS.csv']
items=[(old/n,'old7_development/'+n) for n in names]+[(new/n,'integrated_admission/'+n) for n in ['REAL_FIT_ADMISSION.json','WRONG_PAIR_COVERAGE.csv']]
items += [(runtime/'public/WRONG_PAIR_AXIS_REGRESSION.json','integrated_admission/WRONG_PAIR_AXIS_REGRESSION.json')]
manifest={}
for p,name in items:
    if p.suffix=='.csv':
        with p.open(newline='',encoding='utf-8') as f:columns=next(csv.reader(f))
        assert not set(columns)&{'patient','channel','path','y','label','prediction','probability','D1_checkpoint','M8_checkpoint','bank','validation_path'}
    manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
target=runtime/'phase3_development_public_20261016_01.zip';assert not target.exists()
with ZipFile(target,'x',compression=ZIP_DEFLATED) as z:
    for p,name in items:z.write(p,'phase3_results/'+name)
    z.writestr('phase3_results/PUBLIC_SHA256.json',json.dumps(manifest,indent=2,sort_keys=True))
print('PHASE3_ANONYMOUS_DEVELOPMENT_ADMISSION_ONLY_PACKAGE_PASS',len(items),target.stat().st_size)
