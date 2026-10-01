"""Fail closed on the public audit artifact contract."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=['PROTOCOL_LOCK.json','BASELINE_REPLAY_AUDIT.json','LAYER_MAPPING.md','REPRESENTATION_EXTRACTION_AUDIT.json','DOMAIN_CLASSIFICATION_BY_LAYER.csv','CENTER_CLASSIFICATION_BY_LAYER.csv','SHIFT_SMD_BY_LAYER.csv','SHIFT_COVARIANCE_BY_LAYER.csv','SHIFT_MMD_BY_LAYER.csv','SHIFT_FRECHET_BY_LAYER.csv','CENTER_COMPOSITION_AUDIT.csv','CENTER_CONDITIONED_SHIFT.csv','LOCALIZATION_PERFORMANCE_BY_CENTER.csv','LABEL_DIRECTION_COSINE_BY_LAYER.csv','LINEAR_PROBE_GENERALIZATION.csv','TASK_DIRECTION_SHIFT.csv','CENTER_LABEL_DIRECTION_COSINE.csv','pca_train_test.pdf','pca_center.pdf','pca_label.pdf','FINAL_REPORT.md']
missing=[x for x in REQUIRED if not (ROOT/x).is_file() or (ROOT/x).stat().st_size==0]
if missing:raise RuntimeError(f'missing/empty outputs: {missing}')
if json.loads((ROOT/'BASELINE_REPLAY_AUDIT.json').read_text())['status']!='PASS':raise RuntimeError('baseline gate failed')
e=json.loads((ROOT/'REPRESENTATION_EXTRACTION_AUDIT.json').read_text())
assert (e['train_edfs'],e['test_edfs'],e['train_channel_units'],e['test_channel_units'])==(296,237,21438,16543)
for p in ROOT.glob('*.csv'):
    if pd.read_csv(p).empty:raise RuntimeError(f'empty CSV: {p.name}')
for p in ROOT.glob('*.pdf'):
    if p.read_bytes()[:5]!=b'%PDF-':raise RuntimeError(f'invalid PDF header: {p.name}')
d=pd.read_csv(ROOT/'DOMAIN_CLASSIFICATION_BY_LAYER.csv');assert set(d.layer)==set(['R0','R1','R2','R3','R4','R5']);assert ((d.auroc>=0)&(d.auroc<=1)).all()
if not (ROOT/'FINAL_REPORT.md').read_text(encoding='utf-8').startswith('| Layer |'):raise RuntimeError('required first table is not first')
print(json.dumps({'status':'PASS','required_outputs':len(REQUIRED),'csv_files_validated':len(list(ROOT.glob('*.csv'))),'pdf_files_validated':3},indent=2))
