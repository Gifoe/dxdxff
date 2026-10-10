"""Validate only compact public artifacts before committing delivery."""
import csv
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    expected_audit=['D0_D2_REPRODUCTION.json','RAW_ALIGNMENT_AUDIT.json','RAW_COVERAGE_BY_CENTER.csv',
        'RAW_VALIDITY_AUDIT.csv','ONSET_PROVENANCE_AUDIT.md','RAW_FEATURE_ALIGNMENT_AUDIT.json',
        'WINDOW_SAMPLING_AUDIT.json','RAW_NORMALIZATION_AUDIT.json','MODEL_PARAMETER_AUDIT.json',
        'FUSION_INITIALIZATION_PARITY.json','MIL_MASKING_AUDIT.json']
    expected_results=['TRAINING_HISTORY.csv','VALIDATION_SUMMARY.csv','VALIDATION_BY_FOLD.csv',
        'VALIDATION_BY_CENTER.csv','PAIRED_BOOTSTRAP.csv','FUSION_GATE_AUDIT.csv',
        'RAW_INFORMATION_COMPLEMENTARITY.csv','RANKING_CHANGE_AUDIT.csv','ERROR_CORRECTION_AUDIT.csv',
        'EFFICIENCY_AUDIT.csv','VALIDATION_GATE.json','RUN_STATUS.json']
    for name in expected_audit:
        assert (root/'audit'/name).is_file(),name
    for name in expected_results:
        assert (root/'results'/name).is_file(),name
    for name in ['README.md','SOURCE_AUDIT.md','PROTOCOL_LOCK.json','IMPLEMENTATION_AUDIT.md','FINAL_REPORT.md']:
        assert (root/name).is_file(),name
    status=json.loads((root/'results/RUN_STATUS.json').read_text())
    assert status['status']=='COMPLETE' and status['registered_runs']==15 and not status['outer_test_accessed']
    forbidden={'patient','patient_id','subject_id','channel','channel_id','y','y_true','y_nez','score_nez','logit'}
    csv_count=0
    for path in root.rglob('*.csv'):
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            assert not forbidden & set(reader.fieldnames),path.name
            list(reader)
        csv_count+=1
    with (root/'results/MODEL_SELECTION.csv').open(encoding='utf-8',newline='') as f:
        selection=list(csv.DictReader(f))
    assert len(selection)==15
    assert {(int(r['fold']),r['method']) for r in selection}=={(f,m) for f in range(1,6) for m in ['S0','S1','S2']}
    with (root/'results/PAIRED_BOOTSTRAP.csv').open(encoding='utf-8',newline='') as f:
        boot=list(csv.DictReader(f))
    assert len(boot)==55 and all(int(r['draws'])==10000 and int(r['unique_patient_ids'])==47 for r in boot)
    bad_extensions={'.pt','.pth','.npz','.npy','.pkl','.ckpt','.log','.err','.safetensors'}
    assert not [p for p in root.rglob('*') if p.is_file() and p.suffix in bad_extensions]
    print(f'PUBLIC_DELIVERY_PASS: 15 runs, 55 paired metric contrasts, {csv_count} aggregate CSVs; no private tensor/score artifact')


if __name__=='__main__':
    main()
