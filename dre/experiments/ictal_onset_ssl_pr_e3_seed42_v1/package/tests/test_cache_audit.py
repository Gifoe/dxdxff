import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pickle
import numpy as np
import pytest
import json
import audit_export as a


def build_fake(tmp_path):
    feature={'run_records':[],'patient_index':{}}
    raw={'run_records':[]}
    for sid in ('P001','P002'):
        name=[f'LA-001',f'LA-002']
        key={'subject_id':sid,'run_id':'run-1','sample':{
            'sample_id':'seizure-1','source_seizure_id':'s1',
            'seizure_onset_sec':30.0,'start_sec':0.,'end_sec':60.,
            'channel_names_norm':name,'window_relative_centers_sec':np.array([-14.,-10.,0.,4.]),
        }}
        feature['run_records'].append(key)
        sample=dict(key['sample'])
        sample.update(raw_waveform=np.ones((2,15000),dtype='float32'),
                      raw_temporal_sfreq=250.,raw_temporal_duration_sec=60.,
                      raw_valid_start_sample=0,raw_valid_samples=15000)
        raw['run_records'].append({'subject_id':sid,'run_id':'run-1','sample':sample})
        feature['patient_index'][sid]={'canonical_channels':name,'labels':[1,0],'source_center':'test'}
    for name,obj in [('feature',feature),('raw',raw)]:
        with (tmp_path/f'{name}.pkl').open('wb') as f:pickle.dump(obj,f)
    (tmp_path/'folds.json').write_text(json.dumps({'folds':[{'fold_idx':1,'fit_subjects':['P001'], 'validation_subjects':[], 'test_subjects':['P001','P002']}] }))
    return tmp_path/'feature.pkl',tmp_path/'raw.pkl',tmp_path/'folds.json'


def test_trusted_cache_export_never_claims_independent_onset(tmp_path,monkeypatch):
    f,r,m=build_fake(tmp_path)
    monkeypatch.setattr(a,'EXPECTED_PATIENTS',2)
    monkeypatch.setattr(a,'EXPECTED_SEIZURES',2)
    monkeypatch.setattr(a,'EXPECTED_CHANNELS',4)
    monkeypatch.setattr(a,'frozen_folds',lambda path,subjects: [{'fold_idx':1,'fit_subjects':['P001'],'validation_subjects':[],'test_subjects':['P002']}])
    report=a.extract(f,r,m,tmp_path/'export',allow_source_only=True)
    assert report['status']=='PASS_SOURCE_CODE_ALIGNMENT_ONLY'
    assert report['n_unique_channels']==4 and report['n_seizures']==2
    assert report['independent_onset_provenance_verified_records']==0
    manifest=json.loads((tmp_path/'export'/'manifest.json').read_text())
    saved=tmp_path/'export'/manifest['cohort']['P001']['file']
    with np.load(saved,allow_pickle=False) as npz:
        assert npz['pair'].shape==(1,2,2,2500)
        assert npz['present'].all()
        assert np.array_equal(npz['labels_ez'],[1,0])


def test_missing_valid_baseline_fails_closed(tmp_path,monkeypatch):
    f,r,m=build_fake(tmp_path)
    with r.open('rb') as handle:payload=pickle.load(handle)
    payload['run_records'][0]['sample']['raw_valid_start_sample']=6000
    payload['run_records'][0]['sample']['raw_valid_samples']=9000
    with r.open('wb') as handle:pickle.dump(payload,handle)
    monkeypatch.setattr(a,'EXPECTED_PATIENTS',2)
    monkeypatch.setattr(a,'EXPECTED_SEIZURES',2)
    monkeypatch.setattr(a,'frozen_folds',lambda path,subjects: [])
    with pytest.raises(ValueError,match='include padding'):
        a.extract(f,r,m,tmp_path/'export',allow_source_only=True)
