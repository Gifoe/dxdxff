"""Aggregate-only reconnaissance; never prints identities or clinical labels."""
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    export = Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    with np.load(export, allow_pickle=True) as d:
        print(json.dumps({'npz_fields': d.files,
                          'metadata_shapes': {k: list(d[k].shape) for k in
                                              ['patient', 'channel', 'center'] if k in d.files}}))
    raw = Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl')
    with raw.open('rb') as f:
        d = pickle.load(f)
    r = d['run_records'][0]
    s = r.get('sample', {})
    print(json.dumps({'payload_keys': list(d), 'record_keys': list(r),
                      'sample_keys': list(s), 'shape': list(np.shape(s.get('raw_waveform'))),
                      'raw_metadata': {k: s.get(k) for k in
                                       ['raw_temporal_sfreq', 'raw_temporal_duration_sec',
                                        'raw_valid_start_sample', 'raw_valid_samples']}}))


if __name__ == '__main__':
    main()
