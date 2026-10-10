"""Phase A: aggregate-only inspection of trusted caches; no clinical fields used."""
import gc
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT=Path(r'C:\d2_graphrole_seed42_runtime')
FEATURE=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl')
RAW=Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl')
EXPORT=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
SSS=Path(r'C:\a0_sss_mil_seed42_runtime')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ROOT.mkdir(exist_ok=True); (ROOT/'public').mkdir(exist_ok=True)
    with np.load(EXPORT,allow_pickle=True) as f:
        pid=f['patient'].astype(str); channels=f['channel'].astype(str); sources=f['center'].astype(str)
        names=f['features'].astype(str).tolist()
    assert len(set(pid))==80 and len(set(zip(pid,channels)))==7635 and len(names)==88
    with (SSS/'RAW_LEDGER_PRIVATE.pkl').open('rb') as f: ledger=pickle.load(f)
    assert sha(RAW)==ledger['raw_hash'] and sha(FEATURE)==ledger['feature_hash']
    with FEATURE.open('rb') as f: cache=pickle.load(f)
    records=[r for r in cache['run_records'] if str(r['subject_id']) in set(pid)]
    assert len(records)==256
    stats=[]; samples=[]; fieldnames=set(); samplekeys=set(); settings=[]
    for r in records:
        s=r['sample']; fieldnames.update(r); samplekeys.update(s)
        keys=[k for k in s if 'adj' in k.lower() or 'connect' in k.lower()]
        keys+=['record:'+k for k in r if 'adj' in k.lower() or 'connect' in k.lower()]
        for k in keys:
            v=np.asarray(r[k[7:]] if k.startswith('record:') else s[k])
            if v.ndim<2: continue
            off=v[...,~np.eye(v.shape[-1],dtype=bool)] if v.shape[-2]==v.shape[-1] else v
            stats.append({'field':k,'shape':list(v.shape),'finite':float(np.isfinite(v).mean()),
                          'nonzero':float(np.count_nonzero(off)/off.size),'maxabs':float(np.nanmax(np.abs(v))),
                          'symmetry_error':float(np.nanmax(np.abs(v-v.swapaxes(-1,-2)))) if v.shape[-2]==v.shape[-1] else None,
                          'diagonal_max':float(np.nanmax(np.abs(np.diagonal(v,axis1=-2,axis2=-1)))) if v.shape[-2]==v.shape[-1] else None})
            samples.append(off.reshape(-1)[::max(off.size//2000,1)][:2000])
        for k in cache:
            if any(x in k.lower() for x in ['config','args','version','feature_names']):
                settings.append(k)
    featurekeys=list(cache); del cache,records; gc.collect()
    with RAW.open('rb') as f: raw=pickle.load(f)
    rawrecords=[r for r in raw['run_records'] if str(r['subject_id']) in set(pid)]
    rawstats=[]
    for r in rawrecords:
        s=r['sample']; x=np.asarray(s['raw_waveform']); fs=float(s['raw_temporal_sfreq'])
        rawstats.append([x.shape[0],x.shape[1],fs,int(s['raw_valid_samples'])<x.shape[1]])
    report={'matched_patients':80,'records':len(rawrecords),'canonical_channels':7635,
            'feature_cache_keys':featurekeys,'feature_record_keys':sorted(fieldnames),'feature_sample_keys':sorted(samplekeys),
            'export_feature_names':names,'graph_name_overlap':[n for n in names if any(t in n.lower() for t in ['degree','strength','clustering','eigen','pagerank','kcore','efficien'])],
            'adjacency_fields':sorted(set(v['field'] for v in stats)),'adjacency_arrays':len(stats),
            'zero_adjacency_arrays':sum(v['nonzero']==0 for v in stats),
            'adjacency_shape_examples':[v['shape'] for v in stats[:3]],
            'adjacency_finite_min':min((v['finite'] for v in stats),default=None),
            'adjacency_offdiagonal_nonzero_mean':float(np.mean([v['nonzero'] for v in stats])) if stats else None,
            'adjacency_weight_quantiles':np.quantile(np.concatenate(samples),[0,.01,.25,.5,.75,.99,1]).tolist() if samples else [],
            'adjacency_symmetry_max':max((v['symmetry_error'] for v in stats if v['symmetry_error'] is not None),default=None),
            'adjacency_diagonal_max':max((v['diagonal_max'] for v in stats if v['diagonal_max'] is not None),default=None),
            'raw_sfreq_values':sorted(set(v[2] for v in rawstats)),'raw_channel_range':[min(v[0] for v in rawstats),max(v[0] for v in rawstats)],
            'padded_records':sum(v[3] for v in rawstats),'raw_sha256':ledger['raw_hash'],'feature_cache_sha256':ledger['feature_hash'],
            'export_sha256':sha(EXPORT),'prior_ledger_sha256':sha(SSS/'RAW_LEDGER_PRIVATE.pkl'),
            'independently_verified_onsets':0,'labels_used':False,'outer_labels_read':False}
    (ROOT/'public/GRAPH_SOURCE_INSPECTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
