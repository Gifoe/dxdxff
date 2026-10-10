"""Gate every CACHE graph against uniformly reconstructed measured connectivity."""
import gc
import json
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from inspect_source import ROOT,FEATURE,RAW,EXPORT,SSS,sha
from repository_graph import compute_thresholded_abs_pearson_adjacency

def norm(c): return ''.join(str(c).upper().split()).replace('-','').replace('_','')
def key(r): return (str(r['subject_id']),str(r['run_id']),str(r['sample'].get('sample_id',r['run_id'])),tuple(norm(c) for c in r['channel_names_norm']))
def main():
    with (SSS/'RAW_LEDGER_PRIVATE.pkl').open('rb') as f: ledger=pickle.load(f)
    assert sha(RAW)==ledger['raw_hash'] and sha(FEATURE)==ledger['feature_hash']
    with np.load(EXPORT,allow_pickle=True) as f:
        pid=f['patient'].astype(str); channel=np.array([norm(c) for c in f['channel']]); names=f['features'].astype(str)
    with FEATURE.open('rb') as f: payload=pickle.load(f)
    ff={key(r):{'a':np.asarray(r['sample'].get('window_adjacency',[])),
         'features_shape':np.shape(r['sample']['window_features']),
         'times':np.asarray(r['sample']['window_relative_centers_sec'])} for r in payload['run_records'] if str(r['subject_id']) in set(pid)}
    del payload; gc.collect()
    with RAW.open('rb') as f: payload=pickle.load(f)
    rr={key(r):r['sample'] for r in payload['run_records'] if str(r['subject_id']) in set(pid)}
    counts=defaultdict(int); worst=0.; shapes=True; finite=True; nonzero=[]; parity=True
    window_records=[]; union=set()
    for row in ledger['records']:
        k=(row['patient'],row['run'],row['sample'],row['names']); assert k in ff and k in rr
        s=rr[k]; wave=np.asarray(s['raw_waveform']); a=ff[k]['a']; t=np.asarray(row['times']); C=len(row['names']); W=len(t)
        assert np.array_equal(t,ff[k]['times']) or np.allclose(t,ff[k]['times'],atol=1e-6,rtol=0)
        assert ff[k]['features_shape'][:2]==(W,C)
        good_shape=a.shape==(W,C,C); shapes&=good_shape; counts['records']+=1
        assert row['valid'].shape==(C,W)
        record_valid=[]
        for w,start in enumerate(row['starts']):
            valid=row['valid'][:,w]; ix=np.flatnonzero(valid); counts['candidate_channel_windows']+=C
            if len(ix)<2: counts['invalid_channel_windows']+=C; record_valid.append(False); continue
            rawwindow=wave[ix,start:start+round(2*row['fs'])]
            assert np.isfinite(rawwindow).all() and (np.ptp(rawwindow,axis=1)>0).all()
            rebuilt=compute_thresholded_abs_pearson_adjacency(rawwindow,edge_quantile=.70,min_edge_weight=.10)
            counts['valid_graph_windows']+=1; counts['valid_channel_windows']+=len(ix)
            counts['invalid_channel_windows']+=C-len(ix); record_valid.append(True)
            if good_shape:
                cached=a[w][np.ix_(ix,ix)]; off=cached[~np.eye(len(ix),dtype=bool)]
                ok=np.isfinite(cached).all() and np.allclose(cached,cached.T,atol=1e-7,rtol=0) and np.all(np.diag(cached)==0) and (cached>=0).all() and (cached<=1.000001).all()
                finite&=bool(ok); nonzero.append(float(np.count_nonzero(off)/off.size)); counts['zero_placeholder_graphs']+=int(not np.any(off))
                drift=float(np.max(np.abs(cached-rebuilt))) if ok else float('inf'); worst=max(worst,drift); parity&=drift<=2e-6
            else: parity=False
        union.update((row['patient'],c) for c in row['names'])
        window_records.append({**row,'graph_valid':np.array(record_valid)})
        if counts['records']%32==0: print('GRAPH_SOURCE_AUDITED',counts['records'],'/256',flush=True)
    assert len(ff)==len(rr)==counts['records']==256 and union==set(zip(pid,channel))
    coverage=counts['valid_channel_windows']/counts['candidate_channel_windows']
    cache_pass=shapes and finite and parity and counts['zero_placeholder_graphs']==0 and coverage>=.95
    mode='CACHE' if cache_pass else ('RAW_REBUILT' if coverage>=.95 else 'BLOCKED')
    report={'status':'PASS' if mode!='BLOCKED' else 'GRAPH_SOURCE_INTEGRITY_BLOCKED','mode':mode,
       'patients':80,'canonical_channels':7635,**counts,'channel_window_coverage':coverage,
       'cache_shape_valid':bool(shapes),'cache_numerical_integrity':bool(finite),'all_measured_graph_parity':bool(parity),
       'max_cache_raw_adjacency_drift':worst if np.isfinite(worst) else None,
       'cache_offdiagonal_nonzero_mean':float(np.mean(nonzero)) if nonzero else None,
       'construction':'absolute Pearson float32; positive-edge quantile .70; minweight .10; original single-edge fallback',
       'uniform_semantics_all_centers':bool(cache_pass),'raw_sha256':ledger['raw_hash'],'feature_cache_sha256':ledger['feature_hash'],
       'graph_implementation_sha256':sha(Path(__file__).with_name('repository_graph.py')),'ledger_sha256':sha(SSS/'RAW_LEDGER_PRIVATE.pkl'),
       'independent_onsets_verified':0,'no_clinical_fields_used':True,'outer_test_labels_used':False}
    for filename in ['GRAPH_SOURCE_AUDIT.json','GRAPH_SOURCE_MODE.json']:
        (ROOT/'public'/filename).write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'GRAPH_LEDGER_PRIVATE.pkl').open('wb') as f: pickle.dump(window_records,f)
    print(json.dumps(report),flush=True)
    assert mode!='BLOCKED'

if __name__=='__main__': main()
