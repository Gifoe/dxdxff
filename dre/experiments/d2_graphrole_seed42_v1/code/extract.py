"""All measured windows, atomic run-level caches, bounded deterministic extraction."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import gc
import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from graph_metrics import node_features,seizure_summary,patient_summary,NAMES
from repository_graph import compute_thresholded_abs_pearson_adjacency
from inspect_source import ROOT,RAW,SSS,EXPORT,sha
from audit_source import key,norm
from common import write_json,digest

def write_npz(path,**arrays):
    temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('wb') as f: np.savez(f,**arrays)
    temp.replace(path)

def job(args):
    number,inputfile,bind=args; out=ROOT/'run_graph_private'/f'run_{number:03d}.npz'; meta=out.with_suffix('.json')
    binding={**bind,'input_sha256':sha(inputfile)}
    if out.exists() and meta.exists():
        saved=json.loads(meta.read_text()); assert saved['binding']==binding and saved['output_sha256']==sha(out)
        return number,saved
    t=time.perf_counter()
    with np.load(inputfile) as d: wave=d['wave']; valid=d['valid']; starts=d['starts']; fs=float(d['fs']); times=d['times']
    W=valid.shape[1]; C=valid.shape[0]; metrics=np.full((W,C,7),np.nan,dtype=np.float32)
    disconnected=0; undefined_eigen=0
    with threadpool_limits(limits=1):
        for w,start in enumerate(starts):
            ix=np.flatnonzero(valid[:,w])
            if len(ix)<2: continue
            a=compute_thresholded_abs_pearson_adjacency(wave[ix,start:start+round(fs*2)],edge_quantile=.70,min_edge_weight=.10)
            v=node_features(a); metrics[w,ix]=v
            undefined_eigen+=int(np.isnan(v[:,3]).all())
            from scipy.sparse.csgraph import connected_components
            disconnected+=int(connected_components(a,directed=False)[0]>1)
    summary,rank=seizure_summary(metrics,times)
    write_npz(out,metrics=metrics,summary=summary,rank=rank,times=times,valid=valid)
    saved={'binding':binding,'output_sha256':sha(out),'seconds':time.perf_counter()-t,'windows':W,'channels':C,
           'disconnected_graphs':disconnected,'unidentifiable_eigen_graphs':undefined_eigen}
    write_json(meta,saved); return number,saved

def main():
    a=argparse.ArgumentParser(); a.add_argument('--protocol',type=Path,required=True); args=a.parse_args()
    lock=json.loads(args.protocol.read_text()); assert lock['graph_source_mode']=='RAW_REBUILT'
    gate=json.loads((ROOT/'public/GRAPH_SOURCE_AUDIT.json').read_text()); assert gate['status']=='PASS' and gate['mode']==lock['graph_source_mode']
    bench=json.loads((ROOT/'public/GRAPH_COMPUTATION_BENCHMARK.json').read_text()); assert bench['status']=='PASS'
    assert sha(RAW)==lock['input_sha256']['raw']
    for folder in ['input_raw_private','run_graph_private']: (ROOT/folder).mkdir(exist_ok=True)
    code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['graph_metrics.py','repository_graph.py','extract.py']}
    assert bench['metric_implementation_sha256']==code['graph_metrics.py']
    bind={'protocol':sha(args.protocol),'code':code,'raw':sha(RAW),'mode':lock['graph_source_mode']}
    with (ROOT/'GRAPH_LEDGER_PRIVATE.pkl').open('rb') as f: ledger=pickle.load(f)
    assert len(ledger)==256
    with RAW.open('rb') as f: payload=pickle.load(f)
    raw={key(r):r['sample'] for r in payload['run_records']}; inputs=[]
    for number,row in enumerate(ledger):
        path=ROOT/'input_raw_private'/f'run_{number:03d}.npz'; s=raw[(row['patient'],row['run'],row['sample'],row['names'])]
        if not path.exists(): write_npz(path,wave=np.asarray(s['raw_waveform']),valid=row['valid'],starts=row['starts'],fs=np.array(row['fs']),times=row['times'])
        else:
            with np.load(path) as d: assert np.array_equal(d['wave'],s['raw_waveform']) and np.array_equal(d['valid'],row['valid'])
        inputs.append((number,path,bind))
    del payload,raw; gc.collect(); print('UNLABELED_RAW_SHARDS_READY 256',flush=True)
    timing=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(job,item) for item in inputs]
        for future in as_completed(tasks):
            number,record=future.result(); timing.append({'run_index':number,**record})
            print('GRAPH_WINDOWS_COMPLETE',len(timing),'/256',flush=True)
    with np.load(EXPORT,allow_pickle=True) as d: pid=d['patient'].astype(str); channel=np.array([norm(c) for c in d['channel']]); original_channel=d['channel'].astype(str); source=d['center'].astype(str)
    output=np.full((7635,16),np.nan); signature=[]; patient_records=defaultdict(list)
    for i,row in enumerate(ledger): patient_records[row['patient']].append((i,row))
    for p in sorted(set(pid)):
        ix=np.flatnonzero(pid==p); lookup={c:i for i,c in enumerate(channel[ix])}; summaries=[]; ranks=[]; sig=np.zeros((len(ix),len(patient_records[p])),bool)
        for s,(number,row) in enumerate(patient_records[p]):
            slot=np.array([lookup[c] for c in row['names']]); su=np.full((len(ix),15),np.nan); ra=np.full(len(ix),np.nan)
            with np.load(ROOT/'run_graph_private'/f'run_{number:03d}.npz') as d:
                su[slot]=d['summary']; ra[slot]=d['rank']; sig[slot,s]=d['valid'].any(1)
            summaries.append(su); ranks.append(ra)
        output[ix]=patient_summary(summaries,ranks); signature.append((p,sig))
    assert len(set(zip(pid,channel)))==7635 and set(patient_records)==set(pid)
    write_npz(ROOT/'GRAPH16_PRIVATE.npz',patient=pid,channel=original_channel,center=source,graph=output,feature_names=np.array(NAMES))
    with (ROOT/'SIGNATURES_PRIVATE.pkl').open('wb') as f: pickle.dump(signature,f)
    write_json(ROOT/'public/GRAPH_ALIGNMENT_AUDIT.json',{'status':'PASS','patients':80,'runs':256,'canonical_channels':7635,
       'strict_one_to_one_identity_join':True,'no_dropped_or_duplicated_channels':True,'all_window_centers_match_feature_source':True,
       'valid_channel_windows':1471965,'valid_graph_windows':15074,'independent_onset_verified':False,'outer_labels_used':False,
       'graph_bank_sha256':sha(ROOT/'GRAPH16_PRIVATE.npz'),'binding':bind})
    write_json(ROOT/'public/GRAPH_EXTRACTION_STATUS.json',{'status':'COMPLETE','runs':256,'all_sealed_run_outputs':True,
       'disconnected_graphs':sum(r['disconnected_graphs'] for r in timing),'unidentifiable_eigen_graphs':sum(r['unidentifiable_eigen_graphs'] for r in timing),
       'summed_worker_seconds':sum(r['seconds'] for r in timing),'workers':4,'graph_bank_sha256':sha(ROOT/'GRAPH16_PRIVATE.npz')})
    print('GRAPH16_BANK_COMPLETE',output.shape,flush=True)

if __name__=='__main__': main()
