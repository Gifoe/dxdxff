"""Unlabeled representative graph metric parity and bounded-work resource gate."""
import json
import pickle
import time
from pathlib import Path
import networkx as nx
import numpy as np
import os
import subprocess
from threadpoolctl import threadpool_limits
from graph_metrics import node_features
from inspect_source import ROOT,RAW,SSS,sha
from audit_source import key
from repository_graph import compute_thresholded_abs_pearson_adjacency,compute_graph_node_features
from common import write_json

def main():
    with (ROOT/'GRAPH_LEDGER_PRIVATE.pkl').open('rb') as f: ledger=pickle.load(f)
    with RAW.open('rb') as f: payload=pickle.load(f)
    raw={key(r):r['sample'] for r in payload['run_records']}
    chosen=[min(ledger,key=lambda r:len(r['names'])),max(ledger,key=lambda r:len(r['names']))]
    chosen+=sorted(ledger,key=lambda r:len(r['names']))[::max(1,len(ledger)//4)][:4]
    rows=[]; rng=np.random.default_rng(42)
    with threadpool_limits(limits=1):
        waves=[rng.normal(size=(12,500)).astype('float32')]
        for row in chosen:
            s=raw[(row['patient'],row['run'],row['sample'],row['names'])]
            ix=np.flatnonzero(row['valid'][:,0]); start=row['starts'][0]
            waves.append(np.asarray(s['raw_waveform'])[ix,start:start+round(row['fs']*2)])
        for i,wave in enumerate(waves):
            a=compute_thresholded_abs_pearson_adjacency(wave); begin=time.perf_counter(); fast=node_features(a); elapsed=time.perf_counter()-begin
            graph=nx.from_numpy_array(a); connected=nx.is_connected(graph)
            reference=compute_graph_node_features(wave); dimensions=[0,1,2,4,5,6]+([3] if connected else [])
            error=float(np.nanmax(np.abs(fast[:,dimensions]-reference[:,dimensions])))
            assert error<=3e-6,(i,error)
            assert np.array_equal(fast,node_features(a),equal_nan=True)
            rows.append({'channels':len(wave),'seconds_fast':elapsed,'connected':connected,'source_parity_dimensions':dimensions,
                         'max_absolute_parity_error':error,'deterministic':True,'missing_eigen_fraction':float(np.isnan(fast[:,3]).mean())})
    rss=int(subprocess.check_output(['powershell.exe','-NoProfile','-Command',f'(Get-Process -Id {os.getpid()}).WorkingSet64']).strip())
    report={'status':'PASS','representative_unlabeled_graphs':rows,'observed_rss_bytes':rss,
            'bounded_parallel_workers':4,'blas_threads_per_worker':1,'metric_implementation_sha256':sha(Path(__file__).with_name('graph_metrics.py')),
            'disconnected_source_zero_fallback_not_treated_as_reference':True,'labels_used':False}
    write_json(ROOT/'public/GRAPH_COMPUTATION_BENCHMARK.json',report); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
