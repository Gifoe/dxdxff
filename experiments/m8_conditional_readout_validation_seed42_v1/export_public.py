"""Export hashes of an explicit anonymous-result allowlist after full replay."""
import platform
from pathlib import Path
import numpy as np
import torch
import run_readouts as run
io=run.io

FILES=[
    'FROZEN_READOUT_PLAN.csv','READOUT_SELECTION.csv','READOUT_STATUS.json',
    'READOUT_REPLAY_VERIFICATION.csv','REPLAY_STATUS.json','READOUT_SEED_RESULTS.csv',
    'READOUT_MAIN_RESULTS.csv','READOUT_FOLD_RESULTS.csv','READOUT_SOURCE_RESULTS.csv',
    'READOUT_UNIQUE_ID_BALANCED_RESULTS.csv','READOUT_PAIRED_INTERVALS.csv',
    'READOUT_UTILITY_INTERVALS.csv','READOUT_HEAD_SEED_STABILITY.csv',
    'ALLOCATION_GATE.json','REPORT_STATUS.json','REPLAY_DIAGNOSIS.json','ENVIRONMENT.json',
]


if __name__=='__main__':
    pub=run.PUB
    assert io.jr(pub/'REPORT_STATUS.json')['status']=='COMPLETE_VERIFIED_BOUNDED_INTERNAL_READOUT_TRANSFER'
    assert io.jr(pub/'REPLAY_STATUS.json')['status']=='PASS'
    seal=io.jr(run.ROOT/'READOUT_SCORE_SEAL_PRIVATE.json')
    assert io.bindings()==seal['binding']['parent_binding']['original_scientific_binding']
    for name,h in seal['files'].items():assert io.sha(run.ROOT/name)==h
    dep=io.jr(run.ROOT/'READOUT_ENVIRONMENT_PRIVATE.json')
    io.write(pub/'ENVIRONMENT.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        objective_source_sha256=dep['objective_sha256'],device='CPU',process_affinity=[16],
        intraop_threads_after_imports=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,
        precision='FP32 highest',new_Raw_encoder_or_Outer_inference=False))
    io.write(pub/'PUBLIC_TRANSFER_MANIFEST.json',dict(status='PASS',
        code_sha256=io.sha(Path(__file__)),original_scientific_inputs_and_Outer_seal_unchanged=True,
        sealed_scores_unchanged=True,
        files={n:dict(bytes=(pub/n).stat().st_size,sha256=io.sha(pub/n)) for n in FILES}))
    print('READOUT_ANONYMOUS_TRANSFER_READY',len(FILES),flush=True)
