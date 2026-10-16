"""Restore declared production threads after imports; preserve frozen sources."""
import runpy
from pathlib import Path

import torch
import run_readouts as run
io=run.io
HERE=Path(__file__).resolve().parent


def main():
    original=io.old_modules
    assert io.jr(run.PUB/'READOUT_STATUS.json')['status']=='ALL90_INTERNAL_READOUTS_SEALED_BEFORE_ASSESSMENT_REPORT'
    diagnosis=io.jr(run.PUB/'REPLAY_DIAGNOSIS.json')
    assert diagnosis['tests'][0]['config']['threads']==2
    assert diagnosis['tests'][1]['array_equal'] and diagnosis['tests'][1]['max_absolute_error']==0
    io.freeze(run.ROOT/'REPLAY_EXECUTION_BINDING_PRIVATE.json',dict(
        original_input_freeze_sha256=io.sha(run.ROOT/'INPUT_FREEZE_PRIVATE.json'),
        original_score_seal_sha256=io.sha(run.ROOT/'READOUT_SCORE_SEAL_PRIVATE.json'),
        amendment_sha256=io.sha(HERE/'EXECUTION_ONLY_REPLAY_FIX.json'),driver_sha256=io.sha(Path(__file__))))

    def modules_with_production_import_boundary():
        modules=original();common=modules[0];legal=common.legal
        def production_legal(*args,**kwargs):
            value=legal(*args,**kwargs)
            torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
            return value
        common.legal=production_legal
        return modules

    io.old_modules=modules_with_production_import_boundary
    try:
        runpy.run_path(str(HERE/'readout_report.py'),run_name='__main__')
    finally:
        io.old_modules=original


if __name__=='__main__':
    main()
