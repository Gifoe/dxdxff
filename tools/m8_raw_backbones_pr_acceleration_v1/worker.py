"""Execution-only wrapper: original frozen model/trainer sources are not edited."""
import sys
from pathlib import Path
from types import SimpleNamespace
CODE=Path(r'D:\m8_raw_backbones_pr_seed42_v1')
EXEC=Path(__file__).parent
sys.path.insert(0,str(CODE))
from support import *
import engine
import models.omni_archived as archived
from fast_engine import install


def run(phase,model):
    admission=jr(EXEC/'DEFERRED_FINITE_ADMISSION.json')
    assert admission['status']=='PASS' and admission['logits_loss_all_gradients_BN_buffers_and_RNG_bit_identical']
    assert admission['original_source_input_binding_sha256']==digest(jr(RUNTIME/'INPUT_FREEZE_PRIVATE.json'))
    manifest=EXEC/'EXECUTION_MANIFEST.json'
    if manifest.exists():
        locked=jr(manifest)
        for f,h in locked['execution_code_sha256'].items():assert sha(EXEC/f)==h,f
    original,oldcheck,audit=install(engine,archived)
    assert audit['original_forward_sha256']==admission['original_forward_sha256'] and audit['deferred_forward_sha256']==admission['deferred_forward_sha256']
    a=SimpleNamespace(runtime=RUNTIME,model=model)
    print('VERIFIED_EXECUTION_ONLY_ACCELERATION',phase,model,flush=True)
    if phase=='smoke':
        m=load('execution_fast_four_model_smoke',CODE/'tests/smoke.py')
        m.main(Path(r'D:\m8_raw_backbones_pr_seed42_v1_accelerated_smoke_runtime'))
    elif phase=='train':load('execution_original_frozen_trainer',CODE/'train.py').run(a)
    elif phase=='outer':load('execution_original_frozen_outer',CODE/'outer.py').run(a)
    elif phase=='aggregate':load('execution_original_frozen_aggregate',CODE/'aggregate.py').aggregate_results(RUNTIME,jr(CODE/'configs/PROTOCOL_LOCK.json'))
    elif phase=='verify':load('execution_original_frozen_verifier',CODE/'tests/verify_results.py').run(RUNTIME)
    else:raise ValueError(phase)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','train','outer','aggregate','verify']);p.add_argument('--model',default='TimeConvCNN');args=p.parse_args();run(args.phase,args.model)
