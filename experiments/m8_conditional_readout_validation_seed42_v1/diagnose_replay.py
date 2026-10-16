"""Diagnose the first sealed-score replay mismatch; no outcomes or fitting."""
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_readouts as run
from readout import Readout
io=run.io


def configuration():
    return dict(threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                float32_precision=torch.get_float32_matmul_precision(),mkldnn=torch.backends.mkldnn.enabled,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)


def main():
    torch.set_num_threads(1);before=configuration()
    common,_,_,_,_=io.old_modules();_,core,_,_,_,_,banks,_=common.legal();after=configuration()
    registry=io.read(run.ROOT/'READOUT_REGISTRY_PRIVATE.pkl');entry=registry['entries'][0]
    q=next(e for e in io.read(run.ROOT/'PLAN_PRIVATE.pkl')['entries'] if e['fold']==entry['fold'] and e['screen']==entry['screen'])
    bank=banks[entry['fold']-1];fit=bank['train'];pid=bank['patient'][fit]
    raw=io.read(Path(q['raw_entry']['path']));pool=io.read(io.OLDROOT/'POOL_PRIVATE.pkl')
    x=np.c_[raw['raw'],core.patient_z(pool['summary'][bank['archive_indices'][fit]],pid)]
    refit=np.r_[q['roles']['train'],q['roles']['choice']];held=q['roles']['held']
    pre=io.read(run.ROOT/f'fold{entry["fold"]}_screen{entry["screen"]}_PREPROCESS_PRIVATE.pkl')['pre']
    z,prepared=run.prepared(x,refit)
    assert all(np.array_equal(pre[k],prepared[k]) for k in pre)
    checkpoint=torch.load(Path(entry['cell'])/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
    model=Readout(entry['groups'],entry['head_seed']+1009*entry['fold']+50021*entry['screen'])
    model.load_state_dict(checkpoint['model']);stored=io.read(Path(entry['score']))['logit_nez']
    reports=[]
    for label,threads,precision in [('as_report',torch.get_num_threads(),torch.get_float32_matmul_precision()),
                                     ('explicit_production',1,'highest')]:
        torch.set_num_threads(threads);torch.set_float32_matmul_precision(precision)
        logit=run.prediction(model,z[held])
        reports.append(dict(mode=label,config=configuration(),array_equal=bool(np.array_equal(logit,stored)),
                            max_absolute_error=float(np.max(np.abs(logit.astype(float)-stored))),
                            different_values=int(np.count_nonzero(logit!=stored))))
    io.write(run.PUB/'REPLAY_DIAGNOSIS.json',dict(status='DIAGNOSED_NO_FITTING_OR_OUTCOME_METRICS',
        first_cell=dict(fold=entry['fold'],screen=entry['screen'],seed=entry['head_seed'],arm=entry['arm']),
        before_legal_import=before,after_legal_import=after,tests=reports,
        checkpoint_and_preprocessing_unchanged=True,no_threshold_or_scientific_change=True))
    print(io.jr(run.PUB/'REPLAY_DIAGNOSIS.json'),flush=True)


if __name__=='__main__':
    main()
