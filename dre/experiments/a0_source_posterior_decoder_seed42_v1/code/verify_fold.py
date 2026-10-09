"""Independent deterministic replay of completed private predictions, no new selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import sha,write_json,metric
from posterior import infer,decode

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--fold',type=int,required=True); a=p.parse_args()
    root=a.runtime/f'fold{a.fold}'; torch.set_num_threads(2)
    art=torch.load(root/'VALIDATION_PRIVATE.pt',weights_only=False); post=torch.load(root/'POSTERIOR_FROZEN_PRIVATE.pt',weights_only=False)
    assert art['density_sha256']==sha(root/'POSTERIOR_FROZEN_PRIVATE.pt') and art['scorer_checkpoint']==sha(root/'D2/BEST_PRIVATE.pt')
    bank=torch.load(root/'BANK_PRIVATE.pt',weights_only=False); full=torch.load(root/'D2_FROZEN_PRIVATE.pt',weights_only=False)
    checks=0; drift=0.
    for cell in art['channels']:
        ix=np.flatnonzero(bank['patient']==cell['patient']); ix=ix[np.argsort(bank['channel'][ix].astype(str),kind='stable')]
        assert np.array_equal(cell['channel'],bank['channel'][ix]) and np.array_equal(cell['y'],bank['y'][ix])
        for method,log in [('D1',bank['a0_logits'][ix]),('D3',full['logits'][ix])]:
            q,diag=infer(log,int(bank['g'][ix[0]]),post['densities'][method]); pred,dc=decode(q,cell['channel'],log)
            old=cell['posterior'][method]; assert np.array_equal(q,old['q']) and np.array_equal(~pred,cell['decisions'][method])
            assert diag['pi']==old['diag']['pi'] and dc['k']==old['decoder']['k']; checks+=1
    write_json(root/'INDEPENDENT_REPLAY.json',{'status':'PASS','fold':a.fold,'posterior_patient_cells':checks,'posterior_max_drift':drift,
        'frozen_hashes_passed':True,'decision_bitwise_parity':True,'no_refitting_or_reoptimization':True})
    print('INDEPENDENT_REPLAY_PASS',a.fold,checks,flush=True)

if __name__=='__main__':main()
