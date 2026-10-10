"""Read-only all-fold fusion identity and S0 feature-independence checks."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from audit_raw import ROOT
from common import write_json
from data import RawBank
from model import SSSMIL
from prepare import PRIOR
from runtime import seed_all


def main():
    torch.set_num_threads(2)
    checks=[]
    for fold in range(1,6):
        d=torch.load(PRIOR/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        p=sorted(set(d['patient'][d['val']]))[0]
        ix=d['val'][d['patient'][d['val']]==p][:4]
        bank=RawBank([p])
        inputs=bank.sample(p,d['channel'][ix],42+1009*fold)
        seed_all(42+1009*fold)
        model=SSSMIL('S1',42+1009*fold).eval()
        saved=torch.load(PRIOR/f'fold{fold}/D2/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        model.d2.load_state_dict(saved['model'])
        x=torch.as_tensor(d['x'][ix]); g=torch.as_tensor(d['g'][ix])
        with torch.no_grad():
            fused=model(*inputs,x,g); reference=model.d2(x,g)
            assert torch.equal(fused,reference)
            rawonly=SSSMIL('S0').eval()
            score1=rawonly(*inputs,x,g)
            score2=rawonly(*inputs,None,None)
            assert torch.equal(score1,score2)
        b=bank.bags[str(p)]
        assert np.array_equal(b['mask'].any(-1).numpy(),b['mask'][b['perm']].any(-1).numpy())
        checks.append({'fold':fold,'initial_real_validation_logit_drift':float((fused-reference).abs().max()),
                       'S0_88D_and_source_independence_exact':True,'shuffle_seizure_mask_preserved':True})
    write_json(ROOT/'audit/SUPPLEMENTAL_IDENTITY_AUDIT.json',{'status':'PASS','folds':checks,
        'clinical_labels_used':False,'formal_models_modified':False,'device':'cpu',
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
        'supplemental_read_only_no_new_model_arm':True})
    print('ALL_FIVE_REAL_VALIDATION_GAMMA_ZERO_PARITY_AND_S0_INDEPENDENCE_PASS',flush=True)


if __name__=='__main__':
    main()
