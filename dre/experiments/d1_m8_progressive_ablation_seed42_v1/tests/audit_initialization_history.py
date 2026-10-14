"""Additional read-only audit against initial hashes stored in original D1."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from support import args,jread,sha,write,torch

def run(a):
    q=jread(a.runtime/'results/INITIALIZATION_PARITY.json');rows=[]
    for fold in range(1,6):
        cp=a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt'
        original=torch.load(cp,map_location='cpu',weights_only=False)['binding']['initial_hash']
        new=q['folds'][fold-1]
        assert original==new['A0_sha256']==new['A1_sha256']
        rows.append(dict(fold=fold,original_D1_checkpoint_sha256=sha(cp),original_recorded_initial_state_sha256=original,A0_A1_exact_original_initial_state=True))
    write(a.runtime/'results/ORIGINAL_INITIALIZATION_AUDIT.json',dict(status='PASS',folds=rows,audit_source_sha256=sha(Path(__file__)),new_training=False))
    print('ORIGINAL_D1_RECORDED_INITIALIZATIONS_EXACT_5_OF_5')

if __name__=='__main__':run(args())
