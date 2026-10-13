"""Outcome-free runtime parity against already-completed nuisance preparations."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import scipy
import sklearn
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from conditional_ridge import fit_cross,coordinates

def main():
 root=Path(r'D:\a0_cove_seed42_v1_runtime_r2');files=sorted(root.glob('fold*/BANK_PRIVATE.pt'));assert len(files)==3
 coef=normal=e_delta=q_delta=0.
 for fold,p in enumerate(files,1):
  b=torch.load(p,map_location='cpu',weights_only=False);tr=b['train'];s,e=fit_cross(b['x'][tr],b['patient'][tr],b['mapping'],42+1009*fold)
  normal=max(normal,float(np.abs(s['scale']-b['stats']['scale']).max()));e_delta=max(e_delta,float(np.abs(e/s['scale']-b['e'][tr]).max()));q_delta=max(q_delta,float(np.abs(coordinates(e/s['scale'])-coordinates(b['e'][tr])).max()))
  for j in range(14):
   for k in ['Dcoef','Dintercept','Rcoef','Rintercept']:coef=max(coef,float(np.abs(np.asarray(s['full'][j][k])-np.asarray(b['stats']['full'][j][k])).max()))
 assert coef<1e-12 and normal<1e-12 and e_delta<1e-6 and q_delta<1e-6
 result=dict(status='PASS',folds=3,max_coefficient_drift=coef,max_scale_drift=normal,max_normalized_residual_drift=e_delta,max_q_drift=q_delta,
  python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,cudnn=torch.backends.cudnn.version(),torch_C_sha256=hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest(),no_clinical_model_outcomes_or_outer=True)
 path=Path(r'D:\cove_runtime_parity.json');assert not path.exists();path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
