import torch,numpy as np
from neuroez_c.task2.true_dualset.model import TrueEZNEZDualSet
from neuroez_c.task2.true_dualset.trainer import EMA,loss_fn
from test_true_dualset_routing import seizure
def test_seed_all_accepts_pandas_style_numpy_integer():
 from neuroez_c.task2.true_dualset.trainer import seed_all
 seed_all(np.int64(43))
def test_sizes_and_full_model_smoke():
 m=TrueEZNEZDualSet();views=[]
 for i,(e,n) in enumerate(((1,1),(1,20),(20,1),(24,40))):
  labels=[0]*e+[1]*n;views.append({'patient_key':str(i),'center':'x','outcome_success':i%2,'view_id':0,'seizures':[seizure(labels),seizure(labels)]})
 out=m(views);loss,_=loss_fn(out,[v['outcome_success'] for v in views],.05);opt=torch.optim.AdamW(m.parameters());loss.backward();opt.step();ema=EMA(m,.98);ema.update(m);assert m.parameter_count()<300000 and all(torch.isfinite(x['logit']) for x in out) and any(p.grad is not None for p in m.cross_set.parameters()) and any(p.grad is not None for p in m.patient_head.parameters())
