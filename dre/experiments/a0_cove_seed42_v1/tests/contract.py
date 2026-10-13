"""Executed mapping, FIT-only nuisance, identity, gradients, adapter parity and resume."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from cove_model import constructor

def run(a,o,banks):
 tests=[];core,c=o.core,o.c;identity=0.;residual_drift=0.
 for fold,b in enumerate(banks,1):
  for arm in ['B1','B2','B3']:
   seed=42+1009*fold;fac=constructor(o.original.SourceA0,arm,b['mapping'],b['stats'],b['permutation'],b)
   core.seed_all(seed);m=fac(seed).eval();core.seed_all(seed);base=o.helper.source_free_factory(o.original.SourceA0)(seed).eval()
   for k,v in m.network.state_dict().items():assert torch.equal(v,base.network.state_dict()[k])
   assert sum(p.numel() for p in m.parameters())==8845 and torch.equal(m.alpha,torch.zeros(14)) and torch.equal(m.beta,torch.zeros(14))
   assert set(n for n,p in m.named_parameters() if not n.startswith('network.'))=={'alpha','beta'}
   tx=torch.from_numpy(b['x']);g=torch.as_tensor(b['g']);va=b['val'];fit=b['train']
   with torch.inference_mode():
    xt=m.transformed(tx);ll=m(tx,g);bl=base(tx,g);drift=float((ll-bl).abs().max());identity=max(identity,drift)
    assert torch.equal(xt,tx) and drift<=1e-6
    direct=m.residuals(tx[va]).numpy();residual_drift=max(residual_drift,float(np.abs(direct-b['e'][va]).max()))
    assert np.allclose(direct,b['e'][va],atol=1e-6,rtol=0)
   for device in ['cpu','cuda']:
    m=m.to(device);ix=fit[:64];inp=torch.as_tensor(b['x'][ix],device=device);rr=torch.as_tensor(b['e'][ix],device=device,dtype=torch.float64)
    m.zero_grad();m.train();out=m.transformed(inp,rr);loss=(out*torch.linspace(.1,1.,88,device=device)).sum();loss.backward()
    assert torch.isfinite(m.alpha.grad).all() and torch.isfinite(m.beta.grad).all() and (m.alpha.grad.abs()>1e-10).any() and (m.beta.grad.abs()>1e-10).any()
    # Direct Jacobians: every numerically active coefficient can learn at zero.
    ed=torch.tanh(rr[:,:,0]);er=torch.tanh(rr[:,:,1]);prod=ed*er[:,m.permutation if arm=='B2' else torch.arange(14,device=device)]
    for j in range(14):
     for param,idx,qr in [(m.alpha,m.delta_idx[j],ed[:,j] if arm=='B1' else prod[:,j]),(m.beta,m.ratio_idx[j],er[:,j] if arm=='B1' else prod[:,j])]:
      row=torch.argmax(qr.abs())
      if qr[row].abs()>1e-8:
       grad=torch.autograd.grad(m.transformed(inp,rr)[row,idx],param)[0]
       assert grad[j].abs()>1e-10 and torch.isfinite(grad).all()
    m.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(m(inp),torch.as_tensor(b['y'][ix],device=device,dtype=torch.float32)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with torch.no_grad():m.alpha.fill_(.2);m.beta.fill_(-.3)
    m.eval()
    with torch.inference_mode():
     modified=m.transformed(inp,rr);assert torch.equal(modified[:,b['mapping']['zero']+b['mapping']['masked']],inp[:,b['mapping']['zero']+b['mapping']['masked']])
     for dest,src in b['mapping']['duplicates']:assert torch.equal(modified[:,dest],modified[:,src])
     indexed=m(torch.as_tensor(va[:32],device=device,dtype=torch.float32)[:,None]);direct=m(torch.as_tensor(b['x'][va[:32]],device=device));assert torch.allclose(indexed,direct,atol=1e-6,rtol=0)
   del m,base
 tests.extend(['all14_exact_triples_and_masked88_interface','all15_original_backbone_initializations','zero28_identity_on_all_legal_samples','8845_parameters_only28_new','finite_CPU_CUDA_gradients_and_nonzero_zero_init_coefficient_flow','protected_zero_and_duplicate_updates','OOF_index_adapter_and_fullFIT_target_inference_parity'])
 # Exact resumed original trainer with synthetic labels; no clinical selection.
 b=banks[0];ix=b['train'][:16];synthetic=dict(x=b['x'][ix],e=b['e'][ix]);fac=constructor(o.original.SourceA0,'B3',b['mapping'],b['stats'],b['permutation'],synthetic)
 sx=np.arange(16,dtype=np.float32)[:,None];yy=np.tile([0,1],8);pid=np.repeat(['s1','s2','s3','s4'],4);g=np.zeros(16,dtype=int);root=a.runtime/'engineering/resume';old=o.trainer.SourceA0;o.trainer.SourceA0=fac
 try:
  o.trainer.train(sx,yy,pid,g,np.arange(8),np.arange(8,16),root/'continuous',1051,'synthetic_cove',device='cpu',max_epochs=3,penalty=0.)
  o.trainer.train(sx,yy,pid,g,np.arange(8),np.arange(8,16),root/'resume',1051,'synthetic_cove',device='cpu',max_epochs=3,stop_after=1,penalty=0.)
  o.trainer.train(sx,yy,pid,g,np.arange(8),np.arange(8,16),root/'resume',1051,'synthetic_cove',device='cpu',max_epochs=3,penalty=0.)
 finally:o.trainer.SourceA0=old
 p,q=[torch.load(root/tag/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False) for tag in ['continuous','resume']]
 assert core.state_hash(p['model'])==core.state_hash(q['model']) and p['history']==q['history'] and p['best_key']==q['best_key']
 for k,slot in p['optimizer']['state'].items():
  for n,v in slot.items():assert torch.equal(v,q['optimizer']['state'][k][n])
 assert torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda']))
 tests.append('exact_CPU_original_trainer_model_Adam_history_CPU_CUDA_RNG_resume')
 return dict(tests=tests,test_count=len(tests),max_identity_logit_drift=identity,max_numpy_torch_residual_drift=residual_drift,synthetic_exact_resume=True,GPU_bitwise_resume_not_claimed=True,no_outer_access=True)
