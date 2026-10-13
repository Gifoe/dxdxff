"""Executed source, raw-only purity/pooling/gradient/chunk/resume and split tests."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit_sources import arguments,context
from raw_alignment import support,RawBankStore
from raw_only_model import constructor,patient_z,pool,encoder_class
from metrics import independent


class SyntheticStore:
 def __init__(self):
  generator=torch.Generator().manual_seed(77)
  self.raw=torch.randn(1,2,4,3,500,generator=generator);self.mask=torch.ones(1,2,4,3,dtype=torch.bool)
  self.mask[0,1,0]=False;self.mask[0,0,3,2]=False
 def get(self,p,channels):return self.raw,self.mask


def main():
 a=arguments();o=context(a);lock,ref,c,core,original,trainer,helper,banks,manifest,bind=o
 assert json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text())['binding']==bind
 tests=[]
 starts,valid=support([-29,-28,0,28,29],[],250,60,15000,250,14500);assert starts.tolist()==[0,250,7250,14250,14500] and valid.tolist()==[False,True,True,True,False]
 try:support([0,0],[],250,60,15000,0,15000)
 except AssertionError:pass
 else:raise AssertionError('Duplicate windows accepted')
 tests.append('raw_valid_start_length_duplicate_and_padding_bounds')
 x=torch.tensor([[[1.,2.,0.],[2.,4.,0.],[999.,999.,999.]]],requires_grad=True);cm=torch.tensor([[True,True,False]])
 z=patient_z(x,cm);assert torch.equal(z[0,:,2],torch.zeros(3)) and torch.equal(z[0,2],torch.zeros(3)) and torch.allclose(z[0,:2,:2],torch.tensor([[-1.,-1.],[1.,1.]]))
 (z*torch.arange(z.numel()).reshape_as(z)).sum().backward();assert torch.isfinite(x.grad).all() and torch.equal(x.grad[0,2],torch.zeros(3));tests.append('masked_differentiable_patientZ_and_constant_feature_behavior')
 e=torch.zeros(1,2,2,3,32);wm=torch.zeros(1,2,2,3,dtype=torch.bool);e[0,0,0,:2]=2.;wm[0,0,0,:2]=True;e[0,1,0,:3]=4.;wm[0,1,0,:3]=True;e[0,0,1,0]=7.;wm[0,0,1,0]=True;e[~wm]=999.
 pooled,cc=pool(e,wm);assert torch.equal(pooled[0,0,:32],torch.full((32,),3.)) and torch.equal(pooled[0,0,32:],torch.ones(32))
 assert torch.equal(pooled[0,1,:32],torch.full((32,),7.)) and torch.equal(pooled[0,1,32:],torch.zeros(32)) and cc.all();tests.append('two_level_masked_mean_population_std_missing_seizure')
 st=SyntheticStore();meta=dict(patient=np.array(['synthetic']*4),channel=np.array(['A','B','C','D']),x=np.ones((4,88)))
 fac=constructor(original,a.source,st,meta);core.seed_all(1051);m=fac(1051).cuda().eval();core.seed_all(1051);base=helper.source_free_factory(original.SourceA0)(1051)
 for k,v in m.network.state_dict().items():assert torch.equal(v.cpu(),base.network.state_dict()[k])
 assert sum(p.numel() for p in m.parameters())==18153;tests.append('raw18153_classifier8817_exact_original_initialization')
 ix=torch.arange(4,dtype=torch.float32,device='cuda')[:,None]
 with torch.inference_mode():first=m(ix)
 meta2=dict(meta,x=np.zeros((4,88)),labels=np.ones(4),center=np.array(['ignored']*4));core.seed_all(1051);m2=constructor(original,a.source,st,meta2)(1051).cuda().eval()
 with torch.inference_mode():assert torch.equal(first,m2(ix))
 tests.append('engineered88_value_zeroing_labels_centers_ignored_input_purity')
 saved=st.raw.clone();st.raw[~st.mask]=123456.
 with torch.inference_mode():assert torch.equal(first,m(ix))
 st.raw=saved;tests.append('invalid_waveform_values_and_padding_do_not_affect_logits')
 try:m(ix[:2])
 except AssertionError:pass
 else:raise AssertionError('Partial patient normalization accepted')
 tests.append('patientZ_never_uses_microbatch_subset_statistics')
 Encoder=encoder_class(a.source);core.seed_all(4);enc=Encoder(chunk_size=4).cuda().train();enc2=copy.deepcopy(enc);enc2.chunk_size=256
 rw=torch.randn(1,2,4,3,500);mask=st.mask;coeff=torch.randn(1,2,4,3,32,device='cuda')
 v1=enc(rw,mask);v2=enc2(rw,mask);max_forward=float((v1-v2).abs().max())
 assert torch.allclose(v1,v2,atol=lock['parity']['raw_chunk_forward_atol'],rtol=lock['parity']['relative_tolerance'])
 denominator=int(mask.sum())*v1.shape[-1]
 (v1*coeff).sum().div(denominator).backward();(v2*coeff).sum().div(denominator).backward();max_gradient=0.
 for p,q in zip(enc.parameters(),enc2.parameters()):
  assert p.grad is not None and q.grad is not None and torch.isfinite(p.grad).all();max_gradient=max(max_gradient,float((p.grad-q.grad).abs().max()))
  assert torch.allclose(p.grad,q.grad,atol=lock['parity']['raw_chunk_gradient_atol'],rtol=lock['parity']['relative_tolerance'])
 tests.append('unchanged_RawTiny_GPU_forward_gradient_chunk_equivalence')
 del m,m2,enc,enc2,v1,v2;torch.cuda.empty_cache()
 # Exact CPU model/optimizer resume plus CPU/CUDA RNG-state restore checks.
 # GPU kernel self replay is numerically nondeterministic (qualified separately).
 sx=np.arange(16,dtype=np.float32)[:,None];sy=np.tile([0,1],8);pid=np.repeat(['s1','s2','s3','s4'],4);g=np.zeros(16,dtype=int)
 fac=constructor(original,a.source,st,dict(patient=pid,channel=np.tile(['A','B','C','D'],4)))
 old=trainer.SourceA0;trainer.SourceA0=fac;root=a.runtime/'engineering/raw_resume'
 try:
  trainer.train(sx,sy,pid,g,np.arange(8),np.arange(8,16),root/'continuous',1051,'synthetic',max_epochs=3,penalty=0.,device='cpu')
  trainer.train(sx,sy,pid,g,np.arange(8),np.arange(8,16),root/'resumed',1051,'synthetic',max_epochs=3,stop_after=1,penalty=0.,device='cpu')
  trainer.train(sx,sy,pid,g,np.arange(8),np.arange(8,16),root/'resumed',1051,'synthetic',max_epochs=3,penalty=0.,device='cpu')
 finally:trainer.SourceA0=old
 p,q=[torch.load(root/s/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False) for s in ['continuous','resumed']]
 assert core.state_hash(p['model'])==core.state_hash(q['model']) and p['history']==q['history'] and p['best_key']==q['best_key']
 for k,slot in p['optimizer']['state'].items():
  for n,v in slot.items():assert torch.equal(v,q['optimizer']['state'][k][n])
 assert torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda']));tests.append('raw_exact_CPU_model_Adam_history_and_CPU_CUDA_RNG_restore')
 for y,s,t in [(np.array([0,1,0,1]),np.array([.1,.2,.2,.8]),.2),(np.array([1,1]),np.array([.2,.3]),.5)]:
  aa=independent(y,s,t);bb=c.metric(y,s,t)
  for k in aa:assert np.isclose(aa[k],bb[k],atol=1e-12,rtol=0,equal_nan=True)
 tests.append('score_orientation_threshold_strictness_ties_and_metric_parity')
 # One original FIT patient only: feature zeroing purity + fixed label-swap
 # one-step diagnostic. No clinical validation or outer metric is inspected.
 d=banks[0];first_patient=sorted(set(d['patient'][d['train']]))[0];fi=np.flatnonzero(d['patient']==first_patient);assert set(fi)<=set(d['train'])
 store=RawBankStore(a.runtime,c);fac=constructor(original,a.source,store,dict(patient=d['patient'][fi],channel=d['channel'][fi],x=d['x'][fi]))
 core.seed_all(1051);ma=fac(1051).cuda();core.seed_all(1051);mb=fac(1051).cuda();assert core.state_hash(ma.state_dict())==core.state_hash(mb.state_dict())
 yy=d['y'][fi].copy();permuted=yy.copy();iz=np.flatnonzero(yy==0)[0];inz=np.flatnonzero(yy==1)[0];permuted[[iz,inz]]=permuted[[inz,iz]]
 assert np.array_equal(np.bincount(yy),np.bincount(permuted)) and not np.array_equal(yy,permuted)
 tx=torch.arange(len(fi),device='cuda',dtype=torch.float32)[:,None];pos=torch.tensor(max(int((1-d['y'][d['train']]).sum()),1)/max(int(d['y'][d['train']].sum()),1),device='cuda')
 for tm,yv in [(ma,yy),(mb,permuted)]:
  core.seed_all(1051);tm.train();opt=torch.optim.AdamW(tm.parameters(),lr=.001,weight_decay=.0001);opt.zero_grad(set_to_none=True)
  loss=torch.nn.functional.binary_cross_entropy_with_logits(tm(tx),torch.as_tensor(yv,dtype=torch.float32,device='cuda'),pos_weight=pos);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tm.parameters());torch.nn.utils.clip_grad_norm_(tm.parameters(),1.);opt.step()
 parameter_change=max(float((p-q).abs().max()) for p,q in zip(ma.parameters(),mb.parameters()));assert parameter_change>1e-8
 tests.append('FIT_only_fixed_label_permutation_changes_one_step_training_no_new_arm')
 c.write_json(a.runtime/'public/ENGINEERING_TESTS.json',dict(status='PASS',binding=bind,executed_tests=tests,test_count=len(tests),
  raw_chunk_max_forward_drift=max_forward,raw_chunk_max_gradient_drift=max_gradient,label_swap_one_step_max_parameter_difference=parameter_change,
  synthetic_resume_exact=True,synthetic_resume_device='cpu',GPU_bitwise_numerical_resume_not_claimed=True,clinical_diagnostic_uses_one_FIT_patient_no_selection=True,no_new_outer_metrics_viewed=True,torch=torch.__version__,code_sha256=c.sha(Path(__file__))))
 print('ALL_COMPACT_RAW_FIRST_ENGINEERING_GATES_PASS',len(tests),flush=True)


if __name__=='__main__':main()
