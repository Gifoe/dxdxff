"""Synthetic-only full/chunk/replay numerical qualification; no clinical data."""
import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit_sources import arguments,context
from raw_only_model import encoder_class


def main():
 a=arguments();o=context(a);c=o[2];core=o[3];Encoder=encoder_class(a.source);core.seed_all(4)
 reference=Encoder(chunk_size=256).cuda().train();rw=torch.randn(1,2,4,3,500);mask=torch.ones(1,2,4,3,dtype=torch.bool);mask[0,1,0]=False;mask[0,0,3,2]=False
 coeff=torch.randn(1,2,4,3,32,device='cuda');results=[]
 for denominator,name in [(1,'weighted_sum_fixture'),(int(mask.sum())*32,'valid_element_mean_fixture')]:
  encs=[copy.deepcopy(reference) for _ in range(3)];encs[0].chunk_size=4;outputs=[m(rw,mask) for m in encs]
  for output in outputs:(output*coeff).sum().div(denominator).backward()
  for left,right,pair in [(0,1,'chunk4_vs256'),(1,2,'same256_self_replay')]:
   forward=float((outputs[left]-outputs[right]).abs().max());drift=0.;relative=0.;failed=0
   for p,q in zip(encs[left].parameters(),encs[right].parameters()):
    drift=max(drift,float((p.grad-q.grad).abs().max()));relative=max(relative,float((p.grad-q.grad).abs().max()/p.grad.abs().max().clamp_min(1e-12)))
    failed+=int(not torch.allclose(p.grad,q.grad,atol=3e-5,rtol=3e-5))
   results.append(dict(fixture=name,pair=pair,denominator=denominator,max_forward_drift=forward,max_gradient_drift=drift,max_tensor_relative_gradient_drift=relative,failed_gradient_tensors_at_locked_limits=failed))
  del encs,outputs;torch.cuda.empty_cache()
 c.write_json(a.runtime/'public/RAW_CHUNK_NUMERICAL_DIAGNOSIS.json',dict(status='DIAGNOSTIC_ONLY_NOT_A_GATE_RELAXATION',results=results,clinical_data_used=False,model_or_training_change=False,
  note='Training uses averaged BCE and window/seizure means; synthetic summed coordinate probe accumulates FP32 scale. Loss normalization changes only fixture units, not tolerance or clinical objective.',script_sha256=c.sha(Path(__file__))))
 for r in results:print(r,flush=True)


if __name__=='__main__':main()
