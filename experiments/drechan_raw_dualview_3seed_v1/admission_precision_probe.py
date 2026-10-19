"""Diagnose float64-to-float32 audit graph differences before any formal fit."""
import ctypes
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),ctypes.c_size_t(0xffff0000))
import copy
from types import MethodType
from common import *
from admit import independent_features,equal_tree


def separate_cast_features(self,raw,mask,times,delta,scale,permutation=None):
    saved=self.raw_pr;self.raw_pr=False
    try:v,activity=MODEL.BASE.BASE.PRFactorial.features(self,raw,mask,times,delta,scale,permutation)
    finally:self.raw_pr=saved
    g=((v.double()-self.anchor_mean)/self.anchor_scale).to(v.dtype)
    a=v.double();sd=a.std(0,unbiased=False)
    r=((a-a.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).to(v.dtype)
    return torch.cat([g,r],dim=-1),activity


def main():
    assert not (ROOT/'INPUT_FREEZE_PRIVATE.json').exists() and not (ROOT/'models_private').exists()
    binding,core,fac,records,rec,banks=setup(False);d,scale=data(1,core,rec,banks,store(),binding);inputs=store();counts=[]
    for pid in sorted(d['roles']['train']):
        ii=d['train'][d['patient'][d['train']]==pid];raw,mask,_,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cpu')
        counts.append((int(mask.sum()),str(pid)));del raw,delta
    windows,pid=max(counts);ii=d['train'][d['patient'][d['train']]==pid];tr=d['train']
    y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32);pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    production=make(core,fac,records,1,'FULL',42)
    models={'production':production,'shared_double_audit':copy.deepcopy(production),'separate_double_audit':copy.deepcopy(production)}
    models['shared_double_audit'].features=MethodType(independent_features,models['shared_double_audit'])
    models['separate_double_audit'].features=MethodType(separate_cast_features,models['separate_double_audit'])
    outputs={};losses={};gradients={};opts={}
    for name,m in models.items():
        m.train();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
        out,_=forward(m,d,ii,inputs,scale);loss=MODEL.objective(out,y,pos);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        outputs[name]=out.detach().cpu();losses[name]=loss.detach().cpu()
        gradients[name]={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters()};opts[name]=opt
    rows=[]
    for name in ['shared_double_audit','separate_double_audit']:
        pairs=[(gradients['production'][n],value) for n,value in gradients[name].items()]
        rows.append(dict(case=name,logits_bit_equal=torch.equal(outputs['production'],outputs[name]),loss_bit_equal=torch.equal(losses['production'],losses[name]),
                         all_gradients_bit_equal=all(torch.equal(a,b) for a,b in pairs),maximum_absolute_gradient_difference=max(float((a-b).abs().max()) for a,b in pairs),
                         unequal_parameter_tensors=sum(not torch.equal(a,b) for a,b in pairs)))
    assert all(r['logits_bit_equal'] and r['loss_bit_equal'] for r in rows)
    assert not rows[0]['all_gradients_bit_equal'] and rows[1]['all_gradients_bit_equal'] and rows[1]['maximum_absolute_gradient_difference']==0
    for name in ['production','separate_double_audit']:
        torch.nn.utils.clip_grad_norm_(models[name].parameters(),1.);opts[name].step()
    assert state_hash(models['production'].state_dict())==state_hash(models['separate_double_audit'].state_dict())
    assert equal_tree(opts['production'].state_dict(),opts['separate_double_audit'].state_dict())
    audit=dict(status='PRETRAIN_AUDIT_CAST_GRAPH_MISMATCH_VERIFIED',source_sha256=sha(HERE/'admission_precision_probe.py'),scientific_sources=sources(),
               TRAIN_channels=len(ii),TRAIN_valid_windows=windows,comparisons=rows,separate_cast_model_and_Adam_update_bit_equal=True,
               formal_fits=0,new_assessment=False,new_Outer=False,model_or_loss_or_data_or_tolerance_changed=False,
               explanation='Production casts the global and patient branches separately. The initial independent auditor shared one double tensor,accumulating branch gradients before its single cast. Equal forward real formulas do not imply identical finite-precision gradient graphs.')
    write(PUB/'ADMISSION_PRECISION_PROBE.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
