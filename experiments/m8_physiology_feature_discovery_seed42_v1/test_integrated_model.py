"""Outcome-free94-position folding,RNG,parameter and raw-prefix parity tests."""
import copy
from audit_old7 import *
from torch import nn
from compact_delta_classifier import CompactDelta
P3CODE=Path(r'D:\d1_small_module_exploration_seed42_v1\phase3\code');sys.path.insert(0,str(P3CODE))
P3=load('integration_synthetic_original',P3CODE/'run.py')
M=load('integration_slot_models',TASK/'integrated_model.py')
records=[];offset=0
for count in [2]*8+[1]*6:
    records.append(dict(index=offset,copies=list(range(offset,offset+count)),multiplicity=count));offset+=count
assert offset==22

def network(dtype):return nn.Sequential(nn.LayerNorm(88,dtype=dtype),nn.Linear(88,96,dtype=dtype),nn.GELU(),nn.Dropout(.15),nn.Linear(96,1,dtype=dtype))

torch.manual_seed(42);net=network(torch.float64)
with torch.no_grad():net[0].weight.uniform_(.5,1.5);net[0].bias.normal_(std=.1)
state=torch.get_rng_state();legacy=CompactDelta(net,records);expected=torch.get_rng_state()
torch.set_rng_state(state);base=M.SlotBase(net,records);assert torch.equal(expected,torch.get_rng_state())
x=torch.randn(19,20,dtype=torch.float64);expanded=torch.zeros(19,94,dtype=torch.float64)
for j,r in enumerate(records):expanded[:,r['copies']]=x[:,j,None]
expanded[:,88:]=x[:,14:]
w=torch.cat([net[1].weight,torch.zeros(96,6,dtype=torch.float64)],1)
gamma=torch.cat([net[0].weight,torch.ones(6,dtype=torch.float64)]);beta=torch.cat([net[0].bias,torch.zeros(6,dtype=torch.float64)])
reference=torch.nn.functional.linear(torch.nn.functional.layer_norm(expanded,(94,),gamma,beta,1e-5),w,net[1].bias)
actual=base.first(base.normalized(x));error=float((reference-actual).abs().max().detach());assert error<1e-12
assert sum(p.numel() for p in base.parameters())==2113
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
audits=[]
for device in ['cpu','cuda']:
    torch.manual_seed(1051);net=network(torch.float32);legacy=CompactDelta(net,records)
    original=P3.MorphCorrection(legacy).to(device)
    integrated=M.IntegratedM8(M.SlotBase(net,records),original).to(device)
    assert sum(p.numel() for p in integrated.parameters())==11690
    assert sum(p.numel() for p in integrated.parameters() if p.requires_grad)==9577
    raw=torch.randn(2,6,5,500,device=device);mask=torch.ones(2,6,5,dtype=torch.bool,device=device)
    times=torch.tensor([[-4,-2,2,6,12]]*2,dtype=torch.float32,device=device)
    delta=torch.randn(2,6,5,7,device=device);wide=torch.cat([delta,torch.zeros(2,6,5,3,device=device)],-1)
    with torch.inference_mode():
        left=original.features(raw,mask,times,delta,torch.ones(7,device=device))[0]
        right=integrated.features(raw,mask,times,wide,torch.ones(10,device=device))[0]
    e=float((left-right).abs().max());assert e==0
    tx=torch.randn(6,20,device=device);integrated.eval()
    with torch.inference_mode():q=integrated(tx,raw,mask,times,wide,torch.ones(10,device=device));assert torch.equal(q[0],q[2])
    fn=torch.nn.BCEWithLogitsLoss();opt=torch.optim.AdamW([p for p in integrated.parameters() if p.requires_grad],lr=.0003)
    yy=torch.tensor([0,1,0,1,1,1],device=device,dtype=torch.float32)
    delta10=torch.randn(2,6,5,10,device=device)
    for step in range(3):
        opt.zero_grad(set_to_none=True);pred=integrated(tx,raw,mask,times,delta10,torch.ones(10,device=device))[0]
        loss=fn(pred,yy);loss.backward()
        assert all(p.grad is None for p in integrated.base.parameters())
        assert all(torch.isfinite(p.grad).all() for p in integrated.parameters() if p.grad is not None)
        assert integrated.head[2].weight.grad.norm()>0
        if step>=1:assert integrated.cnn[0].weight.grad.norm()>0
        if step>=2:assert integrated.gate.network[0].weight.grad[:,7:].norm()>0
        opt.step()
    audits.append(dict(device=device,status='PASS',raw_prefix_feature_max_error=e,three_gradient_steps=True))
    print('INTEGRATED_SLOT10_SYNTHETIC_PASS',device,flush=True)
write(OUT/'public/INTEGRATED_MODEL_SYNTHETIC_AUDIT.json',dict(status='PASS',full94_float64_fold_error=error,
    nondefault_LN_affine=True,constructor_CPU_RNG_exact=True,base_parameters=2113,raw_trainable=9577,total_parameters=11690,
    relative_to_legacy=11690/11018,tests=audits,source_sha256=sha(TASK/'integrated_model.py'),test_sha256=sha(__file__)))
