"""Supplemental CPU checkpoint/Adam/dropout resume equivalence; no formal weights."""
import copy
import io
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from audit_raw import ROOT
from common import write_json
from model import SSSMIL
from runtime import seed_all,state_hash,rng_state,restore_rng


def main():
    torch.set_num_threads(2)
    seed_all(902)
    model=SSSMIL('S1')
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    wave=torch.randn(3,2,3,512); aux=torch.randn(3,2,3,2)
    mask=torch.ones(3,2,3,dtype=torch.bool); mask[2,1]=False
    time=torch.zeros(3,2,3); x=torch.randn(3,88); g=torch.tensor([0,1,2]); y=torch.tensor([0.,1.,0.])
    def step(m,o):
        m.train(); o.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(m(wave,aux,mask,time,x,g),y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        o.step()
        return float(loss.detach())
    first=step(model,opt)
    buffer=io.BytesIO()
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),
                'epoch':1,'history':[first],'best':copy.deepcopy(model.state_dict()),'stale':0},buffer)
    uninterrupted=step(model,opt); final=state_hash(model.state_dict())
    # Construction consumes RNG; restoring saved RNG must happen AFTER construction/load.
    resumed=SSSMIL('S1'); optimizer=torch.optim.AdamW(resumed.parameters(),lr=3e-4,weight_decay=1e-4)
    buffer.seek(0); saved=torch.load(buffer,map_location='cpu',weights_only=False)
    resumed.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    replay=step(resumed,optimizer)
    assert replay==uninterrupted and state_hash(resumed.state_dict())==final
    assert saved['epoch']==1 and saved['stale']==0 and saved['history']==[first]
    for key,slots in opt.state_dict()['state'].items():
        for name,value in slots.items():
            other=optimizer.state_dict()['state'][key][name]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
    write_json(ROOT/'audit/RESUME_SERIALIZATION_AUDIT.json',{
        'status':'PASS','device':'cpu','model_state_bitwise_exact':True,'Adam_slots_bitwise_exact':True,
        'dropout_and_rng_replay_exact':True,'epoch_boundary_history_and_patience_preserved':True,
        'formal_checkpoint_modified':False,'supplemental_check_not_new_model_arm':True})
    print('CHECKPOINT_ADAM_RNG_RESUME_BITWISE_PASS',flush=True)


if __name__=='__main__':
    main()
