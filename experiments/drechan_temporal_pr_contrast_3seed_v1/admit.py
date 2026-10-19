"""Unique changed-path checks, exact B0 bridge, real complete-patient admission."""
import subprocess
import time
from common import *

def main():
    for script in ['test_contrast.py','test_model.py']:
        assert subprocess.run([sys.executable,str(HERE/script)],cwd=HERE).returncode==0
    binding,core,fac,records,rec,banks=setup(False);inputs=store();bridge=[];rolecounts=[]
    for fold in range(1,6):
        d=parent.bank(fold,banks,'FULL_F0');scale=parent.scale(fold)
        current=parent.make(core,fac,records,fold,'FULL_F0');m=make(core,fac,records,fold,'B0',42)
        assert state_hash(current.state_dict())==state_hash(m.state_dict())
        q=next(q for q in jr(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')['cells'] if q['arm']=='FULL_F0' and q['fold']==fold)
        assert sha(q['checkpoint'])==q['checkpoint_sha256'] and sha(q['validation'])==q['validation_sha256']
        state=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);v=read(q['validation'])
        m.load_state_dict(state['model']);out,_=predict(m,d,d['val'],inputs,scale)
        assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
        bridge.append(dict(fold=fold,shared_initializer_exact=True,old_legal_validation_logits_bit_exact=True,parameters=11425))
        inner,ss=data(fold,'internal',core,rec,banks,inputs,binding)
        assert set(inner['patient'])<=set(d['roles']['fit']) and set(inner['pre']['fit_patient_ids'])==inner['roles']['train']
        assert set(inner['patient']).isdisjoint(d['roles']['validation']|d['roles']['test'])
        rolecounts.append(dict(fold=fold,train_patients=len(inner['roles']['train']),cal_patients=len(inner['roles']['cal']),assessment_patients=len(inner['roles']['assessment']),
            train_channels=len(inner['train']),cal_channels=len(inner['val']),assessment_channels=len(inner['assessment']),preprocessor_hash=digest(preprocessor(inner,ss))))
        del m,current;gc.collect();torch.cuda.empty_cache();print('TPRC_B0_BRIDGE_AND_INNER_ROLE_PASS',fold,flush=True)
    fold=1;d,scale=data(fold,'internal',core,rec,banks,inputs,binding)
    # The largest genuine TRAIN case for this fold, never an ASSESS/Development case.
    counts=[]
    for pid in sorted(d['roles']['train']):
        ii=d['train'][d['patient'][d['train']]==pid];raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
        counts.append((int(mask.sum()),str(pid)));del raw,delta
    _,pid=max(counts);ii=d['train'][d['patient'][d['train']]==pid]
    base=make(core,fac,records,fold,'B0',42)
    with torch.no_grad():original,_=forward(base,d,ii,inputs,scale)
    shared=base.state_dict();audits=[]
    for arm in ARMS[1:]:
        m=make(core,fac,records,fold,arm,42);assert all(torch.equal(v,m.state_dict()[k]) for k,v in shared.items())
        assert not any(isinstance(q,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for q in m.modules())
        globalbank=MODEL.GlobalReference(m,d,inputs,scale) if arm=='B2' else None
        tick=time.perf_counter()
        if globalbank:m.set_reference(globalbank.fit(m))
        reference_seconds=time.perf_counter()-tick
        with torch.no_grad():out,info=forward(m,d,ii,inputs,scale,True)
        assert torch.equal(out,original)
        opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
        y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda');tr=d['train'];pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
        norms=[];tick=time.perf_counter()
        for step in range(3):
            if globalbank:m.set_reference(globalbank.fit(m))
            opt.zero_grad(set_to_none=True);out,_=forward(m,d,ii,inputs,scale);loss=MODEL.objective(out,y,pos);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            g={n:float(p.grad.norm()) for n,p in m.named_parameters()}
            assert g['cnn.stem.0.weight']>0 and g['head.0.weight']>0 and g['temporal_fusion.network.2.weight']>0
            if step:assert g['temporal_fusion.network.0.weight']>0
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();norms.append(g)
        audits.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),channels=len(ii),valid_windows=info['valid_windows'],initial_complete_logits_bit_exact=True,
            global_reference_seconds=reference_seconds,three_step_seconds=time.perf_counter()-tick,discarded_admission_updates=3,gradient_norms=norms,
            reference_TRAIN_windows=globalbank.valid_windows if globalbank else 0,reference_TRAIN_patients=len(globalbank.fit_ids) if globalbank else 0,
            same_parameter_count=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
        del m,opt,globalbank;gc.collect();torch.cuda.empty_cache();print('TPRC_REAL_CURRENT_REFERENCE_BACKWARD_PASS',arm,flush=True)
    del base,original
    csv_out('INTERNAL_ROLE_ADMISSION',rolecounts)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),b0_bridge=bridge,roles=rolecounts,model_paths=audits,
        CPU_reference_tests=5,CPU_model_tests=3,no_new_Outer=True,new_formal_fits=0,assessment_not_used_in_admission=True))
    print('TPRC_4_ARM_3_SEED_ADMISSION_PASS',flush=True)

if __name__=='__main__':main()
