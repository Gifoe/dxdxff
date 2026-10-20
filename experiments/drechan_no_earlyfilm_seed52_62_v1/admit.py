"""Outcome-free server admission: all10 initializers and both full-patient paths."""
import copy
import re
import subprocess
from common import *
def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)is type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return type(a)is type(b) and a==b
def main():
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],cwd=HERE,capture_output=True,text=True)
    print(test.stdout,end='');print(test.stderr,end='',file=sys.stderr);assert test.returncode==0
    ntests=int(re.search(r'Ran (\d+) tests',test.stderr)[1])
    binding,core,fac,records,rec,banks=setup();inputs=store();roles=[];initials=[]
    for fold in range(1,6):
        d=bank(fold,banks,ARMS[0]);ss=scale(fold)
        counts=[len(set(d['patient'][d['train']])),len(set(d['patient'][d['val']])),len(d['roles']['test'])]
        assert counts==[[51,51,50,52,51][fold-1],13,[16,16,17,15,16][fold-1]]
        assert set(d['patient'][d['train']])==d['roles']['fit']
        assert set(d['patient'][d['val']])==d['roles']['val'] if 'val' in d['roles'] else set(d['patient'][d['val']]).isdisjoint(d['roles']['fit']|d['roles']['test'])
        assert set(d['patient']).isdisjoint(d['roles']['test'])
        assert np.array_equal(query_features(rec['R14'][d['archive_indices']],d['patient'],d['pre'],ARMS[0],core),d['x'])
        roles.append(dict(fold=fold,fit_patients=counts[0],validation_patients=counts[1],test_patients=counts[2],
                          fit_channels=len(d['train']),validation_channels=len(d['val']),original_roles_exact=True))
        for seed in SEEDS:
            source=source_for(core,fac,records,fold,seed)
            old=original.MODEL.CompleteF0(source,'WO_BOTH_GUIDANCE')
            rng=core.rng_state();m=MODEL.NoEarlyF0(source,ARMS[0]);assert equal_tree(rng,core.rng_state())
            assert state_hash(old.state_dict())==state_hash(m.state_dict())
            first=state_hash(m.state_dict());del source,old,m
            m=make(core,fac,records,fold,ARMS[0],seed);assert state_hash(m.state_dict())==first
            assert not m.early and not m.late and not m.guidance_pr and not m.interaction
            assert m.feature_pr and m.raw_pr and m.pre_reference and m.head[0].out_features==64
            assert not hasattr(m,'gate') and sum(p.numel() for p in m.parameters())==9329
            initials.append(dict(arm=ARMS[0],seed=seed,fold=fold,initial_sha256=first,
                old_no_both_guidance_graph_exact=True,repeat_initialization_exact=True,parameters=9329))
            del m;gc.collect();torch.cuda.empty_cache()
        print('NEL2_INITIALIZERS_PASS',fold,flush=True)
    assert len(initials)==10 and len({r['initial_sha256'] for r in initials})==10
    csv_out('ROLE_AUDIT',roles);csv_out('INITIALIZATION_AUDIT',initials)
    fold=2;d=bank(fold,banks,ARMS[0]);candidates=[]
    for pid in sorted(d['roles']['fit']):
        plan=inputs.raw.registry['plans'][str(pid)];assert sha(plan['path'])==plan['sha256']
        p=torch.load(plan['path'],weights_only=False,map_location='cpu');candidates.append((int(p['mask'][0].sum()),str(pid)));del p
    _,pid=max(candidates);ix=d['train'][d['patient'][d['train']]==pid];ss=scale(fold);audits=[]
    for seed in SEEDS:
        source=source_for(core,fac,records,fold,seed)
        old=original.MODEL.CompleteF0(source,'WO_BOTH_GUIDANCE').cuda()
        new=MODEL.NoEarlyF0(source,ARMS[0]).cuda();del source
        assert state_hash(old.state_dict())==state_hash(new.state_dict())
        rng=core.rng_state();results=[];info={}
        for m,fn in [(old,original.forward),(new,forward)]:
            core.restore_rng(rng);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
            y=torch.as_tensor(d['y'][ix],device='cuda',dtype=torch.float32)
            pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda')
            logit,info=fn(m,d,ix,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            assert m.cnn[0].weight.grad.norm()>0 and m.head[0].weight.grad[:,:96].norm()>0 and m.head[0].weight.grad[:,96:].norm()>0
            grads={n:p.grad.detach().clone() for n,p in m.named_parameters()}
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            results.append(dict(logit=logit.detach().clone(),loss=loss.detach().clone(),gradients=grads,
                model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=core.rng_state()))
        assert equal_tree(*results)
        audits.append(dict(seed=seed,fold=fold,channels=len(ix),valid_windows=info['valid_windows'],
            forward_loss_allgradients_clippedAdam_rng_bit_exact=True,discarded_steps_each=1))
        del old,new,m,opt,logit,loss,y,pos,grads,results;gc.collect();torch.cuda.empty_cache()
        print('NEL2_FULL_PATIENT_PARITY_PASS',seed,flush=True)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),unique_tests=ntests,
        initialization_cells_checked=10,expected_reused_cells=0,expected_fresh_fits=10,
        discarded_full_patient_paths=audits,new_formal_fits=0,no_new_Outer=True,no_performance_gate=True))
    print('NEL2_ADMISSION_PASS_NO_FORMAL_FITS',flush=True)
if __name__=='__main__':main()
