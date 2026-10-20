"""Pre-fit strict10reuse/25initializer gates and5 discarded complete-patient paths."""
import copy
import re
import subprocess
from common import *

def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b

def main():
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],cwd=HERE,capture_output=True,text=True)
    print(test.stdout,end='');print(test.stderr,end='',file=sys.stderr);assert test.returncode==0
    ntests=int(re.search(r'Ran (\d+) tests',test.stderr)[1])
    binding,core,fac,records,rec,banks=setup();inputs=store();roles=[];reuse=[];initials=[]
    for fold in range(1,6):
        d=bank(fold,banks,'FULL');ss=scale(fold)
        counts=[len(set(d['patient'][d['train']])),len(set(d['patient'][d['val']])),len(d['roles']['test'])]
        assert counts==[[51,51,50,52,51][fold-1],13,[16,16,17,15,16][fold-1]]
        assert set(d['patient'][d['train']])==d['roles']['fit']
        assert set(d['patient'][d['val']]).isdisjoint(d['roles']['fit']|d['roles']['test'])
        roles.append(dict(fold=fold,fit_patients=counts[0],validation_patients=counts[1],test_patients=counts[2],
                          fit_channels=len(d['train']),validation_channels=len(d['val']),original_roles_exact=True))
        assert np.array_equal(query_features(rec['R14'][d['archive_indices']],d['patient'],d['pre'],'FULL',core),d['x'])
        base=parent.make(core,fac,records,fold,'FULL',42)
        initial={k:p.detach().cpu().clone() for k,p in base.named_parameters()};del base
        for arm in ARMS:
            ab=bank(fold,banks,arm);assert np.array_equal(ab['x'],d['x']) and preprocessor(ab,ss)==preprocessor(d,ss)
            m=make(core,fac,records,fold,arm,42)
            for name,p in m.named_parameters():
                if name=='attention_weight':assert torch.equal(p.detach().cpu(),torch.zeros(26));continue
                key=name if not name.startswith('cnn.') or m.early else name.replace('cnn.','cnn.stem.',1)
                actual=p.detach().cpu();expected=initial[key]
                if arm=='A2_CAPACITY_MATCHED':
                    if name=='head.0.weight':actual=actual[:64]
                    if name=='head.0.bias':actual=actual[:64]
                    if name=='head.2.weight':actual=actual[:,:64]
                assert torch.equal(actual,expected),(arm,fold,name)
            assert not m.late and not hasattr(m,'gate') and not m.interaction
            initials.append(dict(arm=arm,seed=42,fold=fold,initial_sha256=state_hash(m.state_dict()),shared_tensors_exact=True,
                                 late_gate_absent=True,parameters=MODEL.PARAMETERS[arm]))
            ref=reference(arm,42,fold)
            if ref is not None:
                r,state,v=ref;assert state_hash(m.state_dict())==v['binding']['initial']
                assert preprocessor(ab,ss)==r['preprocessor']==v['binding']['preprocessor']
                m.load_state_dict(state['model']);before=state_hash(m.state_dict());out=predict(m,ab,ab['val'],inputs,ss,fold,binding)
                assert np.array_equal(out,v['logit']) and state_hash(m.state_dict())==before
                assert np.array_equal(v['archive_indices'],ab['archive_indices'][ab['val']])
                selected=core.select_threshold(ab['y'][ab['val']],probability(out),ab['patient'][ab['val']])
                assert all(abs(float(z)-float(state[k]))<1e-12 for k,z in selected.items())
                winner=max(v['history'],key=lambda q:(q['macro_f1'],q['ez_f1'],-q['epoch']))
                assert winner['epoch']==r['epoch']==state['epoch'] and state['threshold']==r['threshold'] and 6<=len(v['history'])<=30
                q,score_sha,_=reference_score(r)
                reuse.append(dict(arm=arm,source_arm=r['source_arm'],seed=42,fold=fold,initial_state_exact=True,
                    preprocessing_exact=True,validation_logits_bit_exact=True,epoch_and_threshold_exact=True,
                    checkpoint_sha256=r['checkpoint_sha256'],validation_sha256=r['validation_sha256'],sealed_score_sha256=score_sha))
                del state,v
            del m;gc.collect();torch.cuda.empty_cache()
        print('PCA_REUSE_INITIALIZATION_PASS',fold,flush=True)
    assert len(reuse)==10 and len(initials)==25
    csv_out('ROLE_AUDIT',roles);csv_out('ADMISSION_REUSE',reuse);csv_out('INITIALIZATION_AUDIT',initials)
    fold=2;d=bank(fold,banks,'FULL');candidates=[]
    for pid in sorted(d['roles']['fit']):
        plan=inputs.raw.registry['plans'][str(pid)];assert sha(plan['path'])==plan['sha256']
        p=torch.load(plan['path'],weights_only=False,map_location='cpu');candidates.append((int(p['mask'][0].sum()),str(pid)));del p
    _,pid=max(candidates);ss=scale(fold);ix=d['train'][d['patient'][d['train']]==pid];audits=[]
    old=parent.make(core,fac,records,fold,'FULL',42);new=make(core,fac,records,fold,'FULL',42)
    assert state_hash(old.state_dict())==state_hash(new.state_dict());rng=core.rng_state();results=[]
    for m,fn in [(old,parent.forward),(new,forward)]:
        core.restore_rng(rng);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
        y=torch.as_tensor(d['y'][ix],device='cuda',dtype=torch.float32)
        pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda')
        logit,_=fn(m,d,ix,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos);loss.backward()
        gradients={n:p.grad.detach().clone() for n,p in m.named_parameters()};torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        results.append(dict(logit=logit.detach().clone(),loss=loss.detach().clone(),gradients=gradients,
            model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=core.rng_state()))
    assert equal_tree(*results);del old,new,results,opt,logit,loss,y,pos,gradients;gc.collect();torch.cuda.empty_cache()
    for arm in ARMS:
        m=make(core,fac,records,fold,arm,42);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
        y=torch.as_tensor(d['y'][ix],device='cuda',dtype=torch.float32)
        pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda')
        for step in range(2):
            opt.zero_grad(set_to_none=True);logit,info=forward(m,d,ix,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos)
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            assert m.head[0].weight.grad.norm()>0 and m.head[2].weight.grad.norm()>0
            assert dict(m.named_parameters())['cnn.stem.0.weight' if m.early else 'cnn.0.weight'].grad.norm()>0
            if arm=='P2_TIME_ATTENTION':assert m.attention_weight.grad.norm()>0
            if arm=='A2_CAPACITY_MATCHED':assert m.head[0].weight.grad[64:].norm()>0
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        audits.append(dict(arm=arm,fold=fold,channels=len(ix),valid_windows=info['valid_windows'],discarded_steps=2,finite_effective_gradients=True))
        del m,opt,logit,loss,y,pos;gc.collect();torch.cuda.empty_cache();print('PCA_PATIENT_PATH_PASS',arm,flush=True)
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),unique_tests=ntests,initialization_cells_checked=25,
        expected_reused_cells=10,expected_fresh_fits=15,all10_reuse_checks_bit_exact=True,
        main_forward_loss_allgradients_Adam_rng_bit_exact=True,fresh_path_admission=audits,new_formal_fits=0,
        no_new_Outer=True,no_performance_eligibility_gate=True))
    print('PCA_ADMISSION_PASS_NO_FORMAL_FITS',flush=True)

if __name__=='__main__':main()
