"""Real-server admission before205fresh fits; exact50reuses and255initializers."""
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
    t=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],cwd=HERE,capture_output=True,text=True)
    print(t.stdout,end='');print(t.stderr,end='',file=sys.stderr);assert t.returncode==0
    ntests=int(re.search(r'Ran (\d+) tests',t.stderr)[1])
    binding,core,fac,records,rec,banks=setup();inputs=store();roles=[];reuse=[];initials=[]
    pool_source=Path(r'D:\drechan_pooling_capacity_ablation_seed42_v1\model.py')
    assert sha(pool_source)==CONTEXT['freezes']['poolcap']['source']['model.py']
    pool_math=module('nga_admission_capacity_legacy',pool_source)
    sys.modules['common']=OWN;sys.modules['model']=MODEL;sys.path.insert(0,str(HERE))
    for fold in range(1,6):
        d=bank(fold,banks,'FULL');ss=scale(fold)
        counts=[len(set(d['patient'][d['train']])),len(set(d['patient'][d['val']])),len(d['roles']['test'])]
        assert counts==[[51,51,50,52,51][fold-1],13,[16,16,17,15,16][fold-1]]
        assert set(d['patient'][d['train']])==d['roles']['fit'] and set(d['patient']).isdisjoint(d['roles']['test'])
        assert set(d['patient'][d['val']]).isdisjoint(d['roles']['fit']|d['roles']['test'])
        roles.append(dict(fold=fold,fit_patients=counts[0],validation_patients=counts[1],test_patients=counts[2],
                          fit_channels=len(d['train']),validation_channels=len(d['val']),original_roles_exact=True))
        for arm in ARMS:
            ab=bank(fold,banks,arm)
            assert np.array_equal(query_features(rec['R14'][ab['archive_indices']],ab['patient'],ab['pre'],arm,core),ab['x'])
        for seed in SEEDS:
            source=source_for(core,fac,records,fold,seed)
            base=MODEL.NoGuidanceF0(source,'FULL',seed+1009*fold+170003)
            expected={k:v.detach().clone() for k,v in base.named_parameters()};del base
            for arm in ARMS:
                rng=core.rng_state();m=MODEL.NoGuidanceF0(source,arm,seed+1009*fold+170003).cuda()
                after=core.rng_state()
                if arm=='CAPACITY_MATCHED':
                    # Preserve the exact historical recipe: its CPU fork calls
                    # torch.manual_seed, which also reseeds CUDA. Training has
                    # ALWAYS seed_all(training_seed) immediately after make.
                    assert all(equal_tree(rng[k],after[k]) for k in ['python','numpy','torch'])
                    core.restore_rng(rng)
                    capacity_old=pool_math.PoolingCapacity(source,'A2_CAPACITY_MATCHED',seed+1009*fold+170003).cuda()
                    assert state_hash(m.state_dict())==state_hash(capacity_old.state_dict())
                    assert equal_tree(after,core.rng_state())
                    core.seed_all(seed+1009*fold);reset=core.rng_state()
                    core.restore_rng(after);core.seed_all(seed+1009*fold)
                    assert equal_tree(reset,core.rng_state())
                    del capacity_old
                else:assert equal_tree(rng,after)
                for name,p in m.named_parameters():
                    if 'conditioner' in name or name=='attention_weight':continue
                    key=name.replace('cnn.stem.','cnn.') if m.early else name
                    actual=p;target=expected[key]
                    if arm=='CAPACITY_MATCHED':
                        if name=='head.0.weight':actual=actual[:64]
                        if name=='head.0.bias':actual=actual[:64]
                        if name=='head.2.weight':actual=actual[:,:64]
                    if arm=='RAW_ONLY' and name=='head.0.weight':target=target[:,:96]
                    if arm=='PHYSIOLOGY_ONLY' and name=='head.0.weight':target=target[:,96:]
                    assert torch.equal(actual,target),(arm,seed,fold,name)
                assert not m.late and not hasattr(m,'gate') and not m.interaction and m.early==(arm=='EARLY_FILM_RESTORED')
                initials.append(dict(arm=arm,seed=seed,fold=fold,initial_sha256=state_hash(m.state_dict()),
                    shared_tensors_exact=True,late_absent=True,training_rng_reset_exact=True,
                    initializer_CUDA_rng_matches_legacy=True,parameters=MODEL.PARAMETERS[arm]))
                ref=reference(arm,seed,fold)
                if ref is not None:
                    r,state,v=ref;ab=bank(fold,banks,arm)
                    assert state_hash(m.state_dict())==v['binding']['initial']
                    assert preprocessor(ab,ss)==r['preprocessor']==v['binding']['preprocessor']
                    m.load_state_dict(state['model']);before=state_hash(m.state_dict())
                    out=predict(m,ab,ab['val'],inputs,ss,fold,binding)
                    assert np.array_equal(out,v['logit']) and state_hash(m.state_dict())==before
                    assert np.array_equal(v['archive_indices'],ab['archive_indices'][ab['val']])
                    selected=core.select_threshold(ab['y'][ab['val']],probability(out),ab['patient'][ab['val']])
                    assert all(abs(float(z)-float(state[k]))<1e-12 for k,z in selected.items())
                    winner=max(v['history'],key=lambda z:(z['macro_f1'],z['ez_f1'],-z['epoch']))
                    assert winner['epoch']==r['epoch']==state['epoch'] and state['threshold']==r['threshold'] and 6<=len(v['history'])<=30
                    q,sh,_=reference_score(r)
                    reuse.append(dict(arm=arm,seed=seed,fold=fold,reuse_kind=r['reuse_kind'],source_arm=r['source_arm'],
                        initial_preprocessing_validation_epoch_threshold_exact=True,checkpoint_sha256=r['checkpoint_sha256'],
                        validation_sha256=r['validation_sha256'],sealed_score_sha256=sh))
                    del state,v,q
                del m;gc.collect();torch.cuda.empty_cache()
            del source,expected;gc.collect();torch.cuda.empty_cache()
            print('NGA_INITIALIZATION_REUSE_PASS',seed,fold,flush=True)
    assert len(initials)==255 and len(reuse)==50
    csv_out('ROLE_AUDIT',roles);csv_out('INITIALIZATION_AUDIT',initials);csv_out('ADMISSION_REUSE',reuse)
    fold=2;d=bank(fold,banks,'FULL');ss=scale(fold);candidates=[]
    for pid in sorted(d['roles']['fit']):
        p=inputs.raw.registry['plans'][str(pid)];assert sha(p['path'])==p['sha256']
        q=torch.load(p['path'],weights_only=False,map_location='cpu');candidates.append((int(q['mask'][0].sum()),str(pid)));del q
    _,pid=max(candidates);ix=d['train'][d['patient'][d['train']]==pid];parity=[];capacity_parity=[];paths=[]
    for seed in SEEDS:
        source=source_for(core,fac,records,fold,seed)
        old=MODEL.BASE.NoLateF0(source,'WO_EARLY_FILM').cuda();new=MODEL.NoGuidanceF0(source,'FULL').cuda();del source
        rng=core.rng_state();results=[]
        for m in [old,new]:
            core.restore_rng(rng);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
            y=torch.as_tensor(d['y'][ix],device='cuda',dtype=torch.float32)
            pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda')
            logit,info=forward(m,d,ix,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            grads={n:p.grad.detach().clone() for n,p in m.named_parameters()}
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            results.append(dict(logit=logit.detach().clone(),loss=loss.detach().clone(),gradients=grads,
                model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=core.rng_state()))
        assert equal_tree(*results);parity.append(dict(seed=seed,fold=fold,channels=len(ix),valid_windows=info['valid_windows'],
            old_noEarly_forward_loss_allgradients_clippedAdam_rng_exact=True,discarded_steps_each=1))
        del old,new,m,opt,logit,loss,y,pos,grads,results;gc.collect();torch.cuda.empty_cache()
        source=source_for(core,fac,records,fold,seed)
        old=pool_math.PoolingCapacity(source,'A2_CAPACITY_MATCHED',seed+1009*fold+170003).cuda()
        new=MODEL.NoGuidanceF0(source,'CAPACITY_MATCHED',seed+1009*fold+170003).cuda();del source
        results=[]
        for m in [old,new]:
            core.seed_all(seed+1009*fold);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
            y=torch.as_tensor(d['y'][ix],device='cuda',dtype=torch.float32)
            pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda')
            logit,_=forward(m,d,ix,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            grads={n:p.grad.detach().clone() for n,p in m.named_parameters()}
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            results.append(dict(logit=logit.detach().clone(),loss=loss.detach().clone(),gradients=grads,
                model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=core.rng_state()))
        assert equal_tree(*results)
        capacity_parity.append(dict(seed=seed,fold=fold,legacy_constructor_and_training_rng_reset_exact=True,
            forward_loss_allgradients_clippedAdam_rng_exact=True,discarded_steps_each=1))
        del old,new,m,opt,logit,loss,y,pos,grads,results;gc.collect();torch.cuda.empty_cache()
    for arm in ARMS:
        ab=bank(fold,banks,arm);m=make(core,fac,records,fold,arm,42);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
        ii=ab['train'][ab['patient'][ab['train']]==pid];y=torch.as_tensor(ab['y'][ii],device='cuda',dtype=torch.float32)
        pos=torch.tensor(float((1-ab['y'][ab['train']]).sum()/ab['y'][ab['train']].sum()),device='cuda')
        for step in range(2):
            opt.zero_grad(set_to_none=True);logit,info=forward(m,ab,ii,inputs,ss,fold,binding);loss=MODEL.objective(logit,y,pos)
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),arm
            assert m.head[2].weight.grad.norm()>0
            if m.branch!='phys':assert dict(m.named_parameters())['cnn.stem.0.weight' if m.early else 'cnn.0.weight'].grad.norm()>0
            if arm=='P2_TIME_ATTENTION':assert m.attention_weight.grad.norm()>0
            if arm=='CAPACITY_MATCHED':assert m.head[0].weight.grad[64:].norm()>0
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        paths.append(dict(arm=arm,seed=42,fold=fold,channels=len(ii),valid_windows=info['valid_windows'],discarded_steps=2,finite_effective_gradients=True))
        del m,opt,y,pos,logit,loss;gc.collect();torch.cuda.empty_cache();print('NGA_PATIENT_PATH_PASS',arm,flush=True)
    support=0;fits=set();vals=set();support_folds=[]
    for fold in range(1,6):
        d=bank(fold,banks,'P2_TIME_ATTENTION');ss=scale(fold);m=make(core,fac,records,fold,'P2_TIME_ATTENTION',42);m.train();nfit=nval=0
        for role,indices in [('FIT',d['train']),('VALIDATION',d['val'])]:
            for pid in sorted(set(d['patient'][indices])):
                ii=indices[d['patient'][indices]==pid];m.zero_grad(set_to_none=True)
                logit,_=forward(m,d,ii,inputs,ss,fold,binding);assert torch.isfinite(logit).all()
                logit.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
                if role=='FIT':fits.add(str(pid));nfit+=1
                else:vals.add(str(pid));nval+=1
                support+=1;del logit
        support_folds.append(dict(fold=fold,FIT_complete_patients=nfit,validation_complete_patients=nval,zero_updates=True))
        del m;gc.collect();torch.cuda.empty_cache();print('NGA_P2_COMPLETE_SUPPORT_PASS',fold,flush=True)
    assert support==320 and len(fits)==79 and len(vals)==47 and len(fits|vals)==80
    csv_out('POOLING_SUPPORT_ADMISSION',support_folds)
    write(PUB/'ALIAS_AUDIT.json',dict(status='ALIASES_NOT_INDEPENDENT_HYPOTHESES',aliases=MODEL.ALIASES,
        mismatch_not_applicable_without_FiLM=True,no_Early_Late_interaction=True))
    write(PUB/'ADMISSION.json',dict(status='PASS',source=sources(),unique_tests=ntests,initialization_cells_checked=255,
        expected_reused_cells=50,expected_fresh_fits=205,all50_reuse_checks_bit_exact=True,main_complete_patient_parity=parity,
        capacity_legacy_parity=capacity_parity,discarded_complete_patient_paths=paths,
        P2_complete_support_label_free_backward_checks=320,P2_unique_patients=80,
        no_new_Outer=True,new_formal_fits=0,seed_set_outcome_informed=True,no_performance_gate=True))
    print('NGA_ADMISSION_PASS_NO_FORMAL_FITS',flush=True)
if __name__=='__main__':main()
