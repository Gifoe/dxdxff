"""Outcome-free runtime migration audit; all one-step audit weights discarded.

Baseline/compare operate on the exact preserved epoch6 checkpoint. Never update
formal model files, score ASSESS or Outer, alter selection or loosen tolerances.
"""
import sys
from common import *

def optimizer_tensors(opt):
    d=opt.state_dict();result={}
    for k,v in d['state'].items():
        for name,t in v.items():
            if isinstance(t,torch.Tensor):result[f'{k}/{name}']=t.detach().cpu().clone()
            else:result[f'{k}/{name}']=torch.tensor(t)
    return result

def main(mode):
    binding,core,fac,records,rec,banks=setup();assert binding==jr(ROOT/'INPUT_FREEZE_PRIVATE.json');inputs=store()
    path=ROOT/'native_backup01_B2_seed43_fold3_epoch6_PRIVATE.pt';q=torch.load(path,weights_only=False,map_location='cpu')
    assert sha(path)=='80a21e7169e3384c1cb51f016db2c7f25dfae5a577d5360c1df7adde470f3cfe'
    d,scale=data(3,'internal',core,rec,banks,inputs,binding);m=make(core,fac,records,3,'B2',43)
    assert state_hash(m.state_dict())==q['binding']['initial'];m.load_state_dict(q['model']);g=MODEL.GlobalReference(m,d,inputs,scale)
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng']);m.train()
    tr=d['train'];groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
    k=np.random.default_rng(43+1009*3+7).permutation(len(groups))[0];ii=groups[int(k)]
    ref=g.fit(m);m.set_reference(ref);opt.zero_grad(set_to_none=True)
    y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32);pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    out,_=forward(m,d,ii,inputs,scale);loss=MODEL.objective(out,y,pos);loss.backward()
    grads={k:p.grad.detach().cpu().clone() for k,p in m.named_parameters()};assert all(torch.isfinite(v).all() for v in grads.values())
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();torch.cuda.synchronize()
    snapshot=dict(logit=out.detach().cpu(),loss=loss.detach().cpu(),gradients=grads,model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},
        optimizer=optimizer_tensors(opt),rng=core.rng_state(),reference=reference_copy(ref))
    private=ROOT/'RUNTIME_BASELINE_ONE_STEP_PRIVATE.pt'
    if mode=='baseline':
        assert not private.exists();atomic_torch(private,snapshot);print('READ_ONLY_FORMAL_STATE_BASELINE_CAPTURED',flush=True);return
    assert mode=='compare';baseline=torch.load(private,weights_only=False,map_location='cpu')
    assert torch.equal(snapshot['logit'],baseline['logit']) and torch.equal(snapshot['loss'],baseline['loss'])
    for name in ['gradients','model','optimizer','reference']:
        assert set(snapshot[name])==set(baseline[name])
        assert all(torch.equal(v,baseline[name][k]) for k,v in snapshot[name].items()),name
    save(ROOT/'RUNTIME_RNG_COMPARE_PRIVATE.pkl',snapshot['rng'])
    # core RNG serialization can contain numpy arrays/tuples; canonical pickle
    # comparison is replaced by exact recursive value/tensor comparison.
    def same(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    assert same(snapshot['rng'],baseline['rng'])
    m.load_state_dict(q['best']['model']);assert reference_hash(g.fit(m))==reference_hash(q['best']['reference']);del m,opt,g
    checks=[];prechecks=[]
    for fold in range(1,6):
        dd,ss=data(fold,'internal',core,rec,banks,inputs,binding)
        x,pre=core.prepare(rec['R14'][dd['archive_indices']],dd['patient'],dd['train'])
        assert np.array_equal(x,dd['x'])
        assert all(np.array_equal(np.asarray(v),np.asarray(dd['pre'][k])) for k,v in pre.items())
        magnitudes=[]
        for pid in sorted(dd['roles']['train']):
            j=dd['train'][dd['patient'][dd['train']]==pid];raw,mask,times,delta,_=inputs.get(str(pid),dd['channel'][j].astype(str).tolist(),'cpu')
            magnitudes.append(np.median(np.abs(delta[mask].numpy()),axis=0));del raw,delta
        s=np.maximum(np.median(np.stack(magnitudes),axis=0),1e-30).astype(np.float32);assert np.array_equal(s,ss.cpu().numpy())
        prechecks.append(dict(fold=fold,TRAIN_preprocessing_bit_exact=True,TRAIN_scale_bit_exact=True))
    with (PUB/'internal/TRAIN_CELLS.csv').open(newline='',encoding='utf-8') as f:cells=list(csv.DictReader(f))
    assert len(cells)==30
    for r in cells:
        arm,seed,fold=r['arm'],int(r['seed']),int(r['fold']);cell=ROOT/'models_private/internal'/arm/f'seed{seed}'/f'fold{fold}'
        dd,ss=data(fold,'internal',core,rec,banks,inputs,binding);m=make(core,fac,records,fold,arm,seed)
        state=torch.load(cell/'BEST_PRIVATE.pt',weights_only=False,map_location='cpu');v=read(cell/'CALIBRATION_PRIVATE.pkl')
        assert state_hash(m.state_dict())==v['binding']['initial'];m.load_state_dict(state['model']);m.set_reference(state['reference'])
        logits,_=predict(m,dd,dd['val'],inputs,ss);assert np.array_equal(logits,v['logit'])
        sel=core.select_threshold(dd['y'][dd['val']],probability(logits),dd['patient'][dd['val']]);assert all(float(a)==float(state[k]) for k,a in sel.items())
        if arm=='B2':
            gg=MODEL.GlobalReference(m,dd,inputs,ss);assert reference_hash(gg.fit(m))==reference_hash(state['reference']);del gg
        checks.append(dict(arm=arm,seed=seed,fold=fold,initial_state_bit_exact=True,CAL_logits_bit_exact=True,threshold_bit_exact=True))
        del m;gc.collect();torch.cuda.empty_cache();print('COMPLETED_CELL_PARITY_PASS',arm,seed,fold,flush=True)
    # Five original full-FIT B0 validation references, without new Outer inference.
    oldbridge=[]
    for fold in range(1,6):
        dd=parent.bank(fold,banks,'FULL_F0');ss=parent.scale(fold);m=make(core,fac,records,fold,'B0',42)
        r=next(z for z in jr(PARENTROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')['cells'] if z['arm']=='FULL_F0' and z['fold']==fold)
        state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');v=read(r['validation']);m.load_state_dict(state['model'])
        logits,_=predict(m,dd,dd['val'],inputs,ss);assert np.array_equal(logits,v['logit']);oldbridge.append(dict(fold=fold,original_validation_bit_exact=True))
        del m;gc.collect();torch.cuda.empty_cache()
    result=dict(status='PASS',source=sources(),preserved_checkpoint_sha256=sha(path),formal_source_changed=False,all30_completed_cells_exact=checks,
        original_b0_bridge=oldbridge,all5_TRAIN_preprocessing_exact=prechecks,one_complete_update_logits_loss_all_gradients_model_Adam_RNG_bit_exact=True,
        current_and_saved_selected_global_references_bit_exact=True,maximum_gradient_difference=0,discarded_audit_optimizer_updates=1,
        no_formal_checkpoint_changed=True,no_new_assessment_or_Outer=True,python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,
        numpy_backend='isolated pip NumPy OpenBLAS, not inherited Anaconda MKL',same_original_Torch_GPU_binaries=True)
    write(PUB/'ISOLATED_RUNTIME_PARITY_01.json',result);print('ISOLATED_RUNTIME_EXACT_RESUME_ADMISSION_PASS',flush=True)

if __name__=='__main__':main(sys.argv[1])
