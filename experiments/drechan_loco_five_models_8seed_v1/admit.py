"""Outcome-free engineering admission and legal FIT-only backward probes."""
import copy
import subprocess
from types import FunctionType
from common import *
import timeconv_engine as engine

def direct(model,raw,mask,chunk,y,pos):
    r,_,chunks=engine.layout(raw,mask,False,chunk);hs=[]
    for ix in chunks:
        with engine.amp():hs.append(model.encode(r[ix]).float())
    channel=TC.aggregate(engine.dense(torch.cat(hs),mask),mask)
    with engine.amp():ll=model.classify(channel).float()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(ll,y,pos_weight=pos);loss.backward();return loss,ll

def main():
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v']);assert r.returncode==0
    binding,rec,banks=setup();
    from orchestrate import reuse_gate
    reuse_gate()
    # The historical NGA helper guards its own seeds42/44/62. Reuse its EXACT
    # code object in an isolated reference namespace for the new prespecified
    # seeds; never mutate the historical module or the actual model constructor.
    reference_source_for=FunctionType(LEG.source_for.__code__,
        dict(LEG.source_for.__globals__,SEEDS=[42,43,44,45,46,47,48,49]),LEG.source_for.__name__)
    assert reference_source_for.__code__ is LEG.source_for.__code__
    initials=[];roles=[];probes=[]
    import svm
    toy_rng=np.random.default_rng(42);toy_pid=np.repeat(np.arange(12).astype(str),4)
    toy_raw=toy_rng.normal(size=(48,14));toy_tr=np.arange(36);toy_x,toy_pre=feature_fit(toy_raw,toy_pid,toy_tr)
    toy=dict(raw=toy_raw,patient=toy_pid,y=np.tile([0,1,0,1],12).astype(np.float32),train=toy_tr,x=toy_x,pre=toy_pre)
    toy_bundle=svm.fit(toy,1);assert len(toy_bundle['calibration_audit'])==3
    assert np.isfinite(svm.predict(toy_bundle,toy_raw[36:],toy_pid[36:])).all()
    # Patient/source assignments and preprocessors independent of non-FIT values.
    for k,d in enumerate(banks,1):
        roles.append(dict(held_source=SOURCES[k-1],fit_patients=len(d['roles']['fit']),validation_patients=len(d['roles']['validation']),test_patients=len(d['roles']['test']),test_channels=int(np.isin(rec['meta']['patient'],sorted(d['roles']['test'])).sum()),source_patient_disjoint=True))
        changed=d['raw'].copy();changed[d['val']]=changed[d['val']]*137+123
        _,pre=feature_fit(changed,d['patient'],d['train']);assert digest(io.clean(pre))==digest(io.clean(d['pre']))
        assert np.array_equal(feature_query(d['raw'],d['patient'],d['pre']),d['x'])
        source=make_source(k)
        original=reference_source_for(io,SA.SourceA0,CTX['records'],k,TRAIN_SEED)
        assert state_hash(source.state_dict())==state_hash(original.state_dict())
        full=make('OURS_RAW_FEATURE',k);reference=LEG.MODEL.NoGuidanceF0(original,'FULL',TRAIN_SEED+1009*k+170003).cuda()
        assert state_hash(full.state_dict())==state_hash(reference.state_dict())
        for arm in ARMS:
            if arm=='LINEAR_SVM_FEATURE_PR':continue
            m=make(arm,k);m2=make(arm,k);assert state_hash(m.state_dict())==state_hash(m2.state_dict())
            if arm in OURS:
                if arm=='OURS_RAW':assert torch.equal(m.head[0].weight,full.head[0].weight[:,:96])
                elif arm=='OURS_FEATURE':assert torch.equal(m.head[0].weight,full.head[0].weight[:,96:])
                if arm!='OURS_FEATURE':
                    assert all(torch.equal(v,full.cnn.state_dict()[n]) for n,v in m.cnn.state_dict().items())
                assert not m.early and not m.late and not m.interaction
            initials.append(dict(arm=arm,held_source=SOURCES[k-1],parameters=sum(p.numel() for p in m.parameters()),initial_sha256=state_hash(m.state_dict()),scratch=True,shared_parts_exact=True))
            del m,m2;gc.collect();torch.cuda.empty_cache()
        del source,original,reference,full;gc.collect();torch.cuda.empty_cache()
    # Complete actual legal FIT patient, no channel/window subsampling.
    d=banks[0];tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    store=io.RawStore();raw,mask,times=store.get(pid,d['channel'][ii].astype(str).tolist());target=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    for arm in ['OURS_RAW_FEATURE','OURS_RAW','OURS_FEATURE']:
        m=make(arm,1).train();seed_all(TRAIN_SEED+1009);ll=forward(m,arm,d,ii,store);loss=LEG.MODEL.objective(ll,target,pos);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        m.eval()
        with torch.no_grad():
            if arm=='OURS_RAW':
                dirty=dict(d,x=np.full_like(d['x'],12345));assert torch.equal(forward(m,arm,d,ii,store),forward(m,arm,dirty,ii,store))
            elif arm=='OURS_RAW_FEATURE':
                dirty=dict(d,x=d['x'].copy());dirty['x'][:,8:10]=1e7
                assert torch.equal(forward(m,arm,d,ii,store),forward(m,arm,dirty,ii,store))
        probes.append(dict(arm=arm,complete_FIT_patient=True,channels=len(ii),valid_windows=int(mask.sum()) if arm!='OURS_FEATURE' else 0,finite_all_gradients=True,probe_state_discarded=True))
        del m;gc.collect();torch.cuda.empty_cache()
    # Synthetic direct/autocast graph with identical chunk boundaries vs two-pass.
    seed_all(TRAIN_SEED+1009);rr=torch.randn(1,5,3,500,device='cuda');mm=torch.ones(1,5,3,dtype=torch.bool,device='cuda');yy=torch.tensor([0.,1.,0.,1.,1.],device='cuda')
    a=make('TIMECONVCNN_RAW',1).train();b=copy.deepcopy(a);start=rng_state()
    restore_rng(start);la,za=direct(a,rr,mm,5,yy,pos);arng=rng_state()
    restore_rng(start);lb,zb=engine.forward(b,rr,mm,5,True,yy,pos);brng=rng_state()
    assert torch.equal(za,zb) and float(la)==lb
    errors=[]
    for n,p in a.named_parameters():
        other=dict(b.named_parameters())[n];assert torch.allclose(p.grad,other.grad,rtol=1e-3,atol=1e-5),n
        errors.append(float((p.grad-other.grad).abs().max()))
    assert all(torch.equal(v,dict(b.named_buffers())[n]) for n,v in a.named_buffers())
    assert torch.equal(arng['cpu'],brng['cpu']) and torch.equal(arng['cuda'],brng['cuda'])
    del a,b;gc.collect();torch.cuda.empty_cache()
    m=make('TIMECONVCNN_RAW',1).train();loss,ll=engine.forward(m,raw,mask,128,True,target,pos)
    assert np.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    probes.append(dict(arm='TIMECONVCNN_RAW',complete_FIT_patient=True,channels=len(ii),valid_windows=int(mask.sum()),finite_all_gradients=True,probe_state_discarded=True))
    del m;gc.collect();torch.cuda.empty_cache()
    registry=read(io.BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl');assert set(registry['plans'])==set(map(str,rec['meta']['patient']))
    windows=0;source_counts={s:0 for s in SOURCES}
    for pid in sorted(set(rec['meta']['patient'])):
        ix=np.flatnonzero(rec['meta']['patient']==pid);r,ms,t=store.get(pid,rec['meta']['channel'][ix].astype(str).tolist())
        assert torch.isfinite(r).all() and r.shape[-1]==500 and (ms.sum((0,2))>0).all()
        windows+=int(ms.sum());source_counts[str(rec['meta']['center'][ix[0]])]+=int(ms.sum())
    assert windows==1471965
    csv_out('ROLE_AUDIT',roles);csv_out('INITIALIZATION_AUDIT',initials)
    write(PUB/'INPUT_AUDIT.json',dict(status='PASS',patients=80,channels=7635,seizures=256,valid_windows=windows,
        valid_windows_by_source=source_counts,source_labels_not_proven_individual_hospitals=True,old_fitted_banks_read_or_reused=False,
        original_fivefold_roles_not_used=True,raw_caches_copied=False,feature_dimension=12,feature_slots=14))
    write(PUB/'ADMISSION.json',dict(status='PASS',binding=binding,source=sources(),initialization_cells=16,
        original_initializer_exact=True,all_nonFIT_preprocessor_perturbations_invariant=True,probes=probes,
        cache_direct_logits_loss_exact=True,gradient_cache_max_absolute_error=max(errors),gradient_cache_tolerance=dict(rtol=1e-3,atol=1e-5),
        BN_buffers_exact=True,RNG_exact=True,no_test_inference=True,discarded_probe_states=True))
    print('LOCO_ADMISSION_PASS',flush=True)

if __name__=='__main__':main()
