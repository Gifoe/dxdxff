"""Explicit pre-fit CPU/CUDA bridge; original scientific sources are immutable."""
import argparse
import copy
import gc
import common as c
import train_internal as t
from head import Head, ARMS, groups, objective

ROOT=c.ROOT/'cpu_bridge_execution_v2'
PUB=ROOT/'public'
original_setup=c.setup
original_predict=t.predict


def setup():
    binding=original_setup()
    binding=dict(binding,execution={n:c.sha(c.HERE/n) for n in
                 ['execute_cpu_bridge_v2.py','CPU_BRIDGE_AMENDMENT.json']})
    c.freeze(ROOT/'EXECUTION_INPUT_FREEZE_PRIVATE.json',binding)
    c.torch.backends.cuda.matmul.allow_tf32=False
    c.torch.backends.cudnn.allow_tf32=False
    return binding


def native_predict(model,z,patient):
    m=copy.deepcopy(model).cuda().eval()
    out=c.np.empty(len(z),c.np.float32)
    with c.torch.inference_mode():
        for pid in sorted(set(patient)):
            ix=c.np.flatnonzero(patient==pid)
            out[ix]=m(c.torch.from_numpy(z[ix]).cuda(),groups(patient[ix]).cuda()).cpu().numpy()
    del m
    return out


def predict(model,z,patient):
    return native_predict(model,z,patient) if model.arm=='NATIVE' else original_predict(model,z,patient)


def admit():
    np,torch=c.np,c.torch
    binding=setup();reports=[]
    for ctx in c.contexts():
        z,cp=c.inputs(ctx);d=ctx['data'];rr=ctx['roles'];raw=z['1111'];fold=ctx['fold']
        models=[Head(a,cp['model'],fold) for a in ['REFIT','PATIENT','LOCAL']]
        scores=[original_predict(m,raw,d['patient']) for m in models]
        assert np.array_equal(scores[0],scores[1]) and np.array_equal(scores[0],scores[2])
        reference=ctx['stored']['raw_logit_nez']
        exact=native_predict(Head('NATIVE',cp['model'],fold),raw,d['patient'])
        gpu_error=float(np.max(np.abs(exact-reference)));assert gpu_error<=1e-6
        cpu_error=float(np.max(np.abs(scores[0]-reference)))
        pe=float(np.max(np.abs(c.probability(scores[0])-c.probability(reference))));assert pe<=1e-6
        cal=rr['cal'];grid=np.round(np.arange(0,1.0001,.005),3)
        a=c.probability(scores[0][cal]);b=c.probability(reference[cal])
        assert np.array_equal(a[:,None]>=grid,b[:,None]>=grid)
        assert t.select(d,cal,scores[0][cal])['threshold']==cp['threshold']
        assert t.select(d,cal,exact[cal])['threshold']==cp['threshold']
        pid=sorted(set(d['patient']))[0];ix=np.flatnonzero(d['patient']==pid)
        test=copy.deepcopy(models[1])
        with torch.no_grad():test.conditioner.weight.fill_(.001)
        full=original_predict(test,raw,d['patient']);one=original_predict(test,raw[ix],d['patient'][ix])
        isolation=float(np.max(np.abs(full[ix]-one)));assert isolation<=1e-6
        if not reports:
            tr=np.r_[rr['train'],rr['choice']];g=groups(d['patient'][tr]);y=torch.as_tensor(d['y'][tr],dtype=torch.float32)
            pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()))
            native=models[0];other=copy.deepcopy(native)
            score=native(torch.from_numpy(raw[tr]),g);loss=objective(score,y,g,pos);loss.backward()
            ss=other(torch.from_numpy(raw[tr]),g)
            explicit=torch.stack([objective(ss[g==j],y[g==j],torch.zeros(int((g==j).sum()),dtype=torch.long),pos)
                                  for j in range(int(g.max())+1)]).mean()
            explicit.backward();maximum=0.
            assert abs(float(loss.detach())-float(explicit.detach()))<=1e-6
            for p,q in zip(native.parameters(),other.parameters()):
                maximum=max(maximum,float((p.grad-q.grad).abs().max()))
                assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5)
            capacity={a:sum(p.numel() for p in Head(a,cp['model'],fold).parameters()) for a in ARMS}
            assert capacity['REFIT']==10369 and capacity['PATIENT']==capacity['LOCAL']==14097
        reports.append(dict(fold=fold,screen=ctx['screen'],native_CUDA_logit_max_error=gpu_error,
                            CPU_logit_bridge_error_recorded_not_strict_pass=cpu_error,CPU_probability_error=pe,
                            all_calibration_grid_decisions_identical=True,zero_conditioner_identity=True,
                            complete_patient_context_isolation_error=isolation,role_and_source_hashes_verified=True))
        print('MACRO_HEAD_BRIDGE_ADMISSION',fold,ctx['screen'],gpu_error,pe,flush=True)
        del z,cp,models;gc.collect()
    c.write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=c.digest(binding),paths=len(reports),reports=reports,
        parameters=capacity,vectorized_patient_loss_gradient_max_error=maximum,original_CPU_logit_admission_failed=True,
        no_new_Raw_encoder_fits=True,only96_Raw_inputs=True,no_original_Development_or_Outer_used=True,
        execution_amendment_sha256=c.sha(c.HERE/'CPU_BRIDGE_AMENDMENT.json')))
    print('MACRO_HEAD_ALL15_BRIDGE_ADMISSION_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['admit','train','summarize']);a=parser.parse_args()
    c.ROOT=t.ROOT=ROOT;c.PUB=t.PUB=PUB;c.setup=t.setup=setup;t.predict=predict
    if a.action=='admit':admit()
    elif a.action=='train':t.train()
    else:
        import summarize_internal as summary
        summary.main()


if __name__=='__main__':main()
