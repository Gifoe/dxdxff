"""Actual input, frozen-baseline, reference, and gradient admission."""
import subprocess,sys,time
import numpy as np
import torch
from scipy.special import expit
from runtime import *
from reference import fit_global
from pr2 import PR2
from drechan_adapter import DREChanAdapter,expose_raw96
from svm_adapter import probabilities,logits
from losses import objective

def main():
    tick=time.perf_counter()
    subprocess.run([sys.executable,str(HERE/'tests/test_pr2.py')],check=True)
    c=context();parent=c['parent'];rec=c['rec'];meta=rec['meta']
    assert rec['R14'].shape==(7635,14) and len(set(meta['patient']))==80
    # No waveform extraction/copies. All historical parent hashes verified by setup.
    input_binding=dict(parent=digest(c['binding']),archive=rec['binding']['archive'],
                       baseline_configs=sha(Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1\BASELINE_CONFIGS.json')))
    rows=[];outer_ids=[];baseline_hashes={}
    for fold in range(1,6):
        d=fold_bank(fold);data=bank_data(fold);tr,va=data['train'],data['val']
        roles=[set(data['patient'][tr]),set(data['patient'][va]),d['roles']['test']]
        assert all(roles[a].isdisjoint(roles[b]) for a,b in [(0,1),(0,2),(1,2)])
        assert [len(x) for x in roles]==[[51,51,50,52,51][fold-1],13,[16,16,17,15,16][fold-1]]
        outer_ids.extend(roles[2]);assert np.isfinite(data['raw12']).all()
        svm_errors=[]
        for pr in [False,True]:
            name='Linear_SVM_'+('PR' if pr else 'NO_PR');path=SVMROOT/'tabular_models_private'/name/f'fold{fold}.pkl'
            q=load(path);x=data['raw12'].astype(float)
            if pr:x=c['core'].patient_z(x,data['patient'])
            assert c['parent'].legacy.array_sha(x)==q['binding']['features'],'Historical feature bytes differ'
            p=probabilities(q['bundle'],x[va]);archived_p=q['validation_probability']
            error=float(np.max(np.abs(p-archived_p)));assert error<=1e-12
            grid=np.round(np.arange(0,1.0001,.005),3)
            assert np.array_equal(p[:,None]>=grid,archived_p[:,None]>=grid)
            for patient in sorted(set(data['patient'][va])):
                jj=data['patient'][va]==patient;channels=data['channel'][va][jj]
                assert np.array_equal(np.lexsort((channels,-p[jj])),np.lexsort((channels,-archived_p[jj])))
            svm_errors.append(error)
            assert np.allclose(expit(logits(q['bundle'],x[va])),p,atol=1e-15,rtol=1e-14)
            baseline_hashes[f'{name}_{fold}']=sha(path)
        base=base_model(fold);initial=state_hash(base.state_dict());base.eval()
        pid=sorted(roles[0])[0];ii=np.flatnonzero(data['patient']==pid);args=raw_arguments(data,ii,fold)
        with torch.no_grad():original=base(*args[:-1])['logit'].clone()
        wrapper=DREChanAdapter(base)
        with torch.no_grad():disabled=wrapper(*args,enabled=False)
        assert torch.equal(original,-disabled['final']) and disabled['h'].shape[1]==108
        assert state_hash(base.state_dict())==initial
        # Shared fixed FIT-only global statistics from the fresh original initializer.
        hh=[]
        with torch.no_grad():
            for patient in sorted(roles[0]):
                jj=np.flatnonzero(data['patient']==patient);o=wrapper(*raw_arguments(data,jj,fold),enabled=False)
                hh.append(o['h'].cpu());del o
        global108=fit_global(torch.cat(hh));del hh
        stats=dict(D12=fit_global(data['raw12'][tr]),D108=global108,initial_state=initial,
                   fit_patient_digest=digest(sorted(roles[0])),input=input_binding)
        path=ROOT/'references_private'/f'fold{fold}.pt'
        if path.exists():
            old=torch.load(path,weights_only=False)
            assert old['initial_state']==initial and old['input']==input_binding
            assert all(torch.equal(old[k][j],stats[k][j]) for k in ['D12','D108'] for j in ['mean','sd','unit','epsilon'])
        else:atomic_torch(path,stats)
        # Full historical validation replay, without exposing new Outer results.
        cell=next(r for r in parent.frozen()['cells'] if r['arm']=='FULL' and r['seed']==42 and r['fold']==fold)
        saved=torch.load(cell['checkpoint'],map_location='cpu',weights_only=False);base.load_state_dict(saved['model'])
        replay=parent.predict(base,d,va,c['inputs'],parent.scale(fold),fold,c['binding'])
        archived=load(cell['validation']);assert np.array_equal(replay,archived['logit'])
        baseline_hashes[f'DRECHAN_FULL_{fold}']=sha(cell['checkpoint'])
        # One complete real FIT-patient backward proves robust-reference gradient.
        fresh=base_model(fold);plugin=PR2(108,global108,seed=42+1009*fold+270007).cuda()
        with torch.no_grad():plugin.ranking[-1].weight.fill_(.01)
        joint=DREChanAdapter(fresh,plugin);joint.train();out=joint(*args)
        robust_gradient=torch.autograd.grad(out['delta_rank'].square().mean(),tuple(fresh.cnn.parameters()),retain_graph=True,allow_unused=True)
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in robust_gradient)
        y=torch.as_tensor(data['y'][ii],device='cuda',dtype=torch.float32)
        loss=objective(out,y,float(data['y'][tr].sum()/(1-data['y'][tr]).sum()),'drechan');loss.backward()
        cnn_grad=sum(float(p.grad.abs().sum()) for p in fresh.cnn.parameters() if p.grad is not None)
        assert cnn_grad>0 and torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in joint.parameters())
        rows.append(dict(fold=fold,fit_patients=len(roles[0]),validation_patients=13,outer_patients=len(roles[2]),
                         svm_feature_bytes_exact=True,svm_maximum_replay_error=max(svm_errors),svm_all201_thresholds_and_rankings_exact=True,
                         drechan_validation_bit_exact=True,wrapper_disabled_bit_exact=True,
                         reference_FIT_only=True,raw108_gradient=True,robust_residual_CNN_gradient=True,initial_state=initial,reference_sha256=sha(path)))
        del joint,fresh,base,wrapper,plugin,args,out;gc.collect();torch.cuda.empty_cache()
        write(PUBLIC/'ADMISSION_PROGRESS.json',dict(folds=len(rows),total=5))
        print('PR2_REAL_ADMISSION_FOLD_PASS',fold,flush=True)
    assert len(outer_ids)==len(set(outer_ids))==80
    input_binding['baselines']=baseline_hashes
    write(PUBLIC/'ADMISSION.json',dict(status='PASS',source=source_hashes(),input=input_binding,checks=rows,
         synthetic_tests=7,actual_patient_backwards=5,coverage_patients=80,coverage_channels=7635,
         seconds=time.perf_counter()-tick,outer_new_predictions_viewed=False))
    print('PR2_ADMISSION_PASS',flush=True)

if __name__=='__main__':main()
