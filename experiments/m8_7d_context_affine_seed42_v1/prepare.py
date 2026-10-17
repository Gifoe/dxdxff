"""Exact complete-patient ownFIT/Development hidden cache; no own-Outer access."""
import gc
import hashlib
from common import *


def main():
    binding,core,fac,records,rec,banks,fr=setup();inputs=store();seals={};parts={};checks=[];coverage=[]
    for fold in range(1,6):
        d=bank(fold,banks);ss=scale(fold);m,r,state,old=body(fr,fold,core,fac,records);before=state_hash(m.state_dict())
        assert set(d['patient'])==d['roles']['fit']|d['roles']['validation'] and set(d['patient']).isdisjoint(d['roles']['test'])
        h=np.empty((len(d['patient']),64),np.float32);out=np.empty(len(h),np.float32);windows=0;channels=0
        bb=dict(input=binding,fold=fold,body_checkpoint=r['checkpoint_sha256'],archive_indices=legacy.array_sha(d['archive_indices']),input14=legacy.array_sha(d['x']),scale=legacy.array_sha(ss.cpu().numpy()))
        for pid in sorted(set(d['patient'])):
            ii=np.flatnonzero(d['patient']==pid);cb=dict(bank=bb,indices=legacy.array_sha(d['archive_indices'][ii]),input=legacy.array_sha(d['x'][ii]))
            token=hashlib.sha256(str(pid).encode()).hexdigest();path=ROOT/'patients_private'/f'fold{fold}'/(token+'.pkl')
            if path.exists():q=read(path);assert q['binding']==cb
            else:
                raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda')
                with torch.inference_mode():
                    z,_=m.features(raw,mask,times,delta,ss);hh=MODEL.hidden(m,z,x);p=m.classify(z,x)
                    assert torch.equal(m.head[2](hh).squeeze(-1),p)
                q=dict(binding=cb,hidden=hh.cpu().numpy(),reference_logit=p.cpu().numpy(),valid_windows=int(mask.sum()));save(path,q)
                del raw,mask,times,delta,x,z,hh,p
            assert q['hidden'].shape==(len(ii),64) and np.isfinite(q['hidden']).all() and np.isfinite(q['reference_logit']).all()
            h[ii]=q['hidden'];out[ii]=q['reference_logit'];windows+=q['valid_windows'];channels+=len(ii);parts[str(path.relative_to(ROOT))]=sha(path)
        assert state_hash(m.state_dict())==before and channels==len(h)
        va=d['val'];assert np.array_equal(old['archive_indices'],d['archive_indices'][va])
        error=float(np.max(np.abs(out[va]-old['logit'])));assert error<=1e-6
        grid=np.round(np.arange(0,1.0001,.005),3);assert np.array_equal(probability(out[va])[:,None]<grid,probability(old['logit'])[:,None]<grid)
        sel=core.select_threshold(d['y'][va],probability(out[va]),d['patient'][va]);assert all(abs(float(v)-float(state[k]))<1e-12 for k,v in sel.items())
        path=ROOT/'banks_private'/f'fold{fold}.pkl';obj=dict(binding=bb,archive_indices=d['archive_indices'],hidden=h,reference_logit=out,reference_threshold=state['threshold'],reference_epoch=state['epoch'],readout={k:v.detach().cpu().clone() for k,v in m.head[2].state_dict().items()})
        if path.exists():q=read(path);assert q['binding']==bb and np.array_equal(q['hidden'],h) and np.array_equal(q['reference_logit'],out)
        else:save(path,obj)
        seals[str(fold)]=sha(path);checks.append(dict(fold=fold,maximum_replay_error=error,all201_decisions_exact=True,body_unchanged=True))
        coverage.append(dict(fold=fold,FIT_patients=len(d['roles']['fit']),Development_patients=len(d['roles']['validation']),complete_patient_appearances=len(set(d['patient'])),channels=len(h),valid_windows=windows,own_Outer_patients=0))
        del m;gc.collect();torch.cuda.empty_cache();print('CONTEXT_AFFINE_REPRESENTATION_FOLD_COMPLETE',fold,flush=True)
    freeze(ROOT/'REPRESENTATION_SEAL_PRIVATE.json',dict(input=binding,banks=seals,patients=parts))
    pd.DataFrame(coverage).to_csv(PUB/'REPRESENTATION_COVERAGE.csv',index=False)
    write(PUB/'REPRESENTATION_STATUS.json',dict(status='PASS',folds=5,patient_appearances=sum(r['complete_patient_appearances'] for r in coverage),
        checks=checks,representation_seal_sha256=sha(ROOT/'REPRESENTATION_SEAL_PRIVATE.json'),no_labels_in_hidden_cache=True,new_Outer_passes=0))
    print('CONTEXT_AFFINE_ALL_REPRESENTATIONS_COMPLETE_NO_OUTER',flush=True)


if __name__=='__main__':main()
