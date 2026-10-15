"""One frozen Outer pass per backbone, full patient-score/latent sealing first."""
import gc
from support import *
from engine import forward
from aggregate import aggregate_results


def run(a):
    lock,core,rec,banks,binding=setup(a.runtime)
    mf=jr(a.runtime/a.model/'MODEL_FREEZE_PRIVATE.json');assert mf['binding']==binding and len(mf['selections'])==10
    freeze=dict(binding=binding,selections=mf['selections'],new_full_outer_label_read=False)
    fp=a.runtime/a.model/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if fp.exists():assert jr(fp)==freeze
    else:write(fp,freeze)
    fh=sha(fp);pub=a.runtime/'public';status=pub/f'{a.model}_OUTER_STATUS.json'
    if status.exists():
        assert jr(status)['freeze_sha256']==fh;print('RAW_OUTER_ALREADY_COMPLETE',a.model,flush=True);return
    store=Store(BASELINE);meta=rec['meta'];files={};cells={}
    for r in freeze['selections']:
        version,fold=r['version'],r['fold'];rr=roles(meta,fold)
        ix=np.flatnonzero(np.isin(meta['patient'],list(rr['test'])))
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['epoch']==r['epoch'] and cp['selection']['threshold']==r['threshold']
        done=jr(Path(r['checkpoint']).parent/'COMPLETE_PRIVATE.json');assert done['best_sha256']==sha(r['checkpoint'])
        model=build(a.model,version=='PR',fold,lock,core).eval();model.load_state_dict(cp['model'],strict=True)
        score=np.empty(len(ix),np.float32)
        for n,pid in enumerate(sorted(rr['test']),1):
            jj=np.flatnonzero(meta['patient'][ix]==pid);gix=ix[jj]
            token=hashlib.sha256(str(pid).encode()).hexdigest()[:24]
            pp=a.runtime/a.model/f'outer_private/{version}/fold{fold}/{token}.pkl'
            if pp.exists():q=read(pp)
            else:
                with torch.inference_mode():
                    raw,mask=raw_patient(store,pid,meta['channel'][gix]);ll,h=forward(model,raw,mask,lock['chunk'][a.model]);p=1-torch.sigmoid(ll).cpu().numpy()
                q=dict(logit_nez=ll.cpu().numpy(),probability_ez=p,channel_embedding=h.cpu().numpy(),indices=gix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(pp,q)
            assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(q['indices'],gix) and 'y' not in q
            assert np.isfinite(q['probability_ez']).all() and np.isfinite(q['channel_embedding']).all()
            score[jj]=q['probability_ez'];files[f'{version}_{fold}_{n}']=sha(pp)
        assert statehash(model.state_dict())==statehash(cp['model'])
        save(a.runtime/a.model/f'outer_private/{version}_fold{fold}.pkl',dict(score_ez=score,indices=ix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']))
        cells[version,fold]=score
        print('RAW_OUTER_CELL_SEALED',a.model,version,fold,len(ix),flush=True)
        del model,cp;gc.collect();torch.cuda.empty_cache()
    sp=a.runtime/a.model/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';obj=dict(freeze_sha256=fh,files=files,all160_patient_cells_sealed_before_full_labels=True)
    if sp.exists():assert jr(sp)==obj
    else:write(sp,obj)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and y.sum()==1743
    frames=[]
    for r in freeze['selections']:
        version,fold=r['version'],r['fold'];ix=np.flatnonzero(np.isin(meta['patient'],list(roles(meta,fold)['test'])))
        p=cells[version,fold]
        frames.append(pd.DataFrame(dict(model=a.model,version=version,fold=fold,patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],probability=p,prediction=p>=r['threshold'])))
    frame=pd.concat(frames,ignore_index=True);patients=pd.DataFrame(EV.patient_rows(frame))
    assert len(patients)==160 and frame.groupby('version').size().eq(7635).all()
    for _,g in patients.groupby('version'):assert len(g)==g.patient.nunique()==80
    save(a.runtime/a.model/'PREDICTIONS_PRIVATE.pkl',frame);save(a.runtime/a.model/'PATIENT_METRICS_PRIVATE.pkl',patients)
    private=a.runtime/a.model/'private/results';private.mkdir(parents=True,exist_ok=True)
    frame.to_csv(private/'predictions.csv',index=False);patients.to_csv(private/'patient_metrics.csv',index=False)
    write(status,dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',model=a.model,freeze_sha256=fh,seal_sha256=sha(sp),patients_per_version=80,channels_per_version=7635,predictions_sha256=sha(a.runtime/a.model/'PREDICTIONS_PRIVATE.pkl'),patient_metrics_sha256=sha(a.runtime/a.model/'PATIENT_METRICS_PRIVATE.pkl'),scientific_status='EXPLORATORY_REPEATED_TEST'))
    aggregate_results(a.runtime,lock)
    print('RAW_BACKBONE_OUTER_COMPLETE',a.model,flush=True)


if __name__=='__main__':run(arguments())
