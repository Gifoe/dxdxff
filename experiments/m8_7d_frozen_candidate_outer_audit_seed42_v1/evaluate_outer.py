"""One frozen full-cohort inference, private exact patient-level resume."""
from common import *

def main():
    binding,core,fac,records,rec,banks,old=setup();fr=read_frozen()
    assert fr['input']==binding and fr['preprocessing']==preprocessing(banks)
    fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');inputs=store();files={};census=[]
    for r in fr['models']:
        arm,fold=r['arm'],r['fold'];d=bank(fold,banks);ss=scale(fold);meta=rec['meta'];roles=d['roles']
        ix=np.flatnonzero(np.isin(meta['patient'],list(roles['test'])))
        assert len(set(meta['patient'][ix]))==[16,16,17,15,16][fold-1]
        key=f'{arm}_{fold}';path=ROOT/'scores_private'/(key+'.pkl')
        if arm=='C2_REFERENCE':
            src=BASE_ROOT/'final_stage_v1/scores_private'/f'C2_INTERACTION_{fold}.pkl'
            assert sha(src)==fr['old_reference_files'][f'C2_INTERACTION_{fold}']
            q=read(src)
            assert np.array_equal(q['indices'],ix) and q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
            result=dict(freeze=fh,indices=ix,score_nez=q['score_nez'],threshold=r['threshold'],checkpoint=r['checkpoint_sha256'],reused_source_sha256=sha(src))
        else:
            z=core.patient_z(rec['R14'][ix],meta['patient'][ix]);pp=d['pre']
            x=((np.where(np.isnan(z),pp['imputer_statistics'],z)-pp['mean'])/pp['scale']).astype(np.float32)
            assert x.shape==(len(ix),14) and np.isfinite(x).all()
            m,_=make(r,core,fac,records,old);before=state_hash(m.state_dict())
            out=np.empty(len(ix),np.float32);windows=0;reused=0
            for n,pid in enumerate(sorted(roles['test']),1):
                jj=np.flatnonzero(meta['patient'][ix]==pid);ii=ix[jj]
                cb=cache_binding(fh,r,ii,x[jj],ss)
                token=hashlib.sha256(str(pid).encode()).hexdigest();cache=ROOT/'patient_scores_private'/arm/f'fold{fold}'/(token+'.pkl')
                if cache.exists():q=read(cache);assert q['binding']==cb;reused+=1
                else:
                    raw,mask,times,delta,_=inputs.get(str(pid),meta['channel'][ii].astype(str).tolist(),'cuda')
                    with torch.inference_mode():score=probability(m(torch.as_tensor(x[jj],device='cuda'),raw,mask,times,delta,ss)['logit'].cpu().numpy())
                    q=dict(binding=cb,score_nez=score,valid_windows=int(mask.sum()));save(cache,q)
                    del raw,mask,times,delta
                assert q['score_nez'].shape==(len(jj),) and np.isfinite(q['score_nez']).all() and np.all((q['score_nez']>=0)&(q['score_nez']<=1))
                out[jj]=q['score_nez'];windows+=q['valid_windows']
                write(PUBLIC/'PROGRESS.json',dict(status='FROZEN_INFERENCE_RUNNING',arm=arm,fold=fold,patient_number_within_fold=n,
                    patients_within_fold=len(roles['test']),sealed_fold_cells=len(files),total_logical_fold_cells=55,no_new_training=True))
                print('OUTER_PATIENT_ATOMIC',arm,fold,n,len(roles['test']),flush=True)
            assert state_hash(m.state_dict())==before
            result=dict(freeze=fh,indices=ix,score_nez=out,threshold=r['threshold'],checkpoint=r['checkpoint_sha256'])
            census.append(dict(arm=arm,fold=fold,patients=len(roles['test']),channels=len(ix),valid_windows=windows,reused_patient_caches=reused))
            del m;gc.collect();torch.cuda.empty_cache()
        if path.exists():
            q=read(path)
            for k in result:
                assert np.array_equal(q[k],result[k]) if isinstance(result[k],np.ndarray) else q[k]==result[k]
        else:save(path,result)
        files[key]=sha(path);print('FROZEN_OUTER_CELL_SEALED',arm,fold,flush=True)
    assert len(files)==55
    for arm in NEW:
        rows=[q for q in census if q['arm']==arm]
        assert len(rows)==5 and sum(q['patients'] for q in rows)==80 and sum(q['channels'] for q in rows)==7635 and sum(q['valid_windows'] for q in rows)==1471965
    metadata=dict(meta=rec['meta'],archive_sha256=rec['binding']['archive'])
    if (ROOT/'METADATA_PRIVATE.pkl').exists():assert read(ROOT/'METADATA_PRIVATE.pkl')['archive_sha256']==metadata['archive_sha256']
    else:save(ROOT/'METADATA_PRIVATE.pkl',metadata)
    freeze(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,files=files,all55_cells_sealed_before_full_truth=True))
    csv_out('RAW_COVERAGE',census)
    write(PUBLIC/'PREDICTION_STATUS.json',dict(status='ALL50_NEW_PLUS5_REUSED_OUTER_SCORES_SEALED',freeze_sha256=fh,
        score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json'),arms=11,patients_each=80,channels_each=7635,new_inference_cells=50,
        reused_reference_cells=5,full_truth_used=False,original7_features_unchanged=True))
    print('FROZEN_CANDIDATE_OUTER_SCORES_COMPLETE_BEFORE_FULL_TRUTH',flush=True)

if __name__=='__main__':main()
