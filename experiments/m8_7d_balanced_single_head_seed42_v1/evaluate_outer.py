"""Freeze every model/threshold, predict without full truth, then seal all cells."""
import gc
import hashlib
from common import *

FINAL=ROOT/'final_stage_v1';PUBLIC=FINAL/'public'
FINAL_FILES=['FINAL_PROTOCOL_LOCK.json','evaluate_outer.py','report_outer.py','verify_outer.py','run_outer.ps1']


def main():
    binding,core,fac,records,rec,banks,old=setup()
    terminal=jr(PUB/'INTERNAL_TERMINAL.json');assert terminal['status']=='COMPLETE_VERIFIED_BALANCED_7D_INTERNAL' and terminal['final_stage_warranted']
    assert jr(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json')['status']=='PASS'
    internal=jr(ROOT/'MODEL_FREEZE_PRIVATE.json');candidate=jr(ROOT/'CANDIDATE_PRIVATE.json')
    assert internal['input']==binding and candidate['selection']['selected']=='C2_INTERACTION' and candidate['selection']['new_qualified']
    assert candidate['model_freeze_sha256']==sha(ROOT/'MODEL_FREEZE_PRIVATE.json')
    FINAL.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True);pre=[];models=[]
    for fold in range(1,6):
        d=bank(fold,banks);ss=scale(fold)
        assert set(d['pre']['fit_patient_ids'])==d['roles']['fit'] and d['x'].shape[1]==14
        pre.append(dict(fold=fold,preprocessor={k:legacy.array_sha(np.asarray(v)) for k,v in d['pre'].items() if k!='fit_patient_ids'},scale=legacy.array_sha(ss.cpu().numpy())))
    for r in internal['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        q=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        models.append(dict(r,threshold=float(q['threshold']),epoch=int(q['epoch'])))
    oldseal=jr(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    frozen=dict(input=binding,internal_model_freeze=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),candidate=sha(ROOT/'CANDIDATE_PRIVATE.json'),
        internal_verification=sha(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json'),selected='C2_INTERACTION',models=models,preprocessing=pre,
        final_sources={n:sha(HERE/n) for n in FINAL_FILES},old_reference=sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),
        old_reference_files={f'S3_{f}':oldseal['files'][f'S3_{f}'] for f in range(1,6)},status='FROZEN_BEFORE_NEW_OUTER_EXPLORATORY')
    freeze(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json',frozen);fh=sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');inputs=store();files={};census=[]
    for r in models:
        arm,fold=r['arm'],r['fold'];d=bank(fold,banks);meta=rec['meta'];roles=d['roles'];ss=scale(fold)
        ix=np.flatnonzero(np.isin(meta['patient'],list(roles['test'])));assert len(set(meta['patient'][ix]))==[16,16,17,15,16][fold-1]
        key=f'{arm}_{fold}';path=FINAL/'scores_private'/(key+'.pkl')
        if arm=='C0_CONCAT':
            source=OLDROOT/'outer_scores_private'/f'S3_{fold}.pkl';assert sha(source)==frozen['old_reference_files'][f'S3_{fold}']
            q=legacy.read(source);assert np.array_equal(q['indices'],ix) and q['checkpoint_sha256']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
            expected_score=q['score_nez'].copy()
            save(path,dict(freeze=fh,indices=ix,score_nez=q['score_nez'],threshold=r['threshold'],checkpoint=r['checkpoint_sha256'],reused_source_sha256=sha(source))) if not path.exists() else None
        else:
            z=core.patient_z(rec['R14'][ix],meta['patient'][ix]);pp=d['pre']
            x=((np.where(np.isnan(z),pp['imputer_statistics'],z)-pp['mean'])/pp['scale']).astype(np.float32)
            assert x.shape==(len(ix),14) and np.isfinite(x).all()
            m=make(core,fac,records,fold,arm);m.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model']);m.eval()
            before=state_hash(m.state_dict());out=np.empty(len(ix),np.float32);windows=0;reused=0
            for pid in sorted(roles['test']):
                jj=np.flatnonzero(meta['patient'][ix]==pid);ii=ix[jj]
                cb=dict(freeze=fh,checkpoint=r['checkpoint_sha256'],indices=legacy.array_sha(ii),input=legacy.array_sha(x[jj]),scale=legacy.array_sha(ss.cpu().numpy()))
                token=hashlib.sha256(str(pid).encode()).hexdigest();cache=FINAL/'patient_scores_private'/arm/f'fold{fold}'/(token+'.pkl')
                if cache.exists():q=read(cache);assert q['binding']==cb;reused+=1
                else:
                    raw,mask,times,delta,_=inputs.get(str(pid),meta['channel'][ii].astype(str).tolist(),'cuda')
                    with torch.inference_mode():score=probability(m(torch.as_tensor(x[jj],device='cuda'),raw,mask,times,delta,ss)['logit'].cpu().numpy())
                    q=dict(binding=cb,score_nez=score,valid_windows=int(mask.sum()));save(cache,q)
                    del raw,mask,times,delta
                assert len(q['score_nez'])==len(jj) and np.isfinite(q['score_nez']).all();out[jj]=q['score_nez'];windows+=q['valid_windows']
            assert state_hash(m.state_dict())==before
            result=dict(freeze=fh,indices=ix,score_nez=out,threshold=r['threshold'],checkpoint=r['checkpoint_sha256'])
            if path.exists():q=read(path);assert q['freeze']==fh and np.array_equal(q['score_nez'],out)
            else:save(path,result)
            census.append(dict(arm=arm,fold=fold,patients=len(roles['test']),channels=len(ix),valid_windows=windows,reused_patient_caches=reused))
            del m;gc.collect();torch.cuda.empty_cache()
        q=read(path);assert q['freeze']==fh and np.array_equal(q['indices'],ix) and q['threshold']==r['threshold'] and q['checkpoint']==r['checkpoint_sha256']
        if arm=='C0_CONCAT':assert np.array_equal(q['score_nez'],expected_score)
        files[key]=sha(path);print('BALANCED_SINGLE_OUTER_CELL_SEALED',arm,fold,flush=True)
    assert len(files)==15
    for arm in NEW:
        rows=[q for q in census if q['arm']==arm];assert sum(q['patients'] for q in rows)==80 and sum(q['channels'] for q in rows)==7635 and sum(q['valid_windows'] for q in rows)==1471965
    save(FINAL/'METADATA_PRIVATE.pkl',dict(meta=rec['meta'],archive_sha256=rec['binding']['archive']))
    freeze(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,files=files,all15_cells_sealed_before_full_truth=True))
    pd.DataFrame(census).to_csv(PUBLIC/'RAW_COVERAGE.csv',index=False)
    write(PUBLIC/'PREDICTION_STATUS.json',dict(status='ALL10_NEW_PLUS5_REUSED_OUTER_SCORES_SEALED',freeze_sha256=fh,score_seal_sha256=sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),
        selected='C2_INTERACTION',models=3,patients_each=80,channels_each=7635,new_inference_cells=10,reused_reference_cells=5,all7_features_unchanged=True,full_truth_used=False))
    print('BALANCED_SINGLE_OUTER_SCORES_COMPLETE_NO_FULL_TRUTH',flush=True)


if __name__=='__main__':main()
