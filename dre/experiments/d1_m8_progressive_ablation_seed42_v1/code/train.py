"""Only A0/A1 scratch training; unchanged original D1 trainer and selections."""
import copy
import time
from support import *
NO_PR=load('progressive_no_pr',ROOT/'code/preprocessing.py')

def run(a):
    lock,c,core,fac,trainer,records,rec,banks,binding=setup(a);pub=a.runtime/'results'
    assert jread(pub/'FEATURE_PARITY.json')['status']=='PASS'
    feature=read(a.runtime/'FEATURES_PRIVATE.pkl');assert feature['binding']==binding
    assert np.array_equal(feature['DELTA14'],rec['R14'])
    def factory(seed=42):return CompactDelta(fac(seed).network,records)
    initialization=[]
    for fold in range(1,6):
        seed=42+1009*fold;states=[]
        for _ in range(3):core.seed_all(seed);states.append(factory(seed).state_dict())
        hashes=[core.state_hash(v) for v in states];assert len(set(hashes))==1
        initialization.append(dict(fold=fold,seed=seed,A0_sha256=hashes[0],A1_sha256=hashes[1],original_D1_constructor_sha256=hashes[2],exact=True,parameters=1537,trained_checkpoint_warmstart=False))
    write(pub/'INITIALIZATION_PARITY.json',dict(status='PASS',folds=initialization))
    refs=references(a,core,banks);selections=[];rows=[];histories=[];preaudit=[]
    oldfactory=trainer.SourceA0;trainer.SourceA0=factory
    try:
        for fold,d in enumerate(banks,1):
            seed=42+1009*fold;va=d['val']
            for arm,key in [('A0','ABS14'),('A1','DELTA14')]:
                x,pre=NO_PR.prepare(feature[key][d['archive_indices']],d['patient'],d['train'])
                assert set(pre['fit_patient_ids'])==d['roles']['fit']
                bp=a.runtime/f'{arm}/fold{fold}/BANK_PRIVATE.pt'
                if bp.exists():
                    b=torch.load(bp,map_location='cpu',weights_only=False);assert b['binding']==binding and np.array_equal(b['x'],x)
                else:core.torch_write(bp,dict(d,x=x,pre=pre,binding=binding))
                print('SCRATCH_CELL_START',arm,fold,flush=True);start=time.perf_counter()
                model,best,history=trainer.train(x,d['y'],d['patient'],d['g'],d['train'],va,bp.parent/'TRAIN',seed,c.digest(dict(binding=binding,bank=sha(bp),arm=arm)),device='cuda',penalty=0.)
                with torch.inference_mode():score=torch.sigmoid(model(torch.as_tensor(x[va],device='cuda'))).cpu().numpy()
                sel=core.select_threshold(d['y'][va],score,d['patient'][va]);assert sel['threshold']==best['threshold']
                cp=bp.parent/'TRAIN/BEST_PRIVATE.pt'; ck=torch.load(cp,map_location='cpu',weights_only=False)
                assert ck['binding']['initial_hash']==initialization[fold-1]['A0_sha256']
                row=dict(arm=arm,fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),bank_sha256=sha(bp),preprocessor_sha256=c.digest({k:pre[k].tolist() for k in ['imputer_statistics','mean','scale','var']}),epoch=best['epoch'],threshold=best['threshold'],epochs_run=len(history),initial_state_sha256=ck['binding']['initial_hash'],reused_frozen=False)
                selections.append(row);histories.extend(dict(arm=arm,fold=fold,**r) for r in history)
                rows.extend(EV.patient_rows(c,d,score,best['threshold'],arm,fold))
                save(a.runtime/f'development_private/{arm}_fold{fold}.pkl',dict(score=score,threshold=best['threshold'],binding=binding,checkpoint_sha256=sha(cp),indices=d['archive_indices'][va]))
                preaudit.append(dict(arm=arm,fold=fold,patient_z=False,fit_only=True,fit_patients=len(d['roles']['fit']),validation_patients=len(d['roles']['validation']),test_excluded=True,finite=True,wall_seconds=time.perf_counter()-start))
                write(pub/'RUN_STATUS.json',dict(status='TRAINING',training_cells_complete=len(selections),training_cells_total=10))
                del model;torch.cuda.empty_cache()
    finally:trainer.SourceA0=oldfactory
    # Frozen reference validation outputs: no model evaluation or refit.
    for r in refs:
        d=banks[r['fold']-1];old=r['old_arm'];fold=r['fold']
        if r['arm']=='A2':
            q=read(a.baseline/f'scores/D1_fold{fold}_PRIVATE.pkl');score=q['score']
            assert q['checkpoint_sha256']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        else:
            q=read(a.m8/f'predictions_private/M8_MACRO_AWARE_fold{fold}.pkl')
            assert q['binding']['checkpoint_sha256']==r['checkpoint_sha256'] and q['binding']['threshold']==r['threshold']
            score=torch.sigmoid(torch.from_numpy(q['modes']['full']['logit'])).numpy()
        assert core.select_threshold(d['y'][d['val']],score,d['patient'][d['val']])['threshold']==r['threshold']
        rows.extend(EV.patient_rows(c,d,score,r['threshold'],r['arm'],fold))
    df=pd.DataFrame(rows)
    for arm,g in df.groupby('arm'):assert (len(g),g.patient.nunique())==(65,47)
    for arm,want in [('A2',[.7259775079508951,.5036010898369186,.6445464619346358,.4435367362135798]),('A3',[.7566327539212014,.5422264205648775,.6667948222399955,.477249338229556])]:
        assert np.allclose(df[df.arm==arm][METRICS[:4]].mean(),want,rtol=0,atol=1e-12)
    save(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows)
    summary(df,pub,'DEVELOPMENT')
    pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False)
    pd.DataFrame([public_selection(r) for r in selections+refs]).to_csv(pub/'MODEL_SELECTION.csv',index=False)
    write(a.runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selections,references=refs,feature_sha256=sha(a.runtime/'FEATURES_PRIVATE.pkl')))
    write(pub/'PREPROCESSING_AUDIT.json',dict(status='PASS',cells=preaudit,PatientZ_A0_A1=False,PatientZ_A2_original=True))
    write(pub/'RUN_STATUS.json',dict(status='DEVELOPMENT_COMPLETE',new_training_cells=10,new_epochs=len(histories),frozen_references=10))
    print('ALL10_SCRATCH_CELLS_DEVELOPMENT_COMPLETE',flush=True)

if __name__=='__main__':run(args())
