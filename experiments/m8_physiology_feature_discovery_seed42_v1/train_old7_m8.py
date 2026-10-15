"""80matchedold7D1+M8fits;development-only,neverloadOuterpredictions/labels."""
import time
from audit_old7 import *
from compact_delta_classifier import CompactDelta

P3CODE=Path(r'D:\d1_small_module_exploration_seed42_v1\phase3\code')
sys.path.insert(0,str(P3CODE))
P3=load('discovery_original_M8_trainer',P3CODE/'run.py')
MaskedM8=load('discovery_masked_M8',TASK/'old7_m8_model.py').MaskedM8
RUN=OUT/'old7_matched_M8';(RUN/'public').mkdir(parents=True,exist_ok=True)


def normalize_rng(q):
    if torch.is_tensor(q):return q.detach().cpu() if q.dtype==torch.uint8 else q
    if isinstance(q,dict):return {k:normalize_rng(v) for k,v in q.items()}
    if isinstance(q,list):return [normalize_rng(v) for v in q]
    if isinstance(q,tuple):return tuple(normalize_rng(v) for v in q)
    return q


def setup_new():
    lock=jr(TASK/'OLD7_M8_ABLATION_LOCK.json')
    c,core,fac,trainer,records,rec,banks,hist,deps=context(SOURCE,BASELINE)
    selection=OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json'
    assert selection.exists() and jr(selection)['selected_only_inside_each_original_FIT']
    # No old Outer freeze/seal/prediction/metric file is read here.
    root=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime')
    pin=jr(root/'INPUT_FREEZE_PRIVATE.json')
    assert pin['code']=={p.name:sha(p) for p in P3CODE.glob('*.py')}
    assert pin['prior_code']=={p.name:sha(p) for p in (P3CODE.parent.parent/'phase2/code').glob('*.py')}
    protected={str(p):sha(p) for p in [root/'INPUT_FREEZE_PRIVATE.json',root/'MODEL_FREEZE_PRIVATE.json']}
    for f in range(1,6):
        cp=root/f'M8_MACRO_AWARE/fold{f}/TRAIN/BEST_PRIVATE.pt';protected[str(cp)]=sha(cp)
        for p in [BASELINE/f'D1/fold{f}/TRAIN/BEST_PRIVATE.pt',BASELINE/f'D1/fold{f}/BANK_PRIVATE.pt',
            Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{f}_PRIVATE.pkl']:
            protected[str(p)]=sha(p)
    bind=dict(source={n:sha(TASK/n) for n in ['OLD7_M8_ABLATION_LOCK.json','train_old7_m8.py','old7_m8_model.py','audit_old7.py','AUDIT_PROTOCOL.json']},
        dependencies=deps,original_M8_code=pin['code'],original_phase2_code=pin['prior_code'],
        trainer_sha256=sha(trainer.__file__),FIT_selection_freeze_sha256=sha(selection),protected=protected,torch=torch.__version__,numpy=np.__version__)
    fp=RUN/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert jr(fp)==bind
    else:write(fp,bind)
    original_restore=core.restore_rng
    def restore(state):return original_restore(normalize_rng(state))
    core.restore_rng=restore;trainer.restore_rng=restore
    core.seed_all(1051);state=core.rng_state();expected=torch.rand(4,device='cuda')
    fake={k:(v.cuda() if torch.is_tensor(v) and v.dtype==torch.uint8 else v) for k,v in state.items()}
    restore(fake);assert torch.equal(expected,torch.rand(4,device='cuda'))
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    return lock,c,core,fac,trainer,records,rec,banks,bind


def build(core,fac,records,cp,fold,keep):
    base=P3.make_base(core,fac,records,cp,42+1009*fold)
    model=MaskedM8(base,keep).cuda()
    original=P3.make(core,fac,records,BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold,'M8_MACRO_AWARE')
    select=lambda obj:{k:v for k,v in obj.state_dict().items() if k.startswith(('cnn.','head.','gate.'))}
    assert core.state_hash(select(model))==core.state_hash(select(original))
    return model,original


def report(c,core,banks,selections,binding):
    selected=OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json'
    if not selected.exists() or not (TASK/'M8_INTEGRATION_LOCK.json').exists():
        write(RUN/'public/STATUS.json',dict(status='TRAINING_COMPLETE_DEVELOPMENT_AGGREGATES_WITHHELD_PENDING_INTEGRATION_LOCK',new_base_cells=80,new_M8_cells=80,
            development_aggregates_not_exposed=True,no_Outer=True,full_goal_complete=False));return
    assert jr(selected)['selected_only_inside_each_original_FIT']
    frames=[]
    for s in selections:
        q=read(Path(s['validation_path']));assert q['D1_checkpoint_sha256']==s['D1_checkpoint_sha256'] and q['M8_checkpoint_sha256']==s['M8_checkpoint_sha256']
        d=banks[s['fold']-1];va=d['val']
        for head,logit,threshold in [('D1',q['D1_logit'],s['D1_threshold']),('M8',q['M8_logit'],s['M8_threshold'])]:
            pn=torch.sigmoid(torch.from_numpy(logit)).numpy()
            frames.append(pd.DataFrame(dict(model=s['variant'],version=head,fold=s['fold'],patient=d['patient'][va],channel=d['channel'][va],center=d['center'][va],
                y=(d['y'][va]==0).astype(int),probability=1-pn.astype(float),prediction=pn<threshold)))
    frame=pd.concat(frames,ignore_index=True);patients=pd.DataFrame(EV.patient_rows(frame));save(RUN/'DEVELOPMENT_PATIENTS_PRIVATE.pkl',patients);save(RUN/'DEVELOPMENT_PREDICTIONS_PRIVATE.pkl',frame)
    metrics=EV.CLASSIFICATION+EV.RANKING;summary=[];folds=[];centers=[];intervals=[];coverage=[]
    for (name,head),g in patients.groupby(['model','version']):
        assert (len(g),g.patient.nunique())==(65,47)
        summary.append(dict(variant=name,head=head,patient_appearances=65,unique_patients=47,**{m:EV.nanmean(g[m]) for m in metrics}))
        for group,target in [('fold',folds),('center',centers)]:
            for value,h in g.groupby(group):target.append(dict(variant=name,head=head,**{group:value},patient_appearances=len(h),unique_patients=h.patient.nunique(),**{m:EV.nanmean(h[m]) for m in metrics}))
        for m in metrics:coverage.append(dict(variant=name,head=head,metric=m,valid_appearances=int(np.isfinite(g[m]).sum()),total_appearances=len(g)))
        if name=='FULL_RETRAIN':continue
        a=g.set_index(['fold','patient']).sort_index();b=patients[(patients.model=='FULL_RETRAIN')&(patients.version==head)].set_index(['fold','patient']).loc[a.index]
        assert a.index.equals(b.index);ids=sorted(g.patient.unique());draw=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)))
        counts=np.stack([(draw==k).sum(1) for k in range(len(ids))],1)
        delta=a[metrics].to_numpy()-b[metrics].to_numpy();group_patient=np.asarray(a.index.get_level_values('patient'))
        for j,m in enumerate(metrics):
            num=np.asarray([np.nansum(delta[group_patient==p,j]) for p in ids]);den=np.asarray([np.isfinite(delta[group_patient==p,j]).sum() for p in ids])
            dist=np.divide(counts@num,counts@den,out=np.full(10000,np.nan),where=(counts@den)>0)
            intervals.append(dict(variant=name,head=head,metric=m,delta=EV.nanmean(delta[:,j]),ci_low=float(np.nanpercentile(dist,2.5)),
                ci_high=float(np.nanpercentile(dist,97.5)),cluster_patients=len(ids),draws=10000,seed=42,selection_conditioned=True))
    for filename,data in [('DEVELOPMENT_SUMMARY',summary),('DEVELOPMENT_BY_FOLD',folds),('DEVELOPMENT_BY_CENTER',centers),('METRIC_COVERAGE',coverage),('PAIRED_CLUSTER_INTERVALS',intervals)]:
        pd.DataFrame(data).to_csv(RUN/'public'/(filename+'.csv'),index=False)
    for p,h in binding['protected'].items():assert sha(p)==h
    write(RUN/'public/STATUS.json',dict(status='ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE',base_cells=80,M8_cells=80,
        variants=16,folds=5,appearances_per_variant=65,unique_validation_patients=47,
        FIT_selection_freeze_before_aggregate_read_sha256=sha(selected),no_Outer=True,full_goal_complete=False))


def main():
    lock,c,core,fac,trainer,records,rec,banks,binding=setup_new()
    if (RUN/'public/STATUS.json').exists() and jr(RUN/'public/STATUS.json')['status']=='ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE':
        print('OLD7_M8_DEVELOPMENT_ALREADY_COMPLETE_NO_REFIT',flush=True);return
    masks=[('FULL_RETRAIN',[1]*7)]+[(f'DROP_{k}',[int(j!=k) for j in range(7)]) for k in range(7)]+[(f'ONLY_{k}',[int(j==k) for j in range(7)]) for k in range(7)]+[('NONE',[0]*7)]
    store=P3.Inputs(BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),RUN);store.arm='M1_SCALED_DGM'
    basefactory=trainer.SourceA0;selections=[];histories=[]
    for variant,keep in masks:
        for fold in range(1,6):
            original=banks[fold-1];bp=BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt';d=torch.load(bp,map_location='cpu',weights_only=False)
            for k in ['patient','channel','y','train','val']:assert np.array_equal(d[k],original[k])
            masked=d['x']*np.repeat(np.asarray(keep,np.float32),2)[None];d=dict(d,x=masked,archive_indices=original['archive_indices'],roles=original['roles'])
            assert set(d['patient']).isdisjoint(d['roles']['test'])
            cell=RUN/variant/f'fold{fold}';bankpath=cell/'BANK_PRIVATE.pt'
            if bankpath.exists():q=torch.load(bankpath,map_location='cpu',weights_only=False);assert q['binding']==binding and np.array_equal(q['x'],masked)
            else:core.torch_write(bankpath,dict(d,binding=binding))
            try:
                trainer.SourceA0=lambda seed=42:CompactDelta(fac(seed).network,records)
                t=time.perf_counter();base,best,h=trainer.train(masked,d['y'],d['patient'],d['g'],d['train'],d['val'],cell/'D1_TRAIN',42+1009*fold,
                    c.digest(dict(binding=binding,variant=variant,bank_sha256=sha(bankpath))),device='cuda',penalty=0.)
                base_seconds=time.perf_counter()-t
            finally:trainer.SourceA0=basefactory
            with torch.inference_mode():base_logit=base(torch.as_tensor(masked[d['val']],device='cuda')).cpu().numpy()
            assert core.select_threshold(d['y'][d['val']],torch.sigmoid(torch.from_numpy(base_logit)).numpy(),d['patient'][d['val']])['threshold']==best['threshold']
            cp=cell/'D1_TRAIN/BEST_PRIVATE.pt';scale=torch.tensor(read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
            if variant=='FULL_RETRAIN' and fold==1 and not (RUN/'public/ADMISSION.json').exists():
                model,original_model=build(core,fac,records,cp,fold,keep)
                audits=P3.admission(model,d,store,scale,core,'M8_MACRO_AWARE')
                tr=d['train'];ii=tr[d['patient'][tr]==sorted(set(d['patient'][tr]))[0]]
                raw,mask,times,delta,perms=store.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
                with torch.inference_mode():
                    # Admission model changed in its discarded smoke,so compare features on a fresh pair.
                    fresh,origin=build(core,fac,records,cp,fold,keep)
                    error=float((fresh.features(raw,mask,times,delta,scale)[0]-origin.features(raw,mask,times,delta,scale)[0]).abs().max());assert error==0
                    dropped,unused=build(core,fac,records,cp,fold,[0,1,1,1,1,1,1])
                    one=dropped.features(raw,mask,times,delta,scale)[0];changed=delta.clone();changed[...,0]=123
                    assert torch.equal(one,dropped.features(raw,mask,times,changed,scale)[0])
                write(RUN/'public/ADMISSION.json',dict(status='PASS',all_ones_feature_parity_max_error=error,dropped_gate_input_independent=True,
                    original_correction_initialization=True,RNG_CPU_byte_resume_checked=True,smoke=audits))
                del model,original_model,fresh,origin,dropped,unused,raw,mask,times,delta,perms
            model,original_model=build(core,fac,records,cp,fold,keep);del original_model
            runtime=RUN/variant;(runtime/'results').mkdir(parents=True,exist_ok=True)
            mb,mh=P3.fit(model,d,store,scale,cell/'M8_TRAIN',42+1009*fold,binding,core,runtime,'M8_MACRO_AWARE',fold)
            mcp=cell/'M8_TRAIN/BEST_PRIVATE.pt';pred=P3.EV.predict(model,d,d['val'],store,scale)
            assert np.max(np.abs(pred['base_logit']-base_logit))<=1e-6
            pp=cell/'VALIDATION_PRIVATE.pkl';q=dict(D1_logit=base_logit,M8_logit=pred['logit'],D1_checkpoint_sha256=sha(cp),M8_checkpoint_sha256=sha(mcp),
                validation_archive_indices=d['archive_indices'][d['val']])
            if pp.exists():old=read(pp);assert old['D1_checkpoint_sha256']==q['D1_checkpoint_sha256'] and old['M8_checkpoint_sha256']==q['M8_checkpoint_sha256']
            else:save(pp,q)
            selections.append(dict(variant=variant,fold=fold,keep=keep,D1_checkpoint=str(cp),D1_checkpoint_sha256=sha(cp),
                M8_checkpoint=str(mcp),M8_checkpoint_sha256=sha(mcp),bank=str(bankpath),bank_sha256=sha(bankpath),validation_path=str(pp),
                D1_threshold=best['threshold'],D1_epoch=best['epoch'],D1_epochs_run=len(h),M8_threshold=mb['threshold'],M8_epoch=mb['epoch'],M8_epochs_run=len(mh),D1_wall_seconds=base_seconds))
            histories += [dict(variant=variant,fold=fold,head='D1',**r) for r in h]
            histories += [dict(variant=variant,head='M8',**{k:v for k,v in r.items() if k!='arm'}) for r in mh]
            save(RUN/'MODEL_SELECTION_PRIVATE.pkl',dict(binding=binding,selections=selections))
            pd.DataFrame([{k:v for k,v in r.items() if k not in ['D1_checkpoint','M8_checkpoint','bank','validation_path']} for r in selections]).to_csv(RUN/'public/MODEL_SELECTION.csv',index=False)
            pd.DataFrame(histories).to_csv(RUN/'public/TRAINING_HISTORY.csv',index=False)
            write(RUN/'public/STATUS.json',dict(status='TRAINING',completed_model_pairs=len(selections),total_model_pairs=80,variant=variant,fold=fold,no_Outer=True,full_goal_complete=False))
            print('OLD7_MATCHED_M8_PAIR_COMPLETE',variant,fold,len(selections),'of80',flush=True)
            del model,base,d;gc.collect();torch.cuda.empty_cache()
    assert len(selections)==80
    report(c,core,banks,selections,binding)
    print('OLD7_MATCHED_M8_TRAINING_MATRIX_COMPLETE',flush=True)


if __name__=='__main__':main()
