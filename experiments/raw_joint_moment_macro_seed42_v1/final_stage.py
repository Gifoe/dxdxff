"""Guarded final15 fresh fits/30 score cells; primary remains legal ownFIT policy."""
import argparse
import gc
import time
import common as c
import train_internal as trainer
from model import Model,objective

FINAL=c.ROOT/'final_stage_v1';PUBLIC=FINAL/'public';OLD=c.previous.WAVE/'final_stage_v1'
PROCEDURES=c.ARMS+['SELECTED']


def setup():
    initial=c.setup();terminal=c.jr(c.PUB/'INTERNAL_TERMINAL.json');verified=c.jr(c.PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json')
    assert terminal['status']=='COMPLETE_VERIFIED_JOINT_RAW_INTERNAL' and terminal['final_stage_warranted'] and verified['status']=='PASS'
    assert verified['all9_patient_metrics_independent'] and verified['independent_CI_metric_checks']==126 and verified['maximum_bootstrap_difference']<1e-12
    sp=c.ROOT/'OWN_FIT_SELECTION_PRIVATE.json';assert c.sha(sp)==verified['ownFIT_selection_sha256']
    selected=c.jr(sp);assert selected['input']==initial and selected['internal_seal']==c.sha(c.ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    choices={r['fold']:r['selected_arm'] for r in selected['choices']};assert set(choices)==set(range(1,6)) and set(choices.values())<=set(c.ARMS)
    assert c.sha(OLD/'MODEL_FREEZE_PRIVATE.pkl')=='4f936da78565bf823f0905a8e7fb6ba6da09f13c8b873c4cff4719e5c6319a9c'
    assert c.sha(OLD/'OUTER_SCORE_SEAL_PRIVATE.json')=='2efe5fe40dbe327c8bfcec1869c214b2c41e2294133288a9f6bed5b46b46298f'
    previous=c.read(OLD/'MODEL_FREEZE_PRIVATE.pkl');refs=[]
    for row in previous['rows']:
        if row['procedure'] not in ['NATIVE','PRECOMPAND_RESIDUAL']:continue
        assert c.sha(row['body'])==row['body_sha256']
        if row['matrix']:assert c.sha(row['matrix'])==row['matrix_sha256']
        refs.append(dict(fold=row['fold'],procedure='NATIVE' if row['procedure']=='NATIVE' else 'FROZEN_PRE',old_procedure=row['procedure'],
            threshold=row['threshold'],source_body_sha256=row['body_sha256'],source_matrix_sha256=row['matrix_sha256']))
    assert len(refs)==10
    binding=dict(internal_input=c.digest(initial),ownFIT_selection_sha256=c.sha(sp),internal_score_seal=c.sha(c.ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
        old_reference_model_freeze=c.sha(OLD/'MODEL_FREEZE_PRIVATE.pkl'),old_reference_score_seal=c.sha(OLD/'OUTER_SCORE_SEAL_PRIVATE.json'),
        reference_rows=refs,raw_registry=c.io.EXPECTED_REG,reconstruction=c.io.EXPECTED_REC,
        source={n:c.sha(c.HERE/n) for n in ['FINAL_PROTOCOL_LOCK.json','final_stage.py','verify_final.py','train_internal.py','model.py','common.py']},
        role='FIT_GRADIENTS_DEVELOPMENT_SELECTION_NO_OUTER',primary='SELECTED')
    FINAL.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True);c.freeze(FINAL/'INPUT_FREEZE_PRIVATE.json',binding)
    trainer.ROOT=FINAL;trainer.PUB=PUBLIC
    def scoped_write(path,value):
        if c.Path(path).name=='RUN_STATUS.json':
            value=dict(value);value.pop('original_Development_Outer_used',None)
            value.update(status='FINAL_JOINT_RAW_FIT_DEVELOPMENT_SELECTION',original_Development_used=True,ownOuter_used=False)
        c.write(path,value)
    trainer.write=scoped_write
    return binding,choices,refs


def admit():
    binding,_,_=setup();_,banks=c.io.identity_and_banks(labels=True);store=c.RawStore();largest=None;unique=set()
    for fold,d in enumerate(banks,1):
        assert set(d['patient'][d['train']])==d['roles']['fit']
        assert set(d['patient'][d['train']]).isdisjoint(d['roles']['validation']|d['roles']['test'])
        for pid in sorted(set(d['patient'][d['train']])):
            if str(pid) in unique:continue
            ii=d['train'][d['patient'][d['train']]==pid];raw,mask,times=store.get(pid,d['channel'][ii],device='cpu');unique.add(str(pid))
            n=int(mask.sum())
            if largest is None or n>largest['windows']:largest=dict(fold=fold,ii=ii,pid=pid,windows=n)
    assert len(unique)==80
    d=banks[largest['fold']-1];ii=largest['ii'];tr=d['train'];raw,mask,times=store.get(largest['pid'],d['channel'][ii])
    assert int(mask.sum())==largest['windows'];y=c.torch.tensor(d['y'][ii],device='cuda',dtype=c.torch.float32)
    pos=c.torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda');records=[]
    for arm in c.TRAINED:
        model=Model(arm,largest['fold']).cuda();before=c.state_hash(model.state_dict());c.torch.cuda.reset_peak_memory_stats()
        c.torch.cuda.synchronize();tick=time.perf_counter();out=model(raw,mask,times);loss=objective(out,y,pos);loss.backward();c.torch.cuda.synchronize()
        assert c.torch.isfinite(loss) and all(p.grad is not None and c.torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.cnn[0].weight.grad.abs().sum()>0 and model.head[2].weight.grad.abs().sum()>0 and c.state_hash(model.state_dict())==before
        records.append(dict(arm=arm,parameters=14577,channels=len(ii),valid_windows=largest['windows'],largest_of_unique_FIT_patients=len(unique),
            seconds=time.perf_counter()-tick,peak_allocated_bytes=c.torch.cuda.max_memory_allocated(),all_gradients_finite=True,optimizer_updates=0))
        del model,out,loss;gc.collect();c.torch.cuda.empty_cache()
    c.write(PUBLIC/'FINAL_ADMISSION.json',dict(status='PASS',input_binding_sha256=c.digest(binding),largest_complete_FIT_patient_backward=records,
        admission_loss_uses_only_ownFIT_labels=True,optimizer_updates=0))
    print('JOINT_RAW_FINAL_COMPLETE_FIT_PATIENT_ADMITTED',flush=True)


def fit():
    binding,choices,refs=setup();gate=c.jr(PUBLIC/'FINAL_ADMISSION.json');assert gate['status']=='PASS' and gate['input_binding_sha256']==c.digest(binding)
    _,banks=c.io.identity_and_banks(labels=True);rows=[];public=[];history=[]
    for fold,d in enumerate(banks,1):
        assert set(d['patient'][d['train']])==d['roles']['fit'] and set(d['patient'][d['val']])==d['roles']['validation']
        ctx=dict(fold=fold,screen='final',data=d,roles=dict(train=d['train'],choice=c.np.array([],int),cal=d['val'],held=d['val']),
            representation_sha256=c.digest(dict(fold=fold,archive_indices=d['archive_indices'].tolist(),scope='ORIGINAL_FIT_AND_DEVELOPMENT_ONLY',raw_registry=c.io.EXPECTED_REG)))
        byarm={}
        for ref in [r for r in refs if r['fold']==fold]:
            row=dict(ref,arm=ref['procedure'],kind='FIXED_REFERENCE');rows.append(row);byarm[row['procedure']]=row
        store=c.RawStore()
        for arm in c.TRAINED:
            summary,_,hh=trainer.fit(ctx,arm,binding,store);bp=FINAL/'models_private'/f'fold{fold}_screenfinal'/arm/'BEST_PRIVATE.pt'
            best=c.torch.load(bp,map_location='cpu',weights_only=False)
            row=dict(fold=fold,procedure=arm,arm=arm,kind='FRESH_TRAINED',checkpoint=str(bp),checkpoint_sha256=c.sha(bp),
                initial_state_sha256=best['binding']['initial_state_sha256'],threshold=summary['threshold'],epoch=summary['selected_epoch'])
            rows.append(row);byarm[arm]=row
            public.append(dict(fold=fold,arm=arm,epochs=summary['epochs'],selected_epoch=summary['selected_epoch'],threshold_NEZ=summary['threshold'],
                parameters=14577,random_start=True,initial_state_sha256=row['initial_state_sha256']))
            history.extend([dict(fold=fold,arm=arm,**r) for r in hh])
            c.pd.DataFrame(public).to_csv(PUBLIC/'FINAL_TRAINING_SELECTION.csv',index=False);c.pd.DataFrame(history).to_csv(PUBLIC/'FINAL_TRAIN_HISTORY.csv',index=False)
        assert len({byarm[a]['initial_state_sha256'] for a in c.TRAINED})==1
        rows.append(dict(byarm[choices[fold]],procedure='SELECTED',selected_arm=choices[fold]))
    assert len(rows)==30;c.save(FINAL/'MODEL_FREEZE_PRIVATE.pkl',dict(input=binding,rows=rows,primary='SELECTED'))
    c.write(PUBLIC/'TRAINING_STATUS.json',dict(status='ALL30_JOINT_RAW_FINAL_MODELS_FROZEN',actual_fresh_fits=15,logical_cells=30,
        model_freeze_sha256=c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl'),new_Outer_passes=0,all_five_architecture_choices_before_Development=True))
    print('JOINT_RAW_ALL30_FINAL_MODELS_FROZEN',flush=True)


def test():
    binding,choices,_=setup();status=c.jr(PUBLIC/'TRAINING_STATUS.json');assert status['status']=='ALL30_JOINT_RAW_FINAL_MODELS_FROZEN'
    assert status['model_freeze_sha256']==c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl');fr=c.read(FINAL/'MODEL_FREEZE_PRIVATE.pkl');assert fr['input']==binding
    for row in fr['rows']:
        if row['kind']=='FRESH_TRAINED':assert c.sha(row['checkpoint'])==row['checkpoint_sha256']
    c.freeze(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json',dict(input=binding,model_freeze=c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl'),primary='SELECTED',procedures=PROCEDURES))
    fh=c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');oldseal=c.jr(OLD/'OUTER_SCORE_SEAL_PRIVATE.json')
    meta,banks=c.io.identity_and_banks(labels=False);store=c.RawStore();seals={};coverage=[]
    for fold,d in enumerate(banks,1):
        ix=c.np.flatnonzero(c.np.isin(meta['patient'],sorted(d['roles']['test'])));foldrows=[r for r in fr['rows'] if r['fold']==fold]
        assert set(meta['patient'][ix]).isdisjoint(d['roles']['fit']|d['roles']['validation']);score={};models={}
        for row in foldrows:
            if row['procedure']=='SELECTED':continue
            if row['kind']=='FIXED_REFERENCE':
                key=f"{row['old_procedure']}_{fold}";path=OLD/'outer_scores_private'/(key+'.pkl');assert c.sha(path)==oldseal['cells'][key]
                q=c.read(path);assert c.np.array_equal(q['indices'],ix) and q['threshold']==row['threshold']
                assert q['body_sha256']==row['source_body_sha256'] and q['matrix_sha256']==row['source_matrix_sha256'];score[row['procedure']]=q['score_nez'].copy()
            else:
                best=c.torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
                assert best['binding']['input']==binding and best['binding']['arm']==row['arm'] and best['selection']['threshold']==row['threshold']
                model=Model(row['arm'],fold).cuda().eval();model.load_state_dict(best['model']);models[row['arm']]=model;score[row['arm']]=c.np.empty(len(ix),c.np.float32)
        windows=0
        for pid in sorted(set(meta['patient'][ix])):
            loc=c.np.flatnonzero(meta['patient'][ix]==pid);ai=ix[loc];path=FINAL/'outer_patient_scores_private'/f'fold{fold}'/(c.io.private_key(pid)+'.pkl')
            pb=dict(freeze=fh,fold=fold,archive_indices=ai.tolist(),waveform_sha256=store.plans[str(pid)]['sha256'])
            if path.exists():q=c.read(path);assert q['binding']==pb
            else:
                raw,mask,times=store.get(pid,meta['channel'][ai])
                with c.torch.inference_mode():pn={arm:c.probability(m(raw,mask,times).cpu().numpy()) for arm,m in models.items()}
                assert all(c.np.isfinite(p).all() for p in pn.values());q=dict(binding=pb,scores=pn,valid_windows=int(mask.sum()),complete_patient_channels=True);c.save(path,q)
            windows+=q['valid_windows'];assert q['complete_patient_channels']
            for arm in c.TRAINED:score[arm][loc]=q['scores'][arm]
        score['SELECTED']=score[choices[fold]].copy()
        for row in foldrows:
            key=f"{row['procedure']}_{fold}";path=FINAL/'outer_scores_private'/(key+'.pkl');q=dict(freeze=fh,configuration=row,indices=ix,score_nez=score[row['procedure']])
            if path.exists():
                old=c.read(path);assert old['freeze']==fh and old['configuration']==row and c.np.array_equal(old['score_nez'],q['score_nez'])
            else:c.save(path,q)
            seals[key]=c.sha(path)
        coverage.append(dict(fold=fold,patients=len(set(meta['patient'][ix])),channels=len(ix),valid_windows=windows))
        c.write(PUBLIC/'OUTER_STATUS.json',dict(status='SEALING_JOINT_RAW_SCORES_NO_FULL_TRUTH',cells=len(seals),total=30))
        print('JOINT_RAW_OUTER_FOLD_SEALED',fold,len(seals),flush=True);del models;gc.collect();c.torch.cuda.empty_cache()
    assert len(seals)==30 and sum(r['patients'] for r in coverage)==80 and sum(r['channels'] for r in coverage)==7635 and sum(r['valid_windows'] for r in coverage)==1471965
    c.freeze(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,cells=seals,all30_scores_before_complete_Outer_truth=True))
    c.write(PUBLIC/'OUTER_STATUS.json',dict(status='ALL30_FROZEN_JOINT_RAW_SCORE_CELLS_SEALED',cells=30,coverage=coverage,
        score_seal_sha256=c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),fullOuter_truth_read=False,primary='SELECTED'))


def sealed_labels():
    status=c.jr(PUBLIC/'OUTER_STATUS.json');assert status['status']=='ALL30_FROZEN_JOINT_RAW_SCORE_CELLS_SEALED'
    assert status['score_seal_sha256']==c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');seal=c.jr(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert len(seal['cells'])==30 and seal['all30_scores_before_complete_Outer_truth'] and seal['freeze']==c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    meta,banks=c.io.identity_and_banks(labels=False);archive=c.Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert c.sha(archive)==c.read(c.io.BASE/'RECONSTRUCTION_PRIVATE.pkl')['binding']['archive']
    with c.np.load(archive,allow_pickle=True) as z:
        assert c.np.array_equal(z['patient'],meta['patient']) and c.np.array_equal(z['channel'],meta['channel']);y=1-z['y']
    assert len(y)==7635 and y.sum()==1743;return seal,meta,banks,y


def report():
    setup();seal,meta,banks,y=sealed_labels();rows=[];pooled=[]
    for procedure in PROCEDURES:
        indices=[];truth=[];prob=[];predictions=[]
        for fold in range(1,6):
            path=FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl';assert c.sha(path)==seal['cells'][f'{procedure}_{fold}'];q=c.read(path)
            assert q['freeze']==seal['freeze'];ix=q['indices'];pn=q['score_nez'];pp=1-pn.astype(float);pred=pn<q['configuration']['threshold']
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test'];indices.extend(ix);truth.extend(y[ix]);prob.extend(pp);predictions.extend(pred)
            for pid in sorted(set(meta['patient'][ix])):
                j=meta['patient'][ix]==pid;ai=ix[j];metric=dict(**c.parent.ev.classification(y[ai],pp[j],pred[j]),**c.parent.ev.ranking(y[ai],pp[j],meta['channel'][ai]))
                rows.append(dict(procedure=procedure,fold=fold,patient=str(pid),center=str(meta['center'][ai][0]),**metric))
        assert len(indices)==len(set(indices))==7635 and set(indices)==set(range(7635));pooled.append(dict(procedure=procedure,**c.parent.ev.classification(truth,prob,predictions)))
    frame=c.pd.DataFrame(rows);assert len(frame)==480 and all(frame[frame.procedure==p].patient.nunique()==80 for p in PROCEDURES)
    c.save(FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl',rows);frame.groupby('procedure')[c.METRICS].mean().reset_index().to_csv(PUBLIC/'OUTER_PATIENT_MEAN.csv',index=False)
    c.pd.DataFrame(pooled).to_csv(PUBLIC/'OUTER_POOLED.csv',index=False);ff=frame.groupby(['procedure','fold'])[c.METRICS].mean().reset_index();ff.to_csv(PUBLIC/'OUTER_FOLDS.csv',index=False)
    ff.groupby('procedure')[c.METRICS].std(ddof=1).reset_index().to_csv(PUBLIC/'OUTER_FOLD_SD.csv',index=False)
    source=[];absolute=[];paired=[];coverage=[]
    for (procedure,center),g in frame.groupby(['procedure','center']):
        n=g.patient.nunique();source.append(dict(procedure=procedure,center=center,patients=n,suppressed=n<5,**{m:float(g[m].mean()) if n>=5 else c.np.nan for m in c.METRICS}))
    for procedure,g in frame.groupby('procedure'):
        absolute.extend([dict(procedure=procedure,**r) for r in c.parent.cluster_interval(g.to_dict('records'))])
        coverage.extend([dict(procedure=procedure,metric=m,finite_patients=int(c.np.isfinite(g[m]).sum()),total_patients=80) for m in c.METRICS])
    pairs=[(p,'NATIVE') for p in PROCEDURES if p!='NATIVE']+[('PRE_JOINT','PRE_MEAN_DUP'),('PRE_JOINT','PRE_ENERGY_DUP'),('PRE_ENERGY_DUP','PRE_MEAN_DUP')]
    for left,right in pairs:
        q=frame[frame.procedure==left].merge(frame[frame.procedure==right],on=['patient','center','fold'],suffixes=('_l','_r'),validate='one_to_one')
        diff=c.pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in c.METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in c.parent.cluster_interval(diff.to_dict('records'))])
    c.pd.DataFrame(source).to_csv(PUBLIC/'OUTER_SOURCE.csv',index=False);c.pd.DataFrame(absolute).to_csv(PUBLIC/'OUTER_ABSOLUTE_CI.csv',index=False)
    c.pd.DataFrame(paired).to_csv(PUBLIC/'OUTER_PAIRED_CI.csv',index=False);c.pd.DataFrame(coverage).to_csv(PUBLIC/'OUTER_METRIC_COVERAGE.csv',index=False)
    points={p:float(frame[frame.procedure==p].macro_f1.mean()) for p in PROCEDURES};assert abs(points['NATIVE']-.6304930529317453)<1e-12 and abs(points['FROZEN_PRE']-.6314324049383006)<1e-12
    c.write(PUBLIC/'FINAL_TERMINAL.json',dict(status='COMPLETE_SIX_JOINT_RAW_PROCEDURES',patient_equal_macro_f1=points,primary='SELECTED',primary_point_over_065=points['SELECTED']>.65,
        goal_requires_independent_completion_audit=True,postOuter_tuning=False,outer_oracle=False,scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False))
    print('JOINT_RAW_FINAL_REPORTED',points,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['admit','fit','test','report']);args=parser.parse_args()
    dict(admit=admit,fit=fit,test=test,report=report)[args.action]()
