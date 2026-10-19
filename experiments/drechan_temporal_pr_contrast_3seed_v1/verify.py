"""Independent selection replays and independent endpoints/gates/bootstrap."""
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score
from common import *

def table(stage,name):
    with (PUB/stage/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def selection(stage):
    binding,core,fac,records,rec,banks=setup();fr=jr(ROOT/f'{stage}_MODEL_FREEZE_PRIVATE.json')
    assert fr['input']==binding and {(r['arm'],r['seed'],r['fold']) for r in fr['cells']}==cells_expected()
    inputs=store();checks=[]
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];d,scale=data(fold,stage,core,rec,banks,inputs,binding)
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
        m=make(core,fac,records,fold,arm,seed);state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');v=read(r['calibration'])
        assert state_hash(m.state_dict())==v['binding']['initial'];m.load_state_dict(state['model']);m.set_reference(state.get('reference'))
        assert preprocessor(d,scale)==r['preprocessor'] and not m.interaction and m.keep7[4]==0
        assert sum(p.numel() for p in m.parameters())==r['parameters']==(11425 if arm=='B0' else 13201)
        assert reference_hash(m.reference)==r['reference_hash']
        if arm=='B2':
            g=MODEL.GlobalReference(m,d,inputs,scale);replayed=g.fit(m)
            assert set(g.fit_ids)==d['roles']['train'] and reference_hash(replayed)==r['reference_hash'];del g
        before=state_hash(m.state_dict());out,_=predict(m,d,d['val'],inputs,scale)
        assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
        selected=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
        assert all(abs(float(value)-float(state[k]))<1e-12 for k,value in selected.items())
        assert state['epoch']==r['epoch'] and state['threshold']==r['threshold'] and state_hash(m.state_dict())==before
        if not r['reused']:
            assert state['binding']==v['binding'] and state['binding']['input']==binding
            assert state['binding']['preprocessor']==r['preprocessor']
            winner=max(v['history'],key=lambda q:(q['macro_f1'],q['ez_f1'],-q['epoch']))
            assert winner['epoch']==state['epoch'] and 6<=len(v['history'])<=30
            for h in v['history']:
                assert h['global_reference_refreshes']==(len(d['roles']['train'])+1 if arm=='B2' else 0)
        checks.append(dict(arm=arm,seed=seed,fold=fold,validation_logits_bit_exact=True,selected_epoch=r['epoch'],threshold=r['threshold'],reference_replayed=arm=='B2'))
        del m;gc.collect();torch.cuda.empty_cache();print('TPRC_SELECTED_CELL_PASS',stage,arm,seed,fold,flush=True)
    csv_out(stage+'/CALIBRATION_REPLAY',checks)
    write(PUB/stage/'SELECTION_VERIFICATION.json',dict(status='PASS',cells=60,all_calibration_logits_bit_exact=True,current_TRAIN_reference_replays=15,
        all_original_FIT_inner_roles_separate=stage=='internal',no_own_assessment_selection=True,all_epochs_thresholds_preprocessors_verified=True))
    freeze(ROOT/f'{stage}_FROZEN_BEFORE_SCORES_PRIVATE.json',dict(input=binding,source=sources(),cells=fr['cells'],selection_verification=sha(PUB/stage/'SELECTION_VERIFICATION.json'),
        model_freeze_sha256=sha(ROOT/f'{stage}_MODEL_FREEZE_PRIVATE.json'),all60_cells_frozen_before_scores=True))
    write(PUB/stage/'FREEZE_STATUS.json',dict(status='ALL60_FROZEN_BEFORE_SCORES',cells=60,freeze_sha256=sha(ROOT/f'{stage}_FROZEN_BEFORE_SCORES_PRIVATE.json')))
    print('TPRC_'+stage.upper()+'_ALL60_SELECTIONS_VERIFIED_AND_FROZEN',flush=True)

def metrics(stage):
    binding,core,fac,records,rec,banks=setup();fr=frozen(stage);seal=jr(ROOT/f'{stage}_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==sha(ROOT/f'{stage}_FROZEN_BEFORE_SCORES_PRIVATE.json') and len(seal['files'])==60 and seal['all60_cells_sealed_before_metrics']
    stored=read(ROOT/f'{stage}_PATIENT_METRICS_PRIVATE.pkl');oldrows={(r['arm'],r['seed'],r['fold'],r['patient']):r for r in stored};rows=[];checks=0;meta=rec['meta'];pooled={}
    assert len(oldrows)==len(stored)
    inputs=store();coverage=table(stage,'COVERAGE');assert len(coverage)==60;raw_counts={};expected_keys=set()
    if stage=='final':
        archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
        with np.load(archive,allow_pickle=True) as q:final_y=1-q['y']
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];key=f'{arm}_seed{seed}_fold{fold}';path=ROOT/'scores_private'/stage/(key+'.pkl');assert sha(path)==seal['files'][key]
        q=read(path);ii=q['indices'];pp=1-q['score_nez'].astype(np.float64);vv=q['score_nez']<r['threshold']
        assert q['binding']['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        assert q['binding']['input']==digest(binding) and q['binding']['freeze']==seal['freeze']
        assert q['binding']['reference_hash']==r['reference_hash'] and q['binding']['preprocessor']==r['preprocessor']
        assert q['binding']['indices']==legacy.array_sha(ii) and len(set(ii))==len(ii)
        b=parent.bank(fold,banks,'FULL_F0');roles=roles_for(b,fold,stage)
        wanted=roles['assessment'] if stage=='internal' else roles['test']
        expected_indices=np.flatnonzero(np.isin(meta['patient'],list(wanted)))
        assert np.array_equal(ii,expected_indices)
        expected_keys.update((arm,seed,fold,str(pid)) for pid in wanted)
        cr=next(z for z in coverage if (z['arm'],int(z['seed']),int(z['fold']))==(arm,seed,fold))
        assert int(cr['patients'])==len(wanted) and int(cr['channels'])==len(ii) and (cr['reused']=='True')==r['reused']
        if fold not in raw_counts:
            count=post=seizures=0
            for pid in sorted(wanted):
                ix=ii[meta['patient'][ii]==pid];raw,mask,times,delta,_=inputs.get(str(pid),meta['channel'][ix].astype(str).tolist(),'cpu')
                count+=int(mask.sum());post+=int((mask&((times>=0)&(times<20))[:,None,:]).sum());seizures+=raw.shape[0];del raw,delta
            raw_counts[fold]=(count,post,seizures)
        count,post,seizures=raw_counts[fold]
        if r['reused']:assert stage=='final' and arm=='B0' and seed==42
        else:
            assert int(cr['valid_windows'])==count
            if arm!='B0':
                assert int(cr['post_valid_windows'])==post
                assert int(cr['reference_valid_coordinates'])+int(cr['reference_fallback_coordinates'])==post*24
                assert 0<=int(cr['q_clipped_coordinates'])<=int(cr['reference_valid_coordinates'])
                if arm=='B3':assert int(cr['time_shifted_valid_windows'])+int(cr['time_singleton_valid_windows'])==post
        if stage=='internal':
            b=banks[fold-1];roles=roles_for(b,fold,'internal');assert set(meta['patient'][ii])==roles['assessment']
            fit=np.flatnonzero(np.isin(b['patient'],list(b['roles']['fit'])));labels={int(j):int(1-y) for j,y in zip(b['archive_indices'][fit],b['y'][fit])}
            yy=np.array([labels[int(j)] for j in ii])
        else:
            yy=final_y[ii];assert len(set(meta['patient'][ii]))==[16,16,17,15,16][fold-1]
        pooled.setdefault((arm,seed),dict(y=[],p=[],v=[]))
        for k,z in [('y',yy),('p',pp),('v',vv)]:pooled[(arm,seed)][k].append(z)
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid;y=yy[j];p=pp[j];v=vv[j];ch=meta['channel'][ii][j].astype(str)
            order=sorted(range(len(y)),key=lambda k:(-float(p[k]),ch[k]));first=next(k for k,i in enumerate(order) if y[i])
            vals=dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),f1=f1_score(y,v,zero_division=0),macro_f1=f1_score(y,v,labels=[0,1],average='macro',zero_division=0),
                nez_f1=f1_score(1-y,1-v.astype(int),zero_division=0),accuracy=accuracy_score(y,v),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(v.mean()-y.mean()))
            expected=oldrows[(arm,seed,fold,str(pid))]
            for m,value in vals.items():assert np.isfinite(value) and abs(expected[m]-value)<1e-12;checks+=1
            rows.append(dict(arm=arm,seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),**vals))
    assert len(rows)==len(oldrows)==len(stored)==len(expected_keys) and set(oldrows)==expected_keys and checks==len(rows)*9
    if stage=='final':
        assert len(rows)==960 and len({r['patient'] for r in rows})==80
        assert sum(v[0] for v in raw_counts.values())==1471965 and sum(v[2] for v in raw_counts.values())==256
        assert sum(int(r['channels']) for r in coverage if r['arm']=='B1' and int(r['seed'])==42)==7635
    else:assert len(rows)==480
    ids=sorted({r['patient'] for r in rows});matrices={};seedvalues={};overall={}
    for arm in ARMS:
        for seed in SEEDS:
            v=[]
            for pid in ids:
                z=[r for r in rows if (r['arm'],r['seed'],r['patient'])==(arm,seed,pid)];assert z
                v.append([[r[m] for m in METRICS] for r in z])
            matrices[(arm,seed)]=np.array([np.mean(q,axis=0) for q in v]);seedvalues[(arm,seed)]=matrices[(arm,seed)].mean(0)
        overall[arm]=sum(matrices[(arm,s)] for s in SEEDS)/3
    main=table(stage,'MAIN_RESULTS');seeds=table(stage,'SEED_RESULTS');sd=table(stage,'SEED_SD');folds=table(stage,'FOLD_RESULTS');foldsd=table(stage,'FOLD_SD');centers=table(stage,'SOURCE_RESULTS')
    assert len(main)==4 and len(seeds)==12 and len(sd)==4 and len(folds)==60 and len(foldsd)==12
    for arm in ARMS:
        row=next(r for r in main if r['arm']==arm);assert int(row['unique_patients'])==len(ids)
        assert max(abs(float(row[m])-overall[arm][:,j].mean()) for j,m in enumerate(METRICS))<1e-12
        for seed in SEEDS:
            row=next(r for r in seeds if r['arm']==arm and int(r['seed'])==seed)
            assert max(abs(float(row[m])-seedvalues[(arm,seed)][j]) for j,m in enumerate(METRICS))<1e-12
            ff=[]
            for fold in range(1,6):
                z=[r for r in rows if (r['arm'],r['seed'],r['fold'])==(arm,seed,fold)];value=vector(z).mean(0);ff.append(value)
                row=next(r for r in folds if r['arm']==arm and int(r['seed'])==seed and int(r['fold'])==fold)
                assert int(row['patients'])==len(z) and max(abs(float(row[m])-value[j]) for j,m in enumerate(METRICS))<1e-12
            row=next(r for r in foldsd if r['arm']==arm and int(r['seed'])==seed)
            assert max(abs(float(row[m])-np.std(ff,axis=0,ddof=1)[j]) for j,m in enumerate(METRICS))<1e-12
        row=next(r for r in sd if r['arm']==arm)
        assert max(abs(float(row[m])-np.std([seedvalues[(arm,s)] for s in SEEDS],axis=0,ddof=1)[j]) for j,m in enumerate(METRICS))<1e-12
        for center in sorted({r['center'] for r in rows}):
            keep=[i for i,pid in enumerate(ids) if next(r for r in rows if r['patient']==pid)['center']==center]
            row=next(r for r in centers if r['arm']==arm and r['center']==center)
            assert int(row['unique_patients'])==len(keep) and (row['suppressed']=='True')==(len(keep)<5)
            if len(keep)>=5:assert max(abs(float(row[m])-overall[arm][keep,j].mean()) for j,m in enumerate(METRICS))<1e-12
            else:assert all(np.isnan(float(row[m])) for m in METRICS)
    ci=table(stage,'INTERVALS');family=table(stage,'PRIMARY_FAMILY_INTERVALS');assert len(ci)==63 and len(family)==12
    definitions=[('absolute',a,[(a,1)]) for a in ARMS]+[('paired','B1_minus_'+a,[('B1',1),(a,-1)]) for a in ['B0','B2','B3']]
    draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)));maximum=0.;intervalchecks=0
    for kind,name,weights in definitions:
        for j,m in enumerate(METRICS):
            v=sum(overall[a][:,j]*w for a,w in weights);samples=v[draw].mean(1);limits=np.quantile(samples,[.025,.975]);mean=v.mean()
            row=next(r for r in ci if r['kind']==kind and r['contrast']==name and r['metric']==m)
            error=max(abs(float(row['mean'])-mean),abs(float(row['low'])-limits[0]),abs(float(row['high'])-limits[1]));assert error<1e-12
            maximum=max(maximum,error);intervalchecks+=1
            if kind=='paired' and m in PRIMARY:
                row=next(r for r in family if r['contrast']==name and r['metric']==m);lim=np.quantile(samples,[.05/24,1-.05/24])
                assert int(row['comparisons'])==12 and max(abs(float(row['mean'])-mean),abs(float(row['low'])-lim[0]),abs(float(row['high'])-lim[1]))<1e-12
    assert intervalchecks==63
    if stage=='internal':
        g=jr(PUB/'internal/GATE.json');idx=[METRICS.index(m) for m in PRIMARY]
        d0=(overall['B1']-overall['B0']).mean(0);d2=(overall['B1']-overall['B2']).mean(0)
        seeds0=sum(all((seedvalues[('B1',s)]-seedvalues[('B0',s)])[j]>=-.01 for j in idx) for s in SEEDS)
        seeds2=sum(all((seedvalues[('B1',s)]-seedvalues[('B2',s)])[METRICS.index(m)]>0 for m in ['auroc','auprc']) for s in SEEDS)
        acceptable=all(d0[j]>=-.01 for j in idx) and seeds0>=2
        mechanism=all(d2[METRICS.index(m)]>0 for m in ['auroc','auprc']) and all(d2[METRICS.index(m)]>=-.01 for m in ['macro_f1','f1']) and seeds2>=2
        assert bool(g['performance_margin_pass'])==acceptable and bool(g['global_reference_comparison_pass'])==mechanism and bool(g['continue_final'])==(acceptable and mechanism)
        assert g['seeds_meeting_all_four_margins']==seeds0 and g['seeds_B1_exceeding_B2_both_ranking_metrics']==seeds2
        for j,m in enumerate(METRICS):assert abs(g['B1_minus_B0'][m]-d0[j])<1e-12 and abs(g['B1_minus_B2'][m]-d2[j])<1e-12
    pooledrows=[]
    for (arm,seed),q in pooled.items():
        y=np.concatenate(q['y']);p=np.concatenate(q['p']);v=np.concatenate(q['v'])
        pooledrows.append(dict(arm=arm,seed=seed,channel_appearances=len(y),aggregation='POOLED_APPEARANCES_NOT_PRIMARY',macro_f1=f1_score(y,v,average='macro'),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),
            f1=f1_score(y,v),nez_f1=f1_score(1-y,1-v.astype(int)),accuracy=accuracy_score(y,v),ez_fraction_bias=float(v.mean()-y.mean())))
    csv_out(stage+'/POOLED_RESULTS',pooledrows)
    write(PUB/stage/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',patient_fold_seed_rows=len(rows),unique_patients=len(ids),patient_metric_checks=checks,
        nominal_interval_checks=63,primary_family_interval_checks=12,maximum_interval_discrepancy=maximum,all4_arms_all3_seeds_all5_folds=True,
        source_preprocessing_selection_and_reference_replay=True,assessment_is_not_epoch_or_threshold_selection=True,all_expected_role_indices_and_raw_coverage_verified=True,
        scientific_status='EXPLORATORY_REPEATED_TEST'))
    print('TPRC_'+stage.upper()+'_INDEPENDENT_METRIC_GATE_BOOTSTRAP_PASS',flush=True)

if __name__=='__main__':
    {'selection':selection,'metrics':metrics}[sys.argv[1]](sys.argv[2])
