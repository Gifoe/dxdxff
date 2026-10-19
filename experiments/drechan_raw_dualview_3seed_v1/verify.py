"""Independent selection replays, sklearn endpoints and scalar-bootstrap checks."""
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score
from common import *

def table(n):
    with (PUB/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def selection():
    binding,core,fac,records,rec,banks=setup();fr=jr(ROOT/'MODEL_FREEZE_PRIVATE.json');assert fr['input']==binding
    assert {(r['arm'],r['seed'],r['fold']) for r in fr['cells']}==cells_expected();inputs=store();checks=[];anchorchecks=[]
    for fold in range(1,6):
        d,scale=data(fold,core,rec,banks,inputs,binding)
        for seed in SEEDS:
            for kind in ['pre','no_pre']:
                a=read(anchor_path(fold,seed,kind))
                expected=next(z for z in table('ANCHOR_ADMISSION') if int(z['fold'])==fold and int(z['seed'])==seed and z['kind']==kind)
                assert sha(anchor_path(fold,seed,kind))==expected['anchor_sha256']
                initial=initialize(core,fac,records,fold,'FULL' if kind=='pre' else 'WO_PRE_REFERENCE',seed)
                assert state_hash(initial.state_dict())==a['binding']['initial'] and a['binding']['kind']==kind
                del initial
                original=parent.make(core,fac,records,fold,'JOINT_PRUNED',seed)
                original.raw_pr=False;original.collect_anchor=True;original.pre_reference=kind=='pre';original.eval()
                arrays=[];windows=0
                with torch.no_grad():
                    for pid in sorted(d['roles']['train']):
                        ii=d['train'][d['patient'][d['train']]==pid]
                        raw,mask,times,delta,_=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
                        value,_=original.features(raw,mask,times,delta,scale)
                        arrays.append(value.double().cpu());windows+=int(mask.sum());del raw,delta,value
                x=torch.cat(arrays);mu=x.mean(0);sd=x.std(0,unbiased=False);sigma=torch.where(sd>1e-8,sd,torch.ones_like(sd))
                assert torch.equal(mu,a['mean']) and torch.equal(sigma,a['scale']) and windows==a['valid_windows']
                assert a['binding']['train_patients']==digest(sorted(d['roles']['train'])) and a['binding']['train_indices']==legacy.array_sha(d['archive_indices'][d['train']])
                anchorchecks.append(dict(fold=fold,seed=seed,kind=kind,initial_TRAIN_anchor_bit_exact=True,channels=len(x),valid_windows=windows))
                del original,x;gc.collect();torch.cuda.empty_cache()
    csv_out('ANCHOR_REPLAY',anchorchecks)
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];d,scale=data(fold,core,rec,banks,inputs,binding)
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
        state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');v=read(r['calibration']);m=make(core,fac,records,fold,arm,seed)
        assert state_hash(m.state_dict())==v['binding']['initial'] and preprocessor(d,scale,arm,fold,seed)==r['preprocessor']
        assert sum(p.numel() for p in m.parameters())==r['parameters']==MODEL.PARAMETERS[arm] and not m.interaction and m.keep7[4]==0
        m.load_state_dict(state['model']);before=state_hash(m.state_dict());out,_=predict(m,d,d['val'],inputs,scale)
        if arm not in REFERENCES:
            anchor=read(anchor_path(fold,seed,anchor_kind(arm)));assert torch.equal(m.anchor_mean.cpu(),anchor['mean']) and torch.equal(m.anchor_scale.cpu(),anchor['scale'])
        assert np.array_equal(out,v['logit']) and np.array_equal(d['archive_indices'][d['val']],v['archive_indices'])
        sel=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
        assert all(abs(float(value)-float(state[k]))<1e-12 for k,value in sel.items())
        assert state['epoch']==r['epoch'] and state['threshold']==r['threshold'] and state_hash(m.state_dict())==before
        winner=max(v['history'],key=lambda q:(q['macro_f1'],q['ez_f1'],-q['epoch']))
        assert winner['epoch']==r['epoch'] and len(v['history'])==r['epochs'] and 6<=len(v['history'])<=30
        if r['reused']:
            old=reference_cell(arm,seed,fold);assert arm in REFERENCES and r['origin_arm']==REFERENCES[arm]
            assert old['checkpoint_sha256']==r['checkpoint_sha256'] and old['calibration_sha256']==r['calibration_sha256']
            assert old['preprocessor']==r['preprocessor'] and state['binding']==v['binding']
        else:assert state['binding']==v['binding'] and state['binding']['input']==binding and state['binding']['preprocessor']==r['preprocessor']
        checks.append(dict(arm=arm,seed=seed,fold=fold,CAL_logits_bit_exact=True,selected_epoch=r['epoch'],threshold=r['threshold'],reused=r['reused']))
        del m;gc.collect();torch.cuda.empty_cache();print('RDV_SELECTED_CELL_PASS',arm,seed,fold,flush=True)
    csv_out('CALIBRATION_REPLAY',checks)
    write(PUB/'SELECTION_VERIFICATION.json',dict(status='PASS',cells=120,initial_TRAIN_anchor_replays=30,all_CAL_bit_exact=True,TRAIN_only_preprocessing=True,ASSESS_not_used_for_selection=True))
    freeze(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json',dict(input=binding,source=sources(),cells=fr['cells'],model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),selection_verification=sha(PUB/'SELECTION_VERIFICATION.json')))
    write(PUB/'FREEZE_STATUS.json',dict(status='ALL120_FROZEN',freeze_sha256=sha(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json')))

def metrics():
    binding,core,fac,records,rec,banks=setup();fr=frozen();seal=jr(ROOT/'SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==sha(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json') and len(seal['files'])==120 and seal['all120_before_metrics']
    stored=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl');oldrows={(r['arm'],r['seed'],r['fold'],r['patient']):r for r in stored};rows=[];pooled={};counts={};inputs=store();coverage=table('COVERAGE')
    assert len(stored)==len(oldrows)==960 and len(coverage)==120;meta=rec['meta']
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];key=f'{arm}_seed{seed}_fold{fold}';path=ROOT/'scores_private'/(key+'.pkl');assert sha(path)==seal['files'][key]
        q=read(path);ii=q['indices'];pp=1-q['score_nez'].astype(np.float64);vv=q['score_nez']<r['threshold'];b=banks[fold-1]
        wanted=prior.roles_for(b,fold,'internal')['assessment'];assert np.array_equal(ii,np.flatnonzero(np.isin(meta['patient'],list(wanted))))
        assert q['binding']==dict(input=digest(binding),freeze=seal['freeze'],checkpoint=r['checkpoint_sha256'],indices=legacy.array_sha(ii),preprocessor=r['preprocessor'])
        assert q['threshold']==r['threshold'];cr=next(z for z in coverage if (z['arm'],int(z['seed']),int(z['fold']))==(arm,seed,fold))
        assert int(cr['patients'])==len(wanted) and int(cr['channels'])==len(ii) and (cr['reused']=='True')==r['reused']
        if fold not in counts:
            total=0
            for pid in sorted(wanted):
                j=ii[meta['patient'][ii]==pid];raw,mask,_,delta,_=inputs.get(str(pid),meta['channel'][j].astype(str).tolist(),'cpu');total+=int(mask.sum());del raw,delta
            counts[fold]=total
        assert int(cr['valid_windows'])==counts[fold]
        allowed=np.flatnonzero(np.isin(b['patient'],list(b['roles']['fit'])));labels={int(i):int(1-y) for i,y in zip(b['archive_indices'][allowed],b['y'][allowed])};yy=np.array([labels[int(i)] for i in ii])
        pooled.setdefault((arm,seed),dict(y=[],p=[],v=[]))
        for k,z in [('y',yy),('p',pp),('v',vv)]:pooled[(arm,seed)][k].append(z)
        for pid in sorted(wanted):
            j=meta['patient'][ii]==pid;y,p,v=yy[j],pp[j],vv[j];ch=meta['channel'][ii][j].astype(str)
            order=sorted(range(len(y)),key=lambda k:(-float(p[k]),ch[k]));first=next(k for k,i in enumerate(order) if y[i])
            values=dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),f1=f1_score(y,v,zero_division=0),macro_f1=f1_score(y,v,labels=[0,1],average='macro',zero_division=0),
                        nez_f1=f1_score(1-y,1-v.astype(int),zero_division=0),accuracy=accuracy_score(y,v),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(v.mean()-y.mean()))
            expected=oldrows[(arm,seed,fold,str(pid))]
            assert all(np.isfinite(value) and abs(expected[m]-value)<1e-12 for m,value in values.items())
            rows.append(dict(arm=arm,seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),**values))
    assert len(rows)==960 and {(r['arm'],r['seed'],r['fold'],r['patient']) for r in rows}==set(oldrows)
    ids=sorted({r['patient'] for r in rows});assert len(ids)==33;byseed={};means={}
    for arm in ARMS:
        for seed in SEEDS:
            byseed[(arm,seed)]=np.array([np.mean([[r[m] for m in METRICS] for r in rows if (r['arm'],r['seed'],r['patient'])==(arm,seed,p)],axis=0) for p in ids])
        means[arm]=sum(byseed[(arm,s)] for s in SEEDS)/3
    main,seeds,sd,folds,foldsd,centers=[table(n) for n in ['MAIN_RESULTS','SEED_RESULTS','SEED_SD','FOLD_RESULTS','FOLD_SD','SOURCE_RESULTS']]
    assert len(main)==len(sd)==8 and len(seeds)==len(foldsd)==24 and len(folds)==120
    for arm in ARMS:
        rr=next(r for r in main if r['arm']==arm);assert int(rr['unique_patients'])==33
        assert all(abs(float(rr[m])-means[arm][:,j].mean())<1e-12 for j,m in enumerate(METRICS));ss=[]
        for seed in SEEDS:
            value=byseed[(arm,seed)].mean(0);ss.append(value);rr=next(r for r in seeds if r['arm']==arm and int(r['seed'])==seed)
            assert all(abs(float(rr[m])-value[j])<1e-12 for j,m in enumerate(METRICS));ff=[]
            for fold in range(1,6):
                z=[r for r in rows if (r['arm'],r['seed'],r['fold'])==(arm,seed,fold)];value=vector(z).mean(0);ff.append(value)
                rr=next(r for r in folds if (r['arm'],int(r['seed']),int(r['fold']))==(arm,seed,fold))
                assert int(rr['patients'])==len(z) and all(abs(float(rr[m])-value[j])<1e-12 for j,m in enumerate(METRICS))
            rr=next(r for r in foldsd if r['arm']==arm and int(r['seed'])==seed);assert all(abs(float(rr[m])-np.std(ff,axis=0,ddof=1)[j])<1e-12 for j,m in enumerate(METRICS))
        rr=next(r for r in sd if r['arm']==arm);assert all(abs(float(rr[m])-np.std(ss,axis=0,ddof=1)[j])<1e-12 for j,m in enumerate(METRICS))
        for center in sorted({r['center'] for r in rows}):
            keep=[i for i,p in enumerate(ids) if next(r for r in rows if r['patient']==p)['center']==center];rr=next(r for r in centers if r['arm']==arm and r['center']==center)
            assert int(rr['unique_patients'])==len(keep) and (rr['suppressed']=='True')==(len(keep)<5)
            assert all(abs(float(rr[m])-means[arm][keep,j].mean())<1e-12 if len(keep)>=5 else np.isnan(float(rr[m])) for j,m in enumerate(METRICS))
    draw=np.random.default_rng(42).integers(0,33,(10000,33));ci=table('INTERVALS');family=table('PRIMARY_FAMILY_INTERVALS');assert len(ci)==144 and len(family)==12;maxerr=0
    definitions=[('absolute',a,[(a,1)]) for a in ARMS]+[('paired',n,list(w.items())) for n,w in CONTRASTS.items()]
    for kind,name,weights in definitions:
        for j,m in enumerate(METRICS):
            x=sum(means[a][:,j]*w for a,w in weights);samples=x[draw].mean(1);limits=np.quantile(samples,[.025,.975]);rr=next(r for r in ci if (r['kind'],r['contrast'],r['metric'])==(kind,name,m))
            err=max(abs(float(rr['mean'])-x.mean()),abs(float(rr['low'])-limits[0]),abs(float(rr['high'])-limits[1]));assert err<1e-12;maxerr=max(maxerr,err)
            if name in PRIMARY_CONTRASTS and m in PRIMARY_METRICS:
                rr=next(r for r in family if (r['contrast'],r['metric'])==(name,m));lim=np.quantile(samples,[.05/24,1-.05/24]);assert int(rr['comparisons'])==12
                assert max(abs(float(rr['mean'])-x.mean()),abs(float(rr['low'])-lim[0]),abs(float(rr['high'])-lim[1]))<1e-12
    for rr in table('COMPONENT_EVIDENCE'):
        w=CONTRASTS[rr['contrast']];indices=[METRICS.index(m) for m in PRIMARY_METRICS];point=sum(means[a].mean(0)*v for a,v in w.items())
        count=sum(all(sum(byseed[(a,s)][:,j].mean()*v for a,v in w.items())>0 for j in indices) for s in SEEDS)
        assert int(rr['seeds_both_positive'])==count and (rr['stable_positive_point_evidence']=='True')==(count>=2 and all(point[j]>0 for j in indices))
        nominal=all(float(next(z for z in ci if z['contrast']==rr['contrast'] and z['metric']==m)['low'])>0 for m in PRIMARY_METRICS)
        simultaneous=all(float(next(z for z in family if z['contrast']==rr['contrast'] and z['metric']==m)['low'])>0 for m in PRIMARY_METRICS)
        assert (rr['both_nominal_lowers_positive']=='True')==nominal and (rr['both_family_lowers_positive']=='True')==simultaneous
    gate=jr(PUB/'QUALIFICATION_GATE.json');idx=[METRICS.index(m) for m in ['auroc','auprc','macro_f1','f1']]
    d0=(means['FULL']-means['B0']).mean(0);sd0=[(byseed[('FULL',s)]-byseed[('B0',s)]).mean(0) for s in SEEDS]
    perf=bool(d0[idx[0]]>0 and all(d0[j]>=-.01 for j in idx[1:]) and sum(x[idx[0]]>0 and all(x[j]>=-.01 for j in idx[1:]) for x in sd0)>=2)
    checks={r['contrast']:r['stable_positive_point_evidence']=='True' for r in table('COMPONENT_EVIDENCE')}
    pr=checks['FEATURE_PR_WITH_RAW'] and checks['RAW_PR_WITH_FEATURE']
    expanded=checks['FULL_MINUS_PRUNED_REF'];global_view=checks['GLOBAL_VIEW'];pre=checks['PRE_REFERENCE']
    assert gate['performance_margin_pass']==perf and gate['both_separate_PR_point_rules_pass']==pr
    assert gate['expanded_reference_point_rule_pass']==expanded and gate['global_view_point_rule_pass']==global_view and gate['pre_reference_point_rule_pass']==pre
    qualified=bool(perf and pr and expanded and global_view and pre)
    assert gate['qualified_for_final_study_planning']==qualified and gate['status']==('PASS' if qualified else 'FAIL')
    assert not gate['new_Outer_authorized'] and not gate['active_goal_achieved']
    parentrows={r['arm']:r for r in csv.DictReader((parent.PUB/'MAIN_RESULTS.csv').open(newline='',encoding='utf-8'))}
    for arm,oldarm in REFERENCES.items():
        assert all(abs(means[arm][:,j].mean()-float(parentrows[oldarm][m]))<1e-12 for j,m in enumerate(METRICS))
    pooledrows=[]
    for (arm,seed),q in pooled.items():
        y,p,v=[np.concatenate(q[k]) for k in ['y','p','v']]
        pooledrows.append(dict(arm=arm,seed=seed,channel_appearances=len(y),aggregation='POOLED_APPEARANCES_NOT_PRIMARY',macro_f1=f1_score(y,v,average='macro'),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),
                               f1=f1_score(y,v),nez_f1=f1_score(1-y,1-v.astype(int)),accuracy=accuracy_score(y,v),ez_fraction_bias=float(v.mean()-y.mean())))
    csv_out('POOLED_RESULTS',pooledrows)
    write(PUB/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',patient_fold_seed_rows=960,patient_metric_checks=8640,unique_patients=33,nominal_interval_checks=144,primary_family_interval_checks=12,
                                                  maximum_interval_discrepancy=maxerr,all_roles_and_raw_window_coverage_verified=True,no_new_Outer=True,active_goal_achieved=False))
    print('RDV_INDEPENDENT_ALL960_8640_144_12_PASS',flush=True)

if __name__=='__main__':{'selection':selection,'metrics':metrics}[sys.argv[1]]()
