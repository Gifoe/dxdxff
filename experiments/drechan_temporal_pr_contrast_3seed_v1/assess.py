"""Frozen score sealing, unique-patient/seed estimands, fixed honest gate."""
from common import *

def score_stage(stage,binding,core,fac,records,rec,banks):
    fr=frozen(stage);fh=sha(ROOT/f'{stage}_FROZEN_BEFORE_SCORES_PRIVATE.json');inputs=store();files={};coverage=[];meta=rec['meta']
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];d,scale=data(fold,stage,core,rec,banks,inputs,binding)
        if stage=='internal':td=d;ix=d['assessment'];archive=d['archive_indices'][ix]
        else:
            archive=np.flatnonzero(np.isin(meta['patient'],list(d['roles']['test'])))
            z=core.patient_z(rec['R14'][archive],meta['patient'][archive]);pp=d['pre']
            x=((np.where(np.isnan(z),pp['imputer_statistics'],z)-pp['mean'])/pp['scale']).astype(np.float32)
            td=dict(x=x,patient=meta['patient'][archive],channel=meta['channel'][archive]);ix=np.arange(len(archive))
        key=f'{arm}_seed{seed}_fold{fold}';path=ROOT/'scores_private'/stage/(key+'.pkl')
        cb=dict(input=digest(binding),freeze=fh,checkpoint=r['checkpoint_sha256'],reference_hash=r['reference_hash'],indices=legacy.array_sha(archive),preprocessor=r['preprocessor'])
        if path.exists():q=read(path);assert q['binding']==cb
        elif r['reused']:
            assert stage=='final' and arm=='B0' and seed==42
            oldkey=f'FULL_F0_{fold}';src=PARENTROOT/'scores_private'/(oldkey+'.pkl')
            assert sha(src)==jr(PARENTROOT/'OUTER_SCORE_SEAL_PRIVATE.json')['files'][oldkey]
            oldq=read(src);assert np.array_equal(oldq['indices'],archive) and oldq['checkpoint']==r['checkpoint_sha256'] and oldq['threshold']==r['threshold']
            q=dict(binding=cb,indices=archive,score_nez=oldq['score_nez'],threshold=r['threshold'],audit=dict(reused_scientific_cell=True),source_sha256=sha(src));save(path,q)
        else:
            m=make(core,fac,records,fold,arm,seed);state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');m.load_state_dict(state['model']);m.set_reference(state['reference']);m.eval()
            out=np.empty(len(ix),np.float32);audit={};before=state_hash(m.state_dict())
            for n,pid in enumerate(sorted(set(td['patient'][ix])),1):
                j=np.flatnonzero(td['patient'][ix]==pid);ii=ix[j]
                bind=dict(**cb,patient_indices=legacy.array_sha(archive[j]),features=legacy.array_sha(td['x'][ii]))
                cache=ROOT/'patient_scores_private'/stage/key/(hashlib.sha256(str(pid).encode()).hexdigest()+'.pkl')
                if cache.exists():z=read(cache);assert z['binding']==bind
                else:
                    with torch.no_grad():logits,info=forward(m,td,ii,inputs,scale,True)
                    z=dict(binding=bind,score_nez=probability(logits.cpu().numpy()),audit=info);save(cache,z)
                out[j]=z['score_nez']
                for k,v in z['audit'].items():audit[k]=audit.get(k,0)+v
                write(PUB/'PROGRESS.json',dict(status='SEALED_PREDICTION',stage=stage,arm=arm,seed=seed,fold=fold,patient_number=n,complete_cells=len(files),total_cells=60))
            assert state_hash(m.state_dict())==before
            q=dict(binding=cb,indices=archive,score_nez=out,threshold=r['threshold'],audit=dict(reused_scientific_cell=False,**audit));save(path,q)
            del m;gc.collect();torch.cuda.empty_cache()
        assert np.array_equal(q['indices'],archive) and q['threshold']==r['threshold'] and np.isfinite(q['score_nez']).all()
        files[key]=sha(path)
        coverage.append(dict(stage=stage,arm=arm,seed=seed,fold=fold,patients=len(set(td['patient'][ix])),channels=len(ix),reused=r['reused'],
            **{k:q['audit'].get(k,0) for k in ['valid_windows','post_valid_windows','reference_valid_coordinates','reference_fallback_coordinates','q_clipped_coordinates','time_shifted_valid_windows','time_singleton_valid_windows']}))
        print('TPRC_SCORE_CELL_SEALED',stage,arm,seed,fold,len(files),60,flush=True)
    assert len(files)==60
    freeze(ROOT/f'{stage}_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,files=files,all60_cells_sealed_before_metrics=True))
    csv_out(stage+'/COVERAGE',coverage)
    write(PUB/stage/'PREDICTION_STATUS.json',dict(status='ALL60_CELLS_SEALED',freeze_sha256=fh,score_seal_sha256=sha(ROOT/f'{stage}_SCORE_SEAL_PRIVATE.json')))

def load_rows(stage,rec,banks):
    fr=frozen(stage);seal=jr(ROOT/f'{stage}_SCORE_SEAL_PRIVATE.json');fh=sha(ROOT/f'{stage}_FROZEN_BEFORE_SCORES_PRIVATE.json')
    assert seal['freeze']==fh and len(seal['files'])==60 and seal['all60_cells_sealed_before_metrics']
    rows=[];meta=rec['meta']
    if stage=='final':
        archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
        assert sha(archive)==rec['binding']['archive']
        with np.load(archive,allow_pickle=True) as q:all_y=1-q['y']
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];key=f'{arm}_seed{seed}_fold{fold}';path=ROOT/'scores_private'/stage/(key+'.pkl')
        assert sha(path)==seal['files'][key];q=read(path);ii=q['indices'];pn=q['score_nez'];p=1-pn.astype(float);v=pn<r['threshold']
        if stage=='internal':
            b=banks[fold-1];allowed_fit=set(b['roles']['fit']);j=np.flatnonzero(np.isin(b['patient'],list(allowed_fit)))
            labelmap={int(i):int(1-y) for i,y in zip(b['archive_indices'][j],b['y'][j])}
            all_y=np.full(7635,np.nan)
            for i in ii:all_y[i]=labelmap[int(i)]
        assert q['binding']['freeze']==fh and q['binding']['checkpoint']==r['checkpoint_sha256'] and np.isfinite(all_y[ii]).all()
        if stage=='internal':
            allowed=roles_for(banks[fold-1],fold,stage)['assessment'];assert set(meta['patient'][ii])==allowed
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid;y=all_y[ii][j].astype(int)
            values=dict(**EV.classification(y,p[j],v[j]),**EV.ranking(y,p[j],meta['channel'][ii][j]))
            rows.append(dict(arm=arm,seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**{m:float(values[m]) for m in METRICS}))
    assert all(np.isfinite(vector(rows)).flat);return rows

def matrix_by_seed(rows):
    ids=sorted({r['patient'] for r in rows});matrices={};collapsed=[]
    for arm in ARMS:
        for seed in SEEDS:
            result=[]
            for pid in ids:
                q=[r for r in rows if r['arm']==arm and r['seed']==seed and r['patient']==pid];assert q
                mean=vector(q).mean(0);result.append(mean)
                collapsed.append(dict(arm=arm,seed=seed,patient=pid,center=q[0]['center'],appearances=len(q),**dict(zip(METRICS,mean.tolist()))))
            matrices[(arm,seed)]=np.array(result)
    return ids,matrices,collapsed

def summarize(stage,rows):
    ids,byseed,collapsed=matrix_by_seed(rows);mean_seed={a:np.mean([byseed[(a,s)] for s in SEEDS],axis=0) for a in ARMS}
    main=[];seeds=[];sd=[];folds=[];centers=[];fold_sd=[];coverage=[]
    for arm in ARMS:
        main.append(dict(arm=arm,unique_patients=len(ids),seeds=3,patient_fold_seed_rows=sum(r['arm']==arm for r in rows),**dict(zip(METRICS,mean_seed[arm].mean(0).tolist()))))
        ss=[]
        for seed in SEEDS:
            value=byseed[(arm,seed)].mean(0);ss.append(value);seeds.append(dict(arm=arm,seed=seed,unique_patients=len(ids),**dict(zip(METRICS,value.tolist()))))
            foldvalues=[]
            for fold in range(1,6):
                rr=[r for r in rows if r['arm']==arm and r['seed']==seed and r['fold']==fold]
                values=vector(rr).mean(0);foldvalues.append(values)
                folds.append(dict(arm=arm,seed=seed,fold=fold,patients=len(rr),**dict(zip(METRICS,values.tolist()))))
            fold_sd.append(dict(arm=arm,seed=seed,**dict(zip(METRICS,np.std(foldvalues,axis=0,ddof=1).tolist()))))
        sd.append(dict(arm=arm,seeds=3,**dict(zip(METRICS,np.std(ss,axis=0,ddof=1).tolist()))))
        coverage.extend(dict(arm=arm,metric=m,unique_patients=len(ids),finite_patients=int(np.isfinite(mean_seed[arm][:,j]).sum()),seeds=3) for j,m in enumerate(METRICS))
        for center in sorted({r['center'] for r in rows}):
            keep=[i for i,pid in enumerate(ids) if next(r for r in rows if r['patient']==pid)['center']==center]
            centers.append(dict(arm=arm,center=center,unique_patients=len(keep),suppressed=len(keep)<5,
                **dict(zip(METRICS,mean_seed[arm][keep].mean(0).tolist() if len(keep)>=5 else [float('nan')]*9))))
    for name,q in [('MAIN_RESULTS',main),('SEED_RESULTS',seeds),('SEED_SD',sd),('FOLD_RESULTS',folds),('FOLD_SD',fold_sd),('SOURCE_RESULTS',centers),('METRIC_COVERAGE',coverage)]:csv_out(stage+'/'+name,q)
    draws=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)));ci=[];family=[]
    definitions=[('absolute',a,{a:1}) for a in ARMS]+[('paired','B1_minus_'+a,{'B1':1,a:-1}) for a in ['B0','B2','B3']]
    for kind,label,weights in definitions:
        x=sum(mean_seed[a]*w for a,w in weights.items());samples=x[draws].mean(1);limits=np.quantile(samples,[.025,.975],axis=0);point=x.mean(0)
        ci.extend(dict(kind=kind,contrast=label,metric=m,mean=float(point[j]),low=float(limits[0,j]),high=float(limits[1,j]),unique_patients=len(ids),seeds=3,draws=10000) for j,m in enumerate(METRICS))
        if kind=='paired':
            lim=np.quantile(samples,[.05/24,1-.05/24],axis=0)
            family.extend(dict(contrast=label,metric=m,mean=float(point[j]),low=float(lim[0,j]),high=float(lim[1,j]),comparisons=12,draws=10000) for j,m in enumerate(METRICS) if m in PRIMARY)
    assert len(ci)==63 and len(family)==12;csv_out(stage+'/INTERVALS',ci);csv_out(stage+'/PRIMARY_FAMILY_INTERVALS',family)
    save(ROOT/f'{stage}_PATIENT_METRICS_PRIVATE.pkl',rows)
    return ids,byseed,mean_seed

def gate(byseed,mean_seed):
    indices=[METRICS.index(m) for m in PRIMARY]
    diff0=(mean_seed['B1']-mean_seed['B0']).mean(0);diff2=(mean_seed['B1']-mean_seed['B2']).mean(0)
    seed0=[(byseed[('B1',s)]-byseed[('B0',s)]).mean(0) for s in SEEDS]
    seed2=[(byseed[('B1',s)]-byseed[('B2',s)]).mean(0) for s in SEEDS]
    performance=bool((diff0[indices]>=-.01).all() and sum((x[indices]>=-.01).all() for x in seed0)>=2)
    a,p,macro,ez=indices
    mechanism=bool(diff2[a]>0 and diff2[p]>0 and diff2[macro]>=-.01 and diff2[ez]>=-.01 and sum(x[a]>0 and x[p]>0 for x in seed2)>=2)
    return dict(status='PASS' if performance and mechanism else 'FAIL',performance_margin_pass=performance,global_reference_comparison_pass=mechanism,continue_final=performance and mechanism,
        B1_minus_B0={m:float(diff0[j]) for j,m in enumerate(METRICS)},B1_minus_B2={m:float(diff2[j]) for j,m in enumerate(METRICS)},
        seeds_meeting_all_four_margins=int(sum((x[indices]>=-.01).all() for x in seed0)),seeds_B1_exceeding_B2_both_ranking_metrics=int(sum(x[a]>0 and x[p]>0 for x in seed2)),
        point_gate_not_formal_noninferiority=True,B3_is_diagnostic_not_selector=True,no_post_assessment_tuning=True,scientific_status='EXPLORATORY_REPEATED_TEST')

def main(stage):
    binding,core,fac,records,rec,banks=setup();score_stage(stage,binding,core,fac,records,rec,banks)
    rows=load_rows(stage,rec,banks);ids,byseed,means=summarize(stage,rows)
    if stage=='internal':write(PUB/'internal/GATE.json',gate(byseed,means))
    write(PUB/stage/'REPORT_STATUS.json',dict(status='ALL4_ARMS_3_SEEDS_REPORTED',stage=stage,patient_fold_seed_rows=len(rows),unique_patients=len(ids),nominal_intervals=63,primary_family_intervals=12))
    print('TPRC_'+stage.upper()+'_ALL4_3SEED_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main(sys.argv[1])
