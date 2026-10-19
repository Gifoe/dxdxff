"""All-cell frozen scores before metrics; no Outer entry point."""
from common import *

def score(binding,core,fac,records,rec,banks):
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json');inputs=store();files={};coverage=[]
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];d,scale=data(fold,core,rec,banks,inputs,binding)
        ix=d['assessment'];archive=d['archive_indices'][ix];key=f'{arm}_seed{seed}_fold{fold}';path=ROOT/'scores_private'/(key+'.pkl')
        cb=dict(input=digest(binding),freeze=fh,checkpoint=r['checkpoint_sha256'],indices=legacy.array_sha(archive),preprocessor=r['preprocessor'])
        if path.exists():q=read(path);assert q['binding']==cb
        elif r['reused']:
            src=prior.ROOT/'scores_private/internal'/f'B0_seed{seed}_fold{fold}.pkl'
            seal=jr(prior.ROOT/'internal_SCORE_SEAL_PRIVATE.json');assert sha(src)==seal['files'][f'B0_seed{seed}_fold{fold}']
            oldq=read(src);assert np.array_equal(oldq['indices'],archive) and oldq['threshold']==r['threshold']
            assert oldq['binding']['checkpoint']==r['checkpoint_sha256'] and oldq['binding']['preprocessor']==r['preprocessor']['summary']
            q=dict(binding=cb,indices=archive,score_nez=oldq['score_nez'],threshold=r['threshold'],audit=oldq['audit'],source_sha256=sha(src));save(path,q)
        else:
            m=make(core,fac,records,fold,arm,seed);state=torch.load(r['checkpoint'],weights_only=False,map_location='cpu');m.load_state_dict(state['model']);m.eval()
            out=np.empty(len(ix),np.float32);audit={};before=state_hash(m.state_dict())
            for number,pid in enumerate(sorted(set(d['patient'][ix])),1):
                j=np.flatnonzero(d['patient'][ix]==pid);ii=ix[j]
                active=d['x'] if m.feature_pr else d['x_no_pr']
                pb=dict(**cb,patient_indices=legacy.array_sha(archive[j]),features=legacy.array_sha(active[ii]))
                cache=ROOT/'patient_scores_private'/key/(hashlib.sha256(str(pid).encode()).hexdigest()+'.pkl')
                if cache.exists():z=read(cache);assert z['binding']==pb
                else:
                    with torch.no_grad():logit,info=forward(m,d,ii,inputs,scale,True)
                    z=dict(binding=pb,score_nez=probability(logit.cpu().numpy()),audit=info);save(cache,z)
                out[j]=z['score_nez']
                for k,v in z['audit'].items():audit[k]=audit.get(k,0)+v
                write(PUB/'PROGRESS.json',dict(status='SEALED_ASSESS_PREDICTION',arm=arm,seed=seed,fold=fold,patient_number=number,complete_cells=len(files)))
            assert state_hash(m.state_dict())==before
            q=dict(binding=cb,indices=archive,score_nez=out,threshold=r['threshold'],audit=audit);save(path,q)
            del m;gc.collect();torch.cuda.empty_cache()
        assert np.array_equal(q['indices'],archive) and np.isfinite(q['score_nez']).all() and q['threshold']==r['threshold']
        files[key]=sha(path)
        coverage.append(dict(arm=arm,seed=seed,fold=fold,patients=len(d['roles']['assessment']),channels=len(ix),reused=r['reused'],
                             **{k:q['audit'].get(k,float('nan')) for k in ['valid_windows','raw_summary_coordinates','raw_summary_sum_abs','raw_summary_sum_square']}))
        print('BPR_ASSESS_SCORE_SEALED',arm,seed,fold,len(files),90,flush=True)
    assert len(files)==90
    freeze(ROOT/'SCORE_SEAL_PRIVATE.json',dict(freeze=fh,files=files,all90_before_metrics=True))
    csv_out('COVERAGE',coverage);write(PUB/'PREDICTION_STATUS.json',dict(status='ALL90_SEALED_BEFORE_METRICS',freeze_sha256=fh,seal_sha256=sha(ROOT/'SCORE_SEAL_PRIVATE.json')))

def rows_from_scores(rec,banks):
    fr=frozen();seal=jr(ROOT/'SCORE_SEAL_PRIVATE.json');assert seal['freeze']==sha(ROOT/'FROZEN_BEFORE_ASSESS_PRIVATE.json') and seal['all90_before_metrics'] and len(seal['files'])==90
    rows=[];meta=rec['meta']
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];key=f'{arm}_seed{seed}_fold{fold}';path=ROOT/'scores_private'/(key+'.pkl');assert sha(path)==seal['files'][key]
        q=read(path);ii=q['indices'];p=1-q['score_nez'].astype(float);v=q['score_nez']<r['threshold']
        b=banks[fold-1];allowed=prior.roles_for(b,fold,'internal')['assessment'];assert set(meta['patient'][ii])==allowed
        fit=np.flatnonzero(np.isin(b['patient'],list(b['roles']['fit'])));labelmap={int(i):int(1-y) for i,y in zip(b['archive_indices'][fit],b['y'][fit])}
        yy=np.array([labelmap[int(i)] for i in ii])
        for pid in sorted(allowed):
            j=meta['patient'][ii]==pid;y=yy[j]
            values=dict(**EV.classification(y,p[j],v[j]),**EV.ranking(y,p[j],meta['channel'][ii][j]))
            rows.append(dict(arm=arm,seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**{m:float(values[m]) for m in METRICS}))
    assert len(rows)==720 and np.isfinite(vector(rows)).all();return rows

def matrices(rows):
    ids=sorted({r['patient'] for r in rows});byseed={}
    for arm in ARMS:
        for seed in SEEDS:
            byseed[(arm,seed)]=np.array([vector([r for r in rows if (r['arm'],r['seed'],r['patient'])==(arm,seed,p)]).mean(0) for p in ids])
    return ids,byseed,{a:np.mean([byseed[(a,s)] for s in SEEDS],axis=0) for a in ARMS}

def summarize(rows):
    ids,byseed,means=matrices(rows);main=[];seeds=[];sd=[];folds=[];foldsd=[];centers=[];coverage=[]
    for arm in ARMS:
        main.append(dict(arm=arm,unique_patients=len(ids),seeds=3,patient_fold_seed_rows=120,**dict(zip(METRICS,means[arm].mean(0).tolist()))))
        values=[]
        for seed in SEEDS:
            v=byseed[(arm,seed)].mean(0);values.append(v);seeds.append(dict(arm=arm,seed=seed,unique_patients=len(ids),**dict(zip(METRICS,v.tolist()))));ff=[]
            for fold in range(1,6):
                rr=[r for r in rows if (r['arm'],r['seed'],r['fold'])==(arm,seed,fold)];v=vector(rr).mean(0);ff.append(v)
                folds.append(dict(arm=arm,seed=seed,fold=fold,patients=len(rr),**dict(zip(METRICS,v.tolist()))))
            foldsd.append(dict(arm=arm,seed=seed,**dict(zip(METRICS,np.std(ff,axis=0,ddof=1).tolist()))))
        sd.append(dict(arm=arm,seeds=3,**dict(zip(METRICS,np.std(values,axis=0,ddof=1).tolist()))))
        coverage.extend(dict(arm=arm,metric=m,unique_patients=len(ids),finite_patients=int(np.isfinite(means[arm][:,j]).sum())) for j,m in enumerate(METRICS))
        for center in sorted({r['center'] for r in rows}):
            keep=[i for i,p in enumerate(ids) if next(r for r in rows if r['patient']==p)['center']==center]
            centers.append(dict(arm=arm,center=center,unique_patients=len(keep),suppressed=len(keep)<5,
                                **dict(zip(METRICS,means[arm][keep].mean(0).tolist() if len(keep)>=5 else [float('nan')]*9))))
    for n,rr in [('MAIN_RESULTS',main),('SEED_RESULTS',seeds),('SEED_SD',sd),('FOLD_RESULTS',folds),('FOLD_SD',foldsd),('SOURCE_RESULTS',centers),('METRIC_COVERAGE',coverage)]:csv_out(n,rr)
    draws=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)));ci=[];family=[];evidence=[]
    definitions=[('absolute',a,{a:1}) for a in ARMS]+[('paired',n,w) for n,w in CONTRASTS.items()]
    for kind,name,weights in definitions:
        x=sum(means[a]*w for a,w in weights.items());sample=x[draws].mean(1);lim=np.quantile(sample,[.025,.975],axis=0);point=x.mean(0)
        ci.extend(dict(kind=kind,contrast=name,metric=m,mean=float(point[j]),low=float(lim[0,j]),high=float(lim[1,j]),unique_patients=len(ids),seeds=3,draws=10000) for j,m in enumerate(METRICS))
        if name in PRIMARY_CONTRASTS:
            ll=np.quantile(sample,[.05/16,1-.05/16],axis=0)
            family.extend(dict(contrast=name,metric=m,mean=float(point[j]),low=float(ll[0,j]),high=float(ll[1,j]),comparisons=8,draws=10000) for j,m in enumerate(METRICS) if m in PRIMARY_METRICS)
            deltas=[sum(byseed[(a,s)].mean(0)*w for a,w in weights.items()) for s in SEEDS];indices=[METRICS.index(m) for m in PRIMARY_METRICS]
            count=sum(all(v[j]>0 for j in indices) for v in deltas)
            evidence.append(dict(contrast=name,mean_AUROC_and_Macro_both_positive=bool(all(point[j]>0 for j in indices)),seeds_both_positive=count,
                                 stable_positive_point_evidence=bool(count>=2 and all(point[j]>0 for j in indices)),
                                 both_nominal_lowers_positive=bool(all(lim[0,j]>0 for j in indices)),both_family_lowers_positive=bool(all(ll[0,j]>0 for j in indices))))
    assert len(ci)==108 and len(family)==8;csv_out('INTERVALS',ci);csv_out('PRIMARY_FAMILY_INTERVALS',family);csv_out('PR_COMPONENT_EVIDENCE',evidence)
    save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',rows)
    idx=[METRICS.index(m) for m in ['auroc','auprc','macro_f1','f1']]
    diff=(means['FULL']-means['B0']).mean(0);seed_diff=[(byseed[('FULL',s)]-byseed[('B0',s)]).mean(0) for s in SEEDS]
    performance=bool(diff[idx[0]]>0 and all(diff[j]>=-.01 for j in idx[1:]) and sum(x[idx[0]]>0 and all(x[j]>=-.01 for j in idx[1:]) for x in seed_diff)>=2)
    pr_pass=all(next(r for r in evidence if r['contrast']==n)['stable_positive_point_evidence'] for n in ['FEATURE_PR_WITH_RAW','RAW_PR_WITH_FEATURE'])
    context=next(r for r in evidence if r['contrast']=='NEW_PATIENT_CONTEXT')['stable_positive_point_evidence']
    write(PUB/'QUALIFICATION_GATE.json',dict(status='PASS' if performance and pr_pass and context else 'FAIL',performance_margin_pass=performance,
                                            both_separate_PR_point_rules_pass=pr_pass,capacity_context_point_rule_pass=context,
                                            qualified_for_separate_core_module_study=bool(performance and pr_pass and context),new_Outer_authorized=False,
                                            point_rule_not_formal_inference=True,active_goal_achieved=False))
    write(PUB/'REPORT_STATUS.json',dict(status='ALL6_ARMS_3SEEDS_REPORTED',patient_fold_seed_rows=len(rows),unique_patients=len(ids),nominal_intervals=108,primary_family_intervals=8,
                                       no_new_Outer=True,active_goal_achieved=False))

def main():
    binding,core,fac,records,rec,banks=setup();score(binding,core,fac,records,rec,banks);summarize(rows_from_scores(rec,banks))
    print('BPR_INTERNAL_ALL6_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
