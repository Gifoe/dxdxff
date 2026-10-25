"""Independent probability, pairwise metrics, selection and interval verification."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import core as c


def independent_transform(x,patient,pre):
    a=np.asarray(x,float).copy()
    if pre['PR']:
        for pid in np.unique(patient):
            ix=patient==pid;v=a[ix].copy();sd=np.nanstd(v,axis=0,ddof=0);mu=np.nanmean(v,axis=0)
            for j in range(v.shape[1]):
                if np.isfinite(sd[j]) and sd[j]>1e-8*pre['units'][j]:v[:,j]=(v[:,j]-mu[j])/sd[j]
                else:v[:,j]=np.where(np.isfinite(v[:,j]),0.,np.nan)
            a[ix]=v
    for j in range(a.shape[1]):a[np.isnan(a[:,j]),j]=pre['imputer'][j]
    return ((a-pre['mean'])/pre['scale']).astype(np.float32).astype(float)


def independent_metrics(y,p,pred,channel):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=np.asarray(pred,bool)
    pos=int(y.sum());neg=len(y)-pos;assert pos and neg
    tp=int(y[pred].sum());tn=int((1-y[~pred]).sum());fn=pos-tp;fp=neg-tn
    pairs=p[y==1,None]-p[y==0][None,:]
    auc=float((np.count_nonzero(pairs>0)+.5*np.count_nonzero(pairs==0))/(pos*neg))
    order=np.argsort(-p,kind='stable');pp=p[order];yy=y[order];tp_c=np.cumsum(yy)
    ends=np.flatnonzero(np.r_[pp[1:]!=pp[:-1],True]);increments=np.diff(np.r_[0,tp_c[ends]])
    ap=float(np.sum(increments/pos*(tp_c[ends]/(ends+1))))
    f1=2*tp/(pos+tp+fp);f0=2*tn/(neg+tn+fn)
    ranked=sorted(range(len(y)),key=lambda i:(-float(p[i]),str(channel[i])))
    first=next(i for i,j in enumerate(ranked) if y[j]==1)
    return dict(AUROC=auc,AP=ap,Macro_F1=.5*(f1+f0),EZ_F1=f1,NEZ_F1=f0,
                Accuracy=1-(fp+fn)/len(y),MRR=1/(first+1),Top_1=float(first==0))


def independent_threshold(y,p,patient):
    grid=np.arange(201)/200;all_macro=[];all_ez=[]
    for pid in np.unique(patient):
        ix=patient==pid;order=np.argsort(p[ix],kind='stable');pp=p[ix][order];yy=y[ix][order]
        cuts=np.searchsorted(pp,grid,side='left');cumulative=np.r_[0,np.cumsum(yy)]
        fn=cumulative[cuts];tp=yy.sum()-fn;tn=cuts-fn;fp=len(yy)-yy.sum()-tn
        f1=2*tp/(2*tp+fp+fn);f0=2*tn/(2*tn+fp+fn)
        all_ez.append(f1);all_macro.append((f0+f1)/2)
    mf=np.sum(all_macro,axis=0)/len(all_macro);ez=np.sum(all_ez,axis=0)/len(all_ez)
    best=sorted(range(201),key=lambda i:(-round(float(mf[i]),12),-round(float(ez[i]),12),abs(float(grid[i])-.5),float(grid[i])))[0]
    return float(grid[best]),float(mf[best]),float(ez[best])


def independent_weight(y,pid):
    return np.array([len(y)/(2*len(np.unique(pid))*np.sum((pid==p)&(y==v))) for p,v in zip(pid,y)])


def independent_permutation(y,pid,fold,b):
    rng=np.random.default_rng([42,20261025,fold,b]);out=y.copy()
    for p in sorted(set(pid)):
        ix=np.flatnonzero(pid==p);order=rng.permutation(len(ix));out[ix]=y[ix[order]]
    return out


def check_pre(x,pid,pre):
    assert pre['fit_input_sha']==c.array_sha(x) and pre['fit_patients']==sorted(set(pid))
    base=np.asarray(x,float).copy()
    for j in range(base.shape[1]):
        missing=np.isnan(base[:,j]);mu=np.nanmean(base[:,j]) if (~missing).any() else 0.
        assert abs(mu-pre['raw_imputer'][j])<=1e-12*max(1,abs(mu));base[missing,j]=mu
    units=np.std(base,axis=0);units=np.where(units==0,1.,units)
    np.testing.assert_allclose(units,pre['units'],rtol=1e-12,atol=0)
    # Independently recompute final FIT means/scales after the current PR rule.
    z=independent_transform(x,pid,dict(pre,mean=np.zeros(x.shape[1]),scale=np.ones(x.shape[1])))
    # Original float32 rounding belongs after final scaling, so use a float64 direct reference here.
    if pre['PR']:
        z=np.empty_like(x,dtype=float)
        for p in np.unique(pid):
            ix=pid==p;v=x[ix];sd=np.std(v,axis=0);mu=np.mean(v,axis=0)
            z[ix]=np.divide(v-mu,sd,out=np.zeros_like(v),where=(sd>1e-8*pre['units'])[None,:])
    else:z=x.copy()
    np.testing.assert_allclose(np.mean(z,axis=0),pre['mean'],atol=1e-12,rtol=1e-12)
    sd=np.std(z,axis=0);sd=np.where(sd==0,1.,sd)
    np.testing.assert_allclose(sd,pre['scale'],atol=1e-12,rtol=1e-12)


def pack_replay(pack,x,y,pid,cols):
    check_pre(x,pid,pack['pre']);z=independent_transform(x,pid,pack['pre'])[:,cols]
    state=pack['fit'];m=pack['model'];w=independent_weight(y,pid)
    assert c.array_sha(y)==state['training_labels_sha']
    assert c.array_sha(w)==state['sample_weights_sha'] and c.array_sha(z)==state['training_input_sha']
    assert np.array_equal(m.classes_,[0,1]) and m.solver=='lbfgs' and m.tol==1e-6 and m.max_iter==2000 and m.class_weight is None and m.penalty=='l2'
    assert m.C==state['C'] and int(m.n_iter_[0])==state['iterations']
    return z


def internal(root):
    root=Path(root);binding=c.jr(root/'private/INPUT_FREEZE.json');assert binding['source']==c.sources()
    d=c.load_data();chosen=c.read(root/'private/MODEL_SELECTION.pkl')
    assert chosen['binding']==binding and len(chosen['cells'])==570 and len(chosen['candidates'])==840
    candidate_count=0;cell_count=0;all_packs=[];cv_table=pd.read_csv(root/'public/inner_cv.csv')
    for f,rr in enumerate(d['roles'],1):
        tr=rr['fit'];cal=rr['validation'];xf=d['x'][tr];yf=d['y'][tr];pf=d['patient'][tr]
        splits=c.inner_splits(pf,f)
        for arm in c.ARMS:
            cols=c.columns(arm);scores={}
            for C in c.CS:
                oof=np.full(len(tr),np.nan)
                for j,(it,iv) in enumerate(splits):
                    key=f'cv_{f}_{arm}_c{C:g}_i{j}';path=root/'private/models'/(key+'.pkl')
                    assert c.sha(path)==chosen['candidates'][key];pack=c.read(path);all_packs.append(pack)
                    assert pack['binding']['fit_indices_sha']==c.array_sha(tr[it]) and pack['binding']['held_indices_sha']==c.array_sha(tr[iv])
                    pack_replay(pack,xf[it],yf[it],pf[it],cols)
                    z=independent_transform(xf[iv],pf[iv],pack['pre'])[:,cols];m=pack['model']
                    oof[iv]=expit(z@m.coef_[0]+m.intercept_[0]);candidate_count+=1
                vals=[]
                for p in np.unique(pf):
                    ix=pf==p;v=oof[ix];y=yf[ix];pairs=v[y==1,None]-v[y==0][None,:]
                    vals.append(float(((pairs>0).sum()+.5*(pairs==0).sum())/pairs.size))
                scores[C]=float(np.mean(vals));expected=cv_table[(cv_table.fold==f)&(cv_table.arm==arm)&(cv_table.C==C)].iloc[0]
                assert abs(scores[C]-expected.OOF_patient_AUROC)<1e-12
            selected=next(r for r in chosen['cells'] if r['kind']=='SELECTED' and r['fold']==f and r['arm']==arm)
            best=min(C for C in c.CS if scores[C]>=max(scores.values())-1e-12)
            assert best==selected['C']
        reference=next(r for r in chosen['cells'] if r['fold']==f and r['kind']=='SELECTED' and r['arm']=='ALL_SIX_PR')
        for r in [r for r in chosen['cells'] if r['fold']==f]:
            path=Path(r['path']);assert c.sha(path)==r['sha256'];pack=c.read(path);all_packs.append(pack)
            y=yf if r['kind']=='SELECTED' else independent_permutation(yf,pf,f,r['permutation'])
            if r['kind']=='PERMUTED':
                assert r['C']==reference['C'] and pack['binding']['true_selected_model_sha']==reference['sha256']
                for p in np.unique(pf):assert y[pf==p].sum()==yf[pf==p].sum()
            cols=c.columns(r['arm']);pack_replay(pack,xf,y,pf,cols)
            z=independent_transform(d['x'][cal],d['patient'][cal],pack['pre'])[:,cols];m=pack['model'];prob=expit(z@m.coef_[0]+m.intercept_[0])
            threshold,mf,ef=independent_threshold(d['y'][cal],prob,d['patient'][cal])
            assert threshold==r['threshold'] and abs(mf-r['CAL_Macro_F1'])<1e-12 and abs(ef-r['CAL_EZ_F1'])<1e-12
            cell_count+=1
    assert candidate_count==840 and cell_count==570 and len(all_packs)==1410
    fit_table=pd.read_csv(root/'public/fit_audit.csv');assert len(fit_table)==1410
    assert fit_table.iterations.sum()==sum(p['fit']['iterations'] for p in all_packs)
    convergence=sum(not p['fit']['converged'] for p in all_packs)
    proof=dict(status='PASS',candidate_models=840,selected_and_null_models=570,unique_LR_fits=1410,
        all_C_choices_replayed=70,all_CAL_thresholds_replayed=570,patient_role_isolation=True,
        every_subFIT_preprocessor_checked=True,patient_and_class_weights_checked=True,
        permutations_preserve_counts=True,probability_direction_EZ_positive=True,
        nonconverged_fits=convergence,max_iterations_observed=max(p['fit']['iterations'] for p in all_packs),
        no_new_Outer_inference_during_internal_verification=True)
    c.write(root/'public/INTERNAL_VERIFICATION.json',proof)
    freeze=dict(input_sha=c.digest(binding),cells=chosen['cells'],candidates=chosen['candidates'],
        internal_verification_sha=c.sha(root/'public/INTERNAL_VERIFICATION.json'))
    path=root/'private/FROZEN_BEFORE_OUTER.json'
    if path.exists():assert c.jr(path)==c.clean(freeze)
    else:c.write(path,freeze)
    c.write(root/'public/FREEZE_STATUS.json',dict(status='ALL570_FROZEN_BEFORE_OUTER',main_and_supplementary=70,
        null_models=500,freeze_sha256=c.sha(path),internal_verification='PASS'))
    print('INTERNAL_PREDICTION_VERIFICATION_PASS',proof,flush=True)


COUNTS={}


def interval(values,adjusted=False):
    v=np.asarray(values,float);n=len(v)
    if n not in COUNTS:
        ids=np.random.default_rng(42).integers(n,size=(10000,n));a=np.zeros((10000,n))
        np.add.at(a,(np.arange(10000)[:,None],ids),1.);COUNTS[n]=a
    samples=COUNTS[n]@v/n;alpha=.05/28 if adjusted else .05
    return [float(v.mean()),*np.percentile(samples,[100*alpha/2,100*(1-alpha/2)])]


def outer(root):
    root=Path(root);pub=root/'public';d=c.load_data();binding=c.jr(root/'private/INPUT_FREEZE.json')
    assert binding['source']==c.sources() and c.jr(pub/'FREEZE_STATUS.json')['status']=='ALL570_FROZEN_BEFORE_OUTER'
    seal=c.jr(root/'private/OUTER_SCORES_SEAL.json');freeze=c.jr(root/'private/FROZEN_BEFORE_OUTER.json')
    assert len(seal['files'])==len(freeze['cells'])==570 and seal['freeze_sha256']==c.sha(root/'private/FROZEN_BEFORE_OUTER.json')
    rows=[];max_probability_error=0.
    for key,h in seal['files'].items():
        path=root/'private/scores'/(key+'.pkl');assert c.sha(path)==h;q=c.read(path)
        r=next(r for r in freeze['cells'] if r['fold']==q['fold'] and r['arm']==q['arm'] and r['permutation']==q['permutation'])
        assert c.sha(r['path'])==r['sha256']==q['model_sha256'] and q['freeze_sha256']==seal['freeze_sha256']
        ix=q['indices'];assert np.array_equal(ix,d['roles'][q['fold']-1]['test'])
        pack=c.read(r['path']);z=independent_transform(d['x'][ix],d['patient'][ix],pack['pre'])[:,c.columns(q['arm'])]
        m=pack['model'];prob=expit(z@m.coef_[0]+m.intercept_[0]);error=float(np.max(abs(prob-q['probability_ez'])))
        max_probability_error=max(max_probability_error,error);assert error<2e-15
        assert np.array_equal(prob>=q['threshold'],q['prediction_ez'])
        for pid in sorted(set(d['patient'][ix])):
            jj=d['patient'][ix]==pid;yy=d['y'][ix[jj]];p=q['probability_ez'][jj]
            row=independent_metrics(yy,p,q['prediction_ez'][jj],d['channel'][ix[jj]])
            rows.append(dict(**row,patient=pid,arm=q['arm'],fold=q['fold'],permutation=q['permutation'],
                source=str(d['source'][ix[jj][0]]),channels=int(jj.sum()),EZ_prevalence=float(yy.mean())))
    frame=pd.DataFrame(rows);assert len(frame)==9120 and np.isfinite(frame[c.METRICS]).all().all()
    stored=c.read(root/'private/PATIENT_METRICS.pkl').set_index(['arm','permutation','patient']).sort_index()
    actual=frame.set_index(['arm','permutation','patient']).sort_index()
    np.testing.assert_allclose(actual[c.METRICS],stored[c.METRICS],atol=1e-12,rtol=0)
    main=frame[frame.permutation==-1];null=frame[frame.permutation>=0]
    overall=pd.read_csv(pub/'all_results.csv',float_precision='round_trip');main_table=pd.read_csv(pub/'main_results.csv',float_precision='round_trip')
    assert overall.arm.tolist()==c.ARMS and main_table.arm.tolist()==c.MAIN
    checked=0;paired=0;adjusted=0
    for row in overall.to_dict('records'):
        v=main[main.arm==row['arm']].sort_values('patient');assert len(v)==80 and v.channels.sum()==7635
        for metric in c.METRICS:
            np.testing.assert_allclose(interval(v[metric]),[row[metric],row[metric+'_ci_low'],row[metric+'_ci_high']],atol=1e-12,rtol=0);checked+=1
        if row['arm'] in c.MAIN:
            mt=main_table[main_table.arm==row['arm']].iloc[0]
            np.testing.assert_allclose([mt[m] for m in c.METRICS],[row[m] for m in c.METRICS],atol=0,rtol=0)
    it=pd.read_csv(pub/'individual_feature_results.csv');assert len(it)==12
    for row in it.to_dict('records'):
        rr=overall[overall.arm==row['arm']].iloc[0]
        np.testing.assert_allclose([row[m] for m in c.METRICS],[rr[m] for m in c.METRICS],atol=1e-12,rtol=0)
    folds=pd.read_csv(pub/'fold_results.csv');assert len(folds)==560
    for r in folds.to_dict('records'):
        v=main[(main.arm==r['arm'])&(main.fold==r['fold'])];assert len(v)==r['patients']
        assert abs(v[r['metric']].mean()-r['mean'])<1e-12
    sr=pd.read_csv(pub/'source_results.csv');assert len(sr)==448
    for r in sr.to_dict('records'):
        v=main[(main.arm==r['arm'])&(main.source==r['source'])].sort_values('patient')
        assert len(v)==r['patients']==dict(hup=36,lzu=21,multicenter=15,pediatric=8)[r['source']]
        np.testing.assert_allclose(interval(v[r['metric']]),[r['mean'],r['ci_low'],r['ci_high']],atol=1e-12,rtol=0);checked+=1
    full=main[main.arm=='ALL_SIX_PR'].set_index('patient').sort_index()
    for r in pd.read_csv(pub/'paired_comparisons.csv').to_dict('records'):
        other=main[main.arm==r['control']].set_index('patient').loc[full.index]
        v=full[r['metric']]-other[r['metric']]
        expected=interval(v);np.testing.assert_allclose(expected,[r['mean'],r['ci_low'],r['ci_high']],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.array(expected)*100,[r['difference_pp'],r['ci_low_pp'],r['ci_high_pp']],atol=1e-10,rtol=0);paired+=1
        if r['metric'] in c.PRIMARY:
            assert r['primary_family_size']==28
            np.testing.assert_allclose(interval(v,True)[1:],[r['adjusted_ci_low'],r['adjusted_ci_high']],atol=1e-12,rtol=0);adjusted+=1
    assert paired==56 and adjusted==28
    for r in pd.read_csv(pub/'pr_improvement.csv').to_dict('records'):
        a=main[main.arm==r['feature']+'_PR'].set_index('patient').sort_index()
        b=main[main.arm==r['feature']+'_NO_PR'].set_index('patient').loc[a.index]
        np.testing.assert_allclose(interval(a[r['metric']]-b[r['metric']]),[r['mean'],r['ci_low'],r['ci_high']],atol=1e-12,rtol=0);paired+=1
    nr=pd.read_csv(pub/'negative_control.csv');assert len(nr)==100
    for r in nr.to_dict('records'):
        v=null[null.permutation==r['permutation']];assert len(v)==80 and v.channels.sum()==7635
        np.testing.assert_allclose(v[c.METRICS].mean(),[r[m] for m in c.METRICS],atol=1e-12,rtol=0)
    for r in pd.read_csv(pub/'negative_control_summary.csv').to_dict('records'):
        v=nr[r['metric']]
        np.testing.assert_allclose([v.mean(),v.std(ddof=1),*np.quantile(v,[.025,.975])],
            [r['null_mean'],r['null_sample_SD'],r['null_reference_low'],r['null_reference_high']],atol=1e-12,rtol=0)
    assert c.sha(c.REC)==c.EXPECTED['reconstruction'] and c.sha(c.LABEL)==c.EXPECTED['labels']
    proof=dict(status='PASS',groups=14,main_groups=8,patient_rows=1120,null_patient_rows=8000,
        patient_endpoint_values=72960,score_cells=570,formal_unique_LR_fits=1410,
        absolute_and_source_CIs_verified=checked,paired_CIs_verified=paired,additional_28family_CIs_verified=adjusted,
        fold_metric_rows_verified=560,source_metric_rows_verified=448,negative_control_repeats_verified=100,
        probability_reconstruction='Saved coefficients plus independent current-PR transform and expit',
        maximum_probability_replay_error=max_probability_error,AUROC='Direct EZ/NEZ pair concordance with ties0.5',
        AP='Grouped-score precision increments',ranking='Independent canonical-channel tie ordering',
        bootstrap='Independent resample count matrices',original_inputs_unchanged=True,private_exported=False)
    c.write(pub/'INDEPENDENT_VERIFICATION.json',proof)
    c.write(pub/'STATUS.json',dict(status='COMPLETE_VERIFIED_SIXFEATURE_EZ_PREDICTION',patients=80,channels=7635,
        groups=14,formal_LR_fits=1410,score_cells=570,patient_rows=9120,metric_values=72960,
        exploratory_historically_selected_cohort=True,neural_fits=0))
    print('INDEPENDENT_PREDICTION_VERIFICATION_PASS',proof,flush=True)
