"""Fixed trees,patient/center holdouts,orientation audits and FIT-only selection."""
from audit_old7 import *
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

stage=jr(TASK/'NONLINEAR_SCREEN_LOCK.json');lock,core,rec,banks,prior=setup()
assert jr(OUT/'public/NUMERICAL_CONTROL_EXTRACTION_STATUS.json')['status']=='STABLE_LOG3_NUMERICAL_CONTROLS_COMPLETE'
first=read(OUT/'CANDIDATE_FEATURES_PRIVATE.pkl');numeric=read(OUT/'NUMERICAL_CONTROL_FEATURES_PRIVATE.pkl')
names=lock['old_feature_names']+first['names']+numeric['names'];assert len(names)==21
rawx=np.concatenate([rec['R14'],first['candidate22'],numeric['summary6']],axis=1);assert rawx.shape==(7635,42) and np.isfinite(rawx).all()
binding=dict(source={n:sha(TASK/n) for n in ['NONLINEAR_SCREEN_LOCK.json','screen_nonlinear.py','audit_old7.py','AUDIT_PROTOCOL.json']},
    prior=prior,candidate_sha256=sha(OUT/'CANDIDATE_FEATURES_PRIVATE.pkl'),numeric_sha256=sha(OUT/'NUMERICAL_CONTROL_FEATURES_PRIVATE.pkl'))
fp=OUT/'NONLINEAR_SCREEN_INPUT_FREEZE_PRIVATE.json'
if fp.exists():assert jr(fp)==binding
else:write(fp,binding)
old=list(range(14));columns=lambda groups:[2*g+j for g in groups for j in [0,1]]
variants=[('FULL7',old)]+[(f'ONLY_OLD_{k}',columns([k])) for k in range(7)]+[(f'DROP_OLD_{k}',columns([g for g in range(7) if g!=k])) for k in range(7)]
variants += [(f'ONLY_NEW_{k}',columns([7+k])) for k in range(14)]+[(f'ADD_NEW_{k}',old+columns([7+k])) for k in range(14)]
variants += [('ALL21',list(range(42))),('ADD_SHAPE5',old+columns([8,9,10,16,17])),('ADD_SPECTRAL5',old+columns([11,12,13,14,15])),
    ('NUMERIC_REPLACE_POWER3',columns([18,19,20,3,4,5,6]))]
assert len(variants)==47
center_variants=[v for v in variants if v[0]=='FULL7' or v[0].startswith(('DROP_OLD','ADD_NEW')) or v[0] in ['ALL21','ADD_SHAPE5','ADD_SPECTRAL5','NUMERIC_REPLACE_POWER3']]
assert len(center_variants)==26
metrics=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1'];rows=[];individual=[];orient=[];isolation=[]


def patient_auc_matrix(xx,y,pid,indices):
    result=[];centers=[]
    for p in sorted(set(pid[indices])):
        ii=indices[pid[indices]==p];yy=y[ii];pos=int(yy.sum());neg=len(yy)-pos
        if not pos or not neg:result.append(np.full(xx.shape[1],np.nan));centers.append(p);continue
        rank=rankdata(xx[ii],axis=0)
        result.append((rank[yy==1].sum(0)-pos*(pos+1)/2)/(pos*neg));centers.append(p)
    return np.asarray(result),centers


for fold,d in enumerate(banks,1):
    fit=d['train'];seed=42+1009*fold;pid=d['patient'];y=(d['y']==0).astype(int);x=core.patient_z(rawx[d['archive_indices']],pid)
    splits=[]
    for inner,(it,iv) in enumerate(GroupKFold(n_splits=3,shuffle=True,random_state=seed).split(x[fit],y[fit],groups=pid[fit])):
        splits.append(('patient',inner+1,fit[it],fit[iv],seed+43*(inner+1),'multiple'))
    centers=sorted(set(d['center'][fit]));assert len(centers)==4
    for k,center in enumerate(centers,1):
        held=fit[d['center'][fit]==center];remaining=fit[d['center'][fit]!=center]
        splits.append(('center',k,remaining,held,seed+700+53*k,str(center)))
    for task,number,remain,held,splitseed,center in splits:
        order=np.random.default_rng(splitseed).permutation(np.asarray(sorted(set(pid[remain]))));ncal=max(1,int(np.ceil(.2*len(order))))
        cal=remain[np.isin(pid[remain],order[:ncal])];tr=remain[~np.isin(pid[remain],order[:ncal])]
        groups=[set(pid[q]) for q in [tr,cal,held]]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
        assert set.union(*groups)==set(pid[fit]) and set.union(*groups).isdisjoint(d['roles']['validation']|d['roles']['test'])
        if task=='center':assert set(d['center'][tr])|set(d['center'][cal])==set(centers)-{center}
        isolation.append(dict(outer_fold=fold,task=task,split=number,held_center=center,subfit_patients=len(groups[0]),
            calibration_patients=len(groups[1]),screen_patients=len(groups[2]),validation_Outer_excluded=True,heldout_did_not_choose_threshold=True))
        ta,_=patient_auc_matrix(x,y,pid,tr);ha,hids=patient_auc_matrix(x,y,pid,held);direction=np.where(np.nanmean(ta,axis=0)>=.5,1,-1)
        hcenter=np.asarray([d['center'][np.flatnonzero(pid==p)[0]] for p in hids]);oriented=np.where(direction[None]>0,ha,1-ha)
        for group in ['ALL']+sorted(set(hcenter)):
            take=np.ones(len(hids),bool) if group=='ALL' else hcenter==group
            for j in range(42):
                a=oriented[take,j];a=a[np.isfinite(a)]
                orient.append(dict(outer_fold=fold,task=task,split=number,held_center=center,summary_center=group,feature=names[j//2],coordinate='mean' if j%2==0 else 'sd',
                    training_orientation=int(direction[j]),training_patient_mean_raw_auc=float(np.nanmean(ta[:,j])),valid_held_patients=len(a),
                    held_mean_oriented_auc=float(a.mean()) if len(a) else np.nan,held_fraction_auc_gt_half=float((a>.5).mean()) if len(a) else np.nan))
        for name,cols in variants if task=='patient' else center_variants:
            path=OUT/f'nonlinear_probes_private/{task}/fold{fold}_split{number}_{name}.pkl'
            cb=dict(binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(),fold=fold,task=task,split=number,variant=name,columns=cols)
            if path.exists():q=read(path);assert q['binding']==cb
            else:
                xx=x[:,cols];imp=SimpleImputer().fit(xx[tr]);z=imp.transform(xx)
                model=HistGradientBoostingClassifier(random_state=seed,**{k:v for k,v in stage['estimator'].items() if k!='class'})
                model.fit(z[tr],y[tr],sample_weight=weights(y[tr],pid[tr]));assert model.n_iter_==100
                pc=model.predict_proba(z[cal])[:,1];sel=EV.threshold(y[cal],pc,pid[cal]);p=model.predict_proba(z[held])[:,1]
                frame=pd.DataFrame(dict(model=name,version=task,fold=fold,patient=pid[held],channel=d['channel'][held],center=d['center'][held],
                    y=y[held],probability=p,prediction=p>=sel['threshold']))
                q=dict(binding=cb,model=model,imputer=imp,frame=frame,patients=EV.patient_rows(frame),threshold_selection=sel,
                    calibration_probability=pc,train_indices=tr,calibration_indices=cal,screen_indices=held);save(path,q)
            g=pd.DataFrame(q['patients']);individual += [dict(r,outer_fold=fold,task=task,split=number,held_center=center,variant=name) for r in q['patients']]
            rows.append(dict(outer_fold=fold,task=task,split=number,held_center=center,variant=name,patients=len(g),channels=int(g.n.sum()),**{m:EV.nanmean(g[m]) for m in metrics}))
            write(OUT/'public/NONLINEAR_SCREEN_STATUS.json',dict(status='RUNNING',completed_cells=len(rows),total_cells=1225,outer_fold=fold,task=task,split=number))
            print('NONLINEAR_FIT_SCREEN_COMPLETE',fold,task,number,name,len(rows),'of1225',flush=True)
assert len(rows)==1225
df=pd.DataFrame(individual);save(OUT/'NONLINEAR_PROBE_PATIENTS_PRIVATE.pkl',df)
pd.DataFrame(rows).to_csv(OUT/'public/NONLINEAR_INNER_AND_CENTER_METRICS.csv',index=False)
pd.DataFrame(orient).to_csv(OUT/'public/FEATURE_DIRECTION_STABILITY.csv',index=False)
pd.DataFrame(isolation).to_csv(OUT/'public/NONLINEAR_PATIENT_CENTER_ISOLATION.csv',index=False)
summaries=[];differences=[];perfold={};selection=[];intervals=[]
for fold in range(1,6):
    dd=df[df.outer_fold==fold];perfold[fold]={}
    for task,gg in dd.groupby('task'):
        baseline=gg[gg.variant=='FULL7'].set_index('patient').sort_index();assert len(baseline)==baseline.index.nunique()
        draws=np.random.default_rng(42).integers(0,len(baseline),size=(10000,len(baseline)))
        for name,g in gg.groupby('variant'):
            g=g.set_index('patient').loc[baseline.index];assert np.array_equal(g.n,baseline.n)
            summaries.append(dict(outer_fold=fold,task=task,variant=name,patients=len(g),**{m:EV.nanmean(g[m]) for m in metrics}))
            delta={m:EV.nanmean(g[m]-baseline[m]) for m in metrics}
            if task=='patient':perfold[fold][name]=dict(delta=delta,support={})
            for number,h in g.groupby('split'):
                b=baseline.loc[h.index];v={m:EV.nanmean(h[m]-b[m]) for m in metrics}
                differences.append(dict(outer_fold=fold,task=task,split=number,variant=name,**{f'delta_{m}':v[m] for m in metrics}))
                if task=='patient':perfold[fold][name]['support'][int(number)]=v
                else:perfold[fold].setdefault(name,dict(delta={},support={})).setdefault('center',[]).append(v)
            if name!='FULL7':
                for m in ['auroc','auprc','macro_f1','f1']:
                    a=(g[m]-baseline[m]).to_numpy();dist=np.nanmean(a[draws],axis=1)
                    intervals.append(dict(outer_fold=fold,task=task,variant=name,metric=m,delta=EV.nanmean(a),
                        ci_low=float(np.nanpercentile(dist,2.5)),ci_high=float(np.nanpercentile(dist,97.5)),draws=10000,selection_conditioned=True))
    # Patient task iterates before/after center lexicographically; combine from raw differences to avoid state overwrite.
    changes=pd.DataFrame([r for r in differences if r['outer_fold']==fold])
    def record(name):
        p=changes[(changes.task=='patient')&(changes.variant==name)];c=changes[(changes.task=='center')&(changes.variant==name)]
        full=next(r for r in summaries if r['outer_fold']==fold and r['task']=='patient' and r['variant']=='FULL7')
        own=next(r for r in summaries if r['outer_fold']==fold and r['task']=='patient' and r['variant']==name)
        delta={m:own[m]-full[m] for m in metrics}
        center_ok=len(c)==4 and all(np.isfinite(c[f'delta_{m}']).all() for m in ['auroc','auprc','macro_f1'])
        for m,median_bound,worst_bound in [('auroc',-.002,-.01),('auprc',-.002,-.01),('macro_f1',-.003,-.015)]:
            center_ok=center_ok and float(c[f'delta_{m}'].median())>=median_bound and float(c[f'delta_{m}'].min())>=worst_bound
        return p,delta,bool(center_ok)
    drops=[];adds=[];newrank=[]
    for k in range(7):
        p,v,center_ok=record(f'DROP_OLD_{k}')
        support=int(((p.delta_auroc>=-.002)&(p.delta_auprc>=-.002)&(p.delta_macro_f1>=-.003)).sum())
        eligible=v['auroc']>=-.002 and v['auprc']>=-.002 and v['macro_f1']>=-.003 and support>=2 and center_ok
        selection.append(dict(outer_fold=fold,kind='delete_old',feature=names[k],eligible=eligible,supporting_screens=support,center_gate=center_ok,**{f'delta_{m}':v[m] for m in metrics}))
        if eligible:drops.append(((v['macro_f1'],v['auprc'],v['auroc'],-k),k))
    for k in range(14):
        p,v,center_ok=record(f'ADD_NEW_{k}')
        support=int((((p.delta_auroc>=.005)|(p.delta_auprc>=.005))&(p.delta_auroc>=-.002)&(p.delta_auprc>=-.002)&(p.delta_macro_f1>=0)).sum())
        eligible=(v['auroc']>=.005 or v['auprc']>=.005) and min(v['auroc'],v['auprc'])>=-.002 and v['macro_f1']>=0 and support>=2 and center_ok
        key=(min(v['auroc'],v['auprc']),v['macro_f1'],v['auroc']+v['auprc'],-k)
        newrank.append((key,k))
        if eligible:adds.append((key,k))
        selection.append(dict(outer_fold=fold,kind='add_new',feature=names[7+k],eligible=eligible,supporting_screens=support,center_gate=center_ok,**{f'delta_{m}':v[m] for m in metrics}))
    dropped=[k for _,k in sorted(drops,reverse=True)[:2]];added=[7+k for _,k in sorted(adds,reverse=True)[:3]]
    diag=[7+k for _,k in sorted(newrank,reverse=True)[:3]]
    perfold[fold]=dict(deleted_old_group_indices=dropped,retained_old_group_indices=[k for k in range(7) if k not in dropped],
        added_new_group_indices=added,new_only_diagnostic_group_indices=diag,
        deleted_names=[names[k] for k in dropped],added_names=[names[k] for k in added],new_only_diagnostic_names=[names[k] for k in diag])
pd.DataFrame(summaries).to_csv(OUT/'public/NONLINEAR_FIT_CROSSFIT_SUMMARY.csv',index=False)
pd.DataFrame(differences).to_csv(OUT/'public/NONLINEAR_PAIRED_SCREEN_DIFFERENCES.csv',index=False)
pd.DataFrame(intervals).to_csv(OUT/'public/NONLINEAR_PAIRED_FIT_INTERVALS.csv',index=False)
pd.DataFrame(selection).to_csv(OUT/'public/NONLINEAR_FEATURE_ELIGIBILITY.csv',index=False)
freeze=dict(binding=binding,names=names,folds=perfold,selected_only_inside_each_original_FIT=True,no_new_M8_validation_read=True)
save(OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.pkl',freeze)
write(OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json',freeze)
write(OUT/'public/PER_FIT_FEATURE_SELECTION.json',dict(status='PROSPECTIVE_FEATURE_LISTS_FROZEN',folds=perfold,
    selection_freeze_sha256=sha(OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json'),original_validation_Outer_excluded=True,full_goal_complete=False))
write(OUT/'public/NONLINEAR_SCREEN_STATUS.json',dict(status='ALL1225_NONLINEAR_PATIENT_CENTER_SCREENS_COMPLETE',patient_cells=705,center_cells=520,
    orientations_fit_subtrain_only=True,feature_selection_freeze_sha256=sha(OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json'),source=binding['source'],
    M8_integration_and_full_goal_pending=True,full_goal_complete=False))
print('ALL1225_NONLINEAR_PATIENT_CENTER_SCREENS_COMPLETE',flush=True)
