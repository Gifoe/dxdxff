"""Independentread-only tree/threshold/label-role/selection verification;noOuter."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit_old7 as A
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,confusion_matrix

lock,core,rec,banks,prior=A.setup();out=A.OUT;stage=A.jr(A.TASK/'NONLINEAR_SCREEN_LOCK.json')
freeze=A.jr(out/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json')
old=A.read(out/'CANDIDATE_FEATURES_PRIVATE.pkl');numeric=A.read(out/'NUMERICAL_CONTROL_FEATURES_PRIVATE.pkl')
allx=A.np.concatenate([rec['R14'],old['candidate22'],numeric['summary6']],axis=1)
xs={f:core.patient_z(allx[d['archive_indices']],d['patient']) for f,d in enumerate(banks,1)}
counts=dict(patient=0,center=0);maximum=0.;metrics_checked=0
for path in sorted((out/'nonlinear_probes_private').glob('*/*.pkl')):
    q=A.read(path);fold=q['binding']['fold'];task=q['binding']['task'];d=banks[fold-1];pid=d['patient'];y=(d['y']==0).astype(int)
    tr,ca,he=[q[k] for k in ['train_indices','calibration_indices','screen_indices']]
    groups=[set(pid[ix]) for ix in [tr,ca,he]]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert set.union(*groups)==set(pid[d['train']]) and set.union(*groups).isdisjoint(d['roles']['validation']|d['roles']['test'])
    if task=='center':assert set(d['center'][he]).isdisjoint(set(d['center'][tr])|set(d['center'][ca]))
    for key,value in stage['estimator'].items():
        if key!='class':assert q['model'].get_params()[key]==value
    x=xs[fold][:,q['binding']['columns']];z=q['imputer'].transform(x)
    assert A.np.allclose(q['imputer'].statistics_,x[tr].mean(0),atol=1e-12,rtol=1e-12)
    p=q['model'].predict_proba(z[he])[:,1];pc=q['model'].predict_proba(z[ca])[:,1];frame=q['frame']
    error=float(A.np.max(A.np.abs(p-frame.probability.to_numpy())));maximum=max(maximum,error);assert error<1e-12
    assert A.np.array_equal(frame.y.to_numpy(),y[he]) and A.np.array_equal(frame.patient.to_numpy(),pid[he])
    assert A.EV.threshold(y[ca],pc,pid[ca])==q['threshold_selection']
    expected=A.np.average(y[tr],weights=A.weights(y[tr],pid[tr]));intercept=float(A.np.log(expected/(1-expected)))
    assert abs(float(q['model']._baseline_prediction.ravel()[0])-intercept)<1e-12
    for person,g in frame.groupby('patient'):
        r=next(v for v in q['patients'] if v['patient']==person);tn,fp,fn,tp=confusion_matrix(g.y,g.prediction,labels=[0,1]).ravel()
        assert A.np.array_equal([tn,fp,fn,tp],[r['TN'],r['FP'],r['FN'],r['TP']])
        if g.y.nunique()==2:
            assert abs(r['auroc']-roc_auc_score(g.y,g.probability))<1e-12
            assert abs(r['macro_f1']-f1_score(g.y,g.prediction,average='macro'))<1e-12
        if g.y.sum():assert abs(r['auprc']-average_precision_score(g.y,g.probability))<1e-12
        metrics_checked+=1
    counts[task]+=1
assert counts==dict(patient=705,center=520)
df=A.read(out/'NONLINEAR_PROBE_PATIENTS_PRIVATE.pkl');metrics=['auroc','auprc','macro_f1'];names=freeze['names'];selected=[]
for fold in range(1,6):
    data=df[df.outer_fold==fold]
    def effects(name,task):
        ref=data[(data.task==task)&(data.variant=='FULL7')].set_index('patient').sort_index()
        g=data[(data.task==task)&(data.variant==name)].set_index('patient').loc[ref.index]
        assert g.index.nunique()==len(g)
        return (g[metrics]-ref[metrics]).mean().to_dict(),[(h[metrics]-ref.loc[h.index,metrics]).mean().to_dict() for _,h in g.groupby('split')]
    def gate_center(name):
        _,parts=effects(name,'center');assert len(parts)==4
        return all(A.np.median([v[m] for v in parts])>=median and min(v[m] for v in parts)>=worst
            for m,median,worst in [('auroc',-.002,-.01),('auprc',-.002,-.01),('macro_f1',-.003,-.015)])
    drops=[];adds=[];diagnostic=[]
    for k in range(7):
        v,parts=effects(f'DROP_OLD_{k}','patient')
        support=sum(all(p[m]>=bound for m,bound in [('auroc',-.002),('auprc',-.002),('macro_f1',-.003)]) for p in parts)
        if all(v[m]>=bound for m,bound in [('auroc',-.002),('auprc',-.002),('macro_f1',-.003)]) and support>=2 and gate_center(f'DROP_OLD_{k}'):
            drops.append(((v['macro_f1'],v['auprc'],v['auroc'],-k),k))
    for k in range(14):
        v,parts=effects(f'ADD_NEW_{k}','patient');key=(min(v['auroc'],v['auprc']),v['macro_f1'],v['auroc']+v['auprc'],-k)
        diagnostic.append((key,7+k))
        def pass_gain(p):return (p['auroc']>=.005 or p['auprc']>=.005) and min(p['auroc'],p['auprc'])>=-.002 and p['macro_f1']>=0
        if pass_gain(v) and sum(pass_gain(p) for p in parts)>=2 and gate_center(f'ADD_NEW_{k}'):adds.append((key,7+k))
    expected=dict(deleted_old_group_indices=[k for _,k in sorted(drops,reverse=True)[:2]],added_new_group_indices=[k for _,k in sorted(adds,reverse=True)[:3]],
        new_only_diagnostic_group_indices=[k for _,k in sorted(diagnostic,reverse=True)[:3]])
    actual=freeze['folds'][str(fold)]
    for key,value in expected.items():assert actual[key]==value,(fold,key,actual[key],value)
    selected.append(dict(fold=fold,all_three_lists_independently_reproduced=True))
assert A.sha(out/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json')==A.jr(out/'public/PER_FIT_FEATURE_SELECTION.json')['selection_freeze_sha256']
A.write(out/'public/NONLINEAR_INDEPENDENT_VERIFICATION.json',dict(status='PASS',cells=counts,patient_metric_rows_checked=metrics_checked,
    maximum_probability_replay_error=maximum,all_thresholds_reproduced=True,all_initial_class_weight_priors_reproduced=True,
    FIT_only_patient_and_center_isolation=True,imputers_subFIT_only=True,all_feature_lists_independently_reproduced=selected,
    no_training_or_Outer_evaluation_called=True,verifier_sha256=A.sha(__file__),full_goal_complete=False))
print('ALL1225_NONLINEAR_CELLS_AND_PER_FIT_SELECTION_INDEPENDENT_VERIFY_PASS',flush=True)
