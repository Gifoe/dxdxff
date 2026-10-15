"""Read-only independent cache/score/threshold/role checks; no refit or Outer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit_old7 as A
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,f1_score

lock,core,rec,banks,binding=A.setup();out=A.OUT
assert A.jr(out/'public/CANDIDATE_SCREEN_STATUS.json')['status']=='ALL390_FIT_ONLY_LINEAR_CANDIDATE_PROBES_COMPLETE'
checked=0;max_error=0.
for path in sorted((out/'candidate_probes_private').glob('*.pkl')):
    q=A.read(path);fold=q['binding']['fold'];d=banks[fold-1];cache=A.read(out/'CANDIDATE_FEATURES_PRIVATE.pkl')
    x=A.np.concatenate([cache['original14'],cache['candidate22']],axis=1)[d['archive_indices']]
    x=core.patient_z(x,d['patient'])[:,q['binding']['columns']]
    tr,ca,he=[q[k] for k in ['train_indices','calibration_indices','screen_indices']]
    groups=[set(d['patient'][ix]) for ix in [tr,ca,he]]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert set.union(*groups)==set(d['patient'][d['train']])
    assert set.union(*groups).isdisjoint(d['roles']['validation']|d['roles']['test'])
    z=q['scaler'].transform(q['imputer'].transform(x));p=q['model'].predict_proba(z[he])[:,1]
    pc=q['model'].predict_proba(z[ca])[:,1];frame=q['frame']
    error=float(A.np.max(A.np.abs(p-frame.probability.to_numpy())));max_error=max(max_error,error);assert error<1e-12
    assert A.np.array_equal(frame.y.to_numpy(),(d['y'][he]==0).astype(int))
    assert A.np.array_equal(frame.patient.to_numpy(),d['patient'][he])
    assert A.np.allclose(pc,q['calibration_probability'],atol=1e-12,rtol=0)
    assert A.EV.threshold((d['y'][ca]==0).astype(int),pc,d['patient'][ca])==q['threshold_selection']
    for pid,g in frame.groupby('patient'):
        r=next(r for r in q['patients'] if r['patient']==pid);tn,fp,fn,tp=confusion_matrix(g.y,g.prediction,labels=[0,1]).ravel()
        assert A.np.array_equal([tn,fp,fn,tp],[r['TN'],r['FP'],r['FN'],r['TP']])
        if g.y.nunique()==2:
            assert abs(roc_auc_score(g.y,g.probability)-r['auroc'])<1e-12
            assert abs(f1_score(g.y,g.prediction,average='macro')-r['macro_f1'])<1e-12
        if g.y.sum():assert abs(average_precision_score(g.y,g.probability)-r['auprc'])<1e-12
    if q['binding']['variant']=='FULL7':
        old=A.read(out/f"old7_probes_private/fold{fold}_screen{q['binding']['inner']}_FULL7.pkl")
        l=A.pd.DataFrame(old['patients']).set_index('patient').sort_index();r=A.pd.DataFrame(q['patients']).set_index('patient').loc[l.index]
        assert A.np.allclose(l[['auroc','auprc','macro_f1','f1']],r[['auroc','auprc','macro_f1','f1']],atol=1e-12,rtol=0,equal_nan=True)
    checked+=1
assert checked==390
from raw_store import Store
aggregate=A.load('independent_original_aggregate',A.SOURCE/'raw_delta_encoder_seed42_v1/code/construct_delta.py').aggregate
store=Store(A.BASELINE);registry=A.read(out/'CANDIDATE_REGISTRY_PRIVATE.pkl');cached=A.read(out/'CANDIDATE_FEATURES_PRIVATE.pkl')
patients=seizures=windows=0;summary_error=0.;rms_error=0.
for number,(pid,r) in enumerate(sorted(registry['files'].items())):
    assert A.sha(r['path'])==r['sha256'];q=A.read(r['path']);plan=store.registry['plans'][pid]
    assert A.sha(plan['path'])==q['raw_bank_sha256']==plan['sha256']
    b=A.torch.load(plan['path'],map_location='cpu',weights_only=False);mask=b['mask'][0].numpy();delta=q['delta']
    assert A.hashlib.sha256(mask.tobytes()).hexdigest()==q['mask_sha256'] and int(mask.sum())==q['windows']
    items=[]
    for s in range(len(mask)):
        v=A.np.full((mask.shape[1],11),A.np.nan,A.np.float32)
        for c in range(mask.shape[1]):
            if mask[s,c].any():v[c]=delta[s,c,mask[s,c]].mean(0)
        items.append(v)
    summary,_=aggregate(items);err=float(A.np.max(A.np.abs(summary-q['summary'])));summary_error=max(summary_error,err)
    assert A.np.allclose(summary,q['summary'],atol=1e-6,rtol=2e-6)
    assert A.np.array_equal(cached['candidate22'][q['archive_indices']],q['summary'])
    if number in [0,15,31,47,63,79]:
        raw=b['raw'][0].numpy()[mask][:32].astype(float);expected=1e-8/((raw*raw).mean(1)+1e-8)
        err=float(A.np.max(A.np.abs(expected-q['rms_floor_fraction'][:len(expected)])));rms_error=max(rms_error,err);assert err==0
    patients+=1;seizures+=q['seizures'];windows+=q['windows'];del b,delta
assert (patients,seizures,windows)==(80,256,1471965)
for p in (out/'public').glob('*.csv'):
    assert not set(A.pd.read_csv(p,nrows=0).columns)&{'patient','channel','path','checkpoint','probability','prediction','y','indices'}
A.write(out/'public/INITIAL_AUDITS_INDEPENDENT_VERIFICATION.json',dict(status='PASS',candidate_linear_probes_replayed=checked,
    replay_max_probability_error=max_error,all_thresholds_reproduced=True,original_validation_and_Outer_excluded=True,
    independent_sklearn_patient_metrics=True,original_FULL7_probe_reproduced_in15screens=True,
    candidate_patients_verified=patients,seizures_verified=seizures,valid_windows_verified=windows,
    summary_independent_original_aggregation_max_error=summary_error,rms_formula_six_patient_spotcheck_max_error=rms_error,
    public_privacy_gate=True,verifier_sha256=A.sha(__file__),no_training_or_Outer_evaluation_called=True,
    complete_M8_feature_discovery_goal=False))
print('INITIAL_AUDITS_390_PROBES_AND_80_PATIENT_CACHE_INDEPENDENT_VERIFY_PASS',flush=True)
