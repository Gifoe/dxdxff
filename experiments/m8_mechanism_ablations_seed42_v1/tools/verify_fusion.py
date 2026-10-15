"""Independent validation-only fusion reconstruction; no training/new inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runner as R

lock,c,core,fac,records,rec,banks,refs,binding=R.setup()
assert R.jr(R.OUT/'public/FINAL_STATUS.json')['status']=='M8_MECHANISM_5FOLD_OUTER_COMPLETE'
fr=R.jr(R.OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json');rawfr=R.jr(R.RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json')
grid=R.pd.read_csv(R.OUT/'public/FUSION_VALIDATION_SEARCH.csv')
assert len(grid)==105
protected=[R.OUT/p for p in ['FROZEN_BEFORE_OUTER_PRIVATE.json','OUTER_PREDICTIONS_SEAL_PRIVATE.json','PREDICTIONS_PRIVATE.pkl']]
before={str(p):R.sha(p) for p in protected};checked=0
for fold,d in enumerate(banks,1):
    p1=R.read(R.BASELINE/f'scores/D1_fold{fold}_PRIVATE.pkl')['score'].astype(float)
    r=next(q for q in rawfr['selections'] if q['fold']==fold)
    assert R.sha(r['checkpoint'])==r['checkpoint_sha256']
    cp=R.torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
    p2=R.torch.sigmoid(R.torch.from_numpy(cp['validation_logit'])).numpy().astype(float)
    choices=[]
    for w in R.np.round(R.np.arange(0,1.00001,.05),2):
        v=core.select_threshold(d['y'][d['val']],(1-float(w))*p1+float(w)*p2,d['patient'][d['val']])
        row=grid[(grid.fold==fold)&R.np.isclose(grid.raw_weight,w,atol=1e-12,rtol=0)]
        assert len(row)==1
        for k in ['threshold','macro_f1','ez_f1']:assert abs(float(row[k].iloc[0])-v[k])<1e-12
        choices.append((v['macro_f1'],v['ez_f1'],-float(w),float(w),v['threshold']));checked+=1
    chosen=max(choices,key=lambda q:q[:3]);s=next(q for q in fr['fusions'] if q['fold']==fold)
    assert (s['raw_weight'],s['threshold'])==(chosen[3],chosen[4])
    ix,p1=R.reference_score('D1_PR',fold,rec['meta']);ix2,p2=R.reference_score('M8_RAW_ONLY',fold,rec['meta'])
    assert R.np.array_equal(ix,ix2)
    expected=(1-s['raw_weight'])*p1.astype(float)+s['raw_weight']*p2.astype(float)
    q=R.read(R.OUT/f'outer_scores_private/A6_LATE_FUSION_fold{fold}.pkl')
    assert R.np.array_equal(q['indices'],ix) and R.np.array_equal(q['score_nez'],expected)
for p,h in before.items():assert R.sha(p)==h
R.write(R.OUT/'public/INDEPENDENT_FUSION_AUDIT.json',dict(status='PASS',validation_candidates_reconstructed=checked,
    all5_weights_and_thresholds_reproduced=True,all5_Outer_scores_exact_convex_frozen_components=True,
    no_training_or_new_inference=True,models_predictions_and_freeze_unchanged=True,verifier_sha256=R.sha(__file__)))
R.write(R.OUT/'public/PUBLIC_SHA256.json',{p.name:R.sha(p) for p in (R.OUT/'public').iterdir() if p.name!='PUBLIC_SHA256.json'})
print('INDEPENDENT_105_VALIDATION_CANDIDATES_AND_5_SEALED_FUSIONS_PASS',flush=True)
