"""Independent sklearn/confusion and multiplicity-bootstrap checks; no fitting."""
from support import *
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,confusion_matrix

def run(a):
    lock,c,core,fac,trainer,records,rec,banks,binding=setup(a);pub=a.runtime/'results'
    fr=jread(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jread(a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    assert fr['binding']==binding and seal['freeze_sha256']==sha(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert references(a,core,banks)==fr['references']
    feature=read(a.runtime/'FEATURES_PRIVATE.pkl');assert np.array_equal(feature['DELTA14'],rec['R14'])
    rows=pd.DataFrame(read(a.runtime/'OUTER_METRICS_PRIVATE.pkl'));meta=rec['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    checked=0
    for r in fr['selections']+fr['references']:
        arm,fold=r['arm'],r['fold'];pp=a.runtime/f'outer_scores_private/{arm}_fold{fold}.pkl'
        assert sha(pp)==seal['files'][f'{arm}_{fold}'] and sha(Path(r['checkpoint']))==r['checkpoint_sha256']
        q=read(pp);ix=q['indices'];ss=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();tau=r['threshold']
        for pid in sorted(set(meta['patient'][ix])):
            k=np.flatnonzero(meta['patient'][ix]==pid);k=k[np.argsort(meta['channel'][ix][k].astype(str),kind='stable')]
            truth=y[ix[k]]==0;pred=ss[k]<tau;tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[False,True]).ravel()
            f=f1_score(truth,pred,labels=[False,True],average=None,zero_division=0)
            rr=rows[(rows.arm==arm)&(rows.fold==fold)&(rows.patient==str(pid))].iloc[0]
            assert [tp,fp,tn,fn]==[rr.TP,rr.FP,rr.TN,rr.FN]
            expected=dict(ez_auroc=roc_auc_score(truth,1-ss[k].astype(float)),ez_auprc=average_precision_score(truth,1-ss[k].astype(float)),macro_f1=f.mean(),ez_f1=f[1],sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),predicted_ez_fraction=pred.mean())
            rank=np.argsort(ss[k],kind='stable');pos=np.flatnonzero(truth[rank]);expected.update(ez_mrr=1/(pos[0]+1),top1_is_ez=float(truth[rank[0]]))
            for name,v in expected.items():assert np.isclose(rr[name],v,rtol=0,atol=1e-12)
            checked+=1
    assert checked==320
    for scope in ['DEVELOPMENT','OUTER']:
        df=pd.DataFrame(read(a.runtime/f'{scope}_METRICS_PRIVATE.pkl'))
        actual=pd.read_csv(pub/f'{scope}_SUMMARY.csv').set_index('arm');expect=df.groupby('arm')[METRICS].mean()
        assert np.allclose(actual.loc[expect.index,METRICS],expect,atol=1e-12,rtol=0)
        for key in ['fold','center']:
            actual=pd.read_csv(pub/f'{scope}_BY_{key.upper()}.csv').set_index(['arm',key]);expect=df.groupby(['arm',key])[METRICS].mean()
            assert np.allclose(actual.loc[expect.index,METRICS],expect,rtol=0,atol=1e-12)
    boot=pd.read_csv(pub/'PAIRED_BOOTSTRAP.csv');ids=np.asarray(sorted(rows.patient.unique()));draw=np.random.default_rng(42).integers(0,80,size=(10000,80))
    counts=np.stack([(draw==j).sum(1) for j in range(80)],axis=1)
    for name,l,r in [('DELTA_effect','A1','A0'),('PR_effect','A2','A1'),('RawCorrection_package_effect','A3','A2')]:
        ll=rows[rows.arm==l].set_index('patient').loc[ids,METRICS];rr=rows[rows.arm==r].set_index('patient').loc[ids,METRICS];delta=ll.to_numpy()-rr.to_numpy();dist=counts@delta/80
        for j,m in enumerate(METRICS):
            v=boot[(boot.increment==name)&(boot.metric==m)].iloc[0]
            assert np.allclose([v.delta,v.ci_low,v.ci_high],[delta[:,j].mean(),*np.percentile(dist[:,j],[2.5,97.5])],rtol=0,atol=1e-12)
    initialization=jread(pub/'INITIALIZATION_PARITY.json');assert initialization['status']=='PASS'
    for r in fr['selections']:
        ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['binding']['initial_hash']==initialization['folds'][r['fold']-1]['A0_sha256']
        assert ck['binding']['penalty']==0 and ck['binding']['max_epochs']==30
        assert np.isclose(ck['threshold']/.005,round(ck['threshold']/.005))
    files={p.name:sha(p) for p in pub.glob('*.csv')}
    write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',new_cells=10,reused_cells=10,patient_metric_rows_checked=checked,bootstrap_independent_multinomial_exact=True,DELTA14_exact_original_R14=True,original_reference_checkpoints_unchanged=True,aggregate_hashes=files,protocol_sha256=sha(ROOT/'PROTOCOL_LOCK.json'),freeze_sha256=sha(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json'),prediction_seal_sha256=sha(a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    write(pub/'RUN_STATUS.json',dict(status='COMPLETE_VERIFIED',new_training_cells=10,new_outer_cells=10,reused_reference_cells=10,patients=80,channels=7635,scientific_terminal=jread(pub/'SCIENTIFIC_GATE.json')['terminal']))
    print('PROGRESSIVE_ABLATION_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':run(args())
