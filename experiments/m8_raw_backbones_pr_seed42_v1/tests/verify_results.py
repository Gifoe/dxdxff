"""Independent posthoc verification from sealed patient/channel predictions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from support import *
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,roc_auc_score,average_precision_score,brier_score_loss,log_loss


def run(runtime):
    lock,core,rec,banks,binding=setup(runtime);pub=runtime/'public';checked=0;allpatients=[]
    for model in lock['models']:
        status=jr(pub/f'{model}_OUTER_STATUS.json');assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
        fr=jr(runtime/model/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(runtime/model/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
        assert fr['binding']==binding and len(fr['selections'])==10
        assert sha(runtime/model/'FROZEN_BEFORE_OUTER_PRIVATE.json')==status['freeze_sha256']==seal['freeze_sha256']
        assert sha(runtime/model/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==status['seal_sha256']
        assert status['predictions_sha256']==sha(runtime/model/'PREDICTIONS_PRIVATE.pkl')
        frame=read(runtime/model/'PREDICTIONS_PRIVATE.pkl');patients=read(runtime/model/'PATIENT_METRICS_PRIVATE.pkl');allpatients.append(patients)
        assert len(frame)==15270 and len(patients)==160
        independent=EV.patient_rows(frame);idframe=pd.DataFrame(independent).set_index(['version','fold','patient']).sort_index()
        pframe=patients.set_index(['version','fold','patient']).sort_index()
        assert idframe.index.equals(pframe.index)
        assert np.allclose(idframe[EV.CLASSIFICATION+EV.RANKING],pframe[EV.CLASSIFICATION+EV.RANKING],rtol=0,atol=1e-12,equal_nan=True)
        for (version,fold,pid),g in frame.groupby(['version','fold','patient']):
            r=pframe.loc[version,fold,pid];y=g.y.to_numpy();p=g.probability.to_numpy();pred=g.prediction.to_numpy()
            assert set(y)=={0,1}
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
            f=f1_score(y,pred,average=None,labels=[0,1],zero_division=0)
            values=dict(accuracy=accuracy_score(y,pred),macro_f1=f.mean(),f1=f[1],nez_f1=f[0],auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p),log_loss=log_loss(y,np.clip(p,1e-15,1-1e-15)),TP=tp,FP=fp,TN=tn,FN=fn,ez_fraction_bias=float(pred.mean()-y.mean()))
            for key,value in values.items():assert np.isclose(value,r[key],rtol=0,atol=1e-12),key
            checked+=1
        for r in fr['selections']:
            cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
            assert sha(r['checkpoint'])==r['checkpoint_sha256'] and cp['binding']['input']==binding
            d=banks[r['fold']-1];q=read(Path(r['checkpoint']).parent/'VALIDATION_PRIVATE.pkl')
            assert np.array_equal(q['indices'],d['archive_indices'][d['val']])
            assert EV.threshold(1-d['y'][d['val']],q['score_ez'],d['patient'][d['val']])['threshold']==r['threshold']
            h=torch.load(Path(r['checkpoint']).parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)['history']
            best=max(h,key=lambda q:(q['validation_macro_f1'],q['validation_positive_f1'],-q['epoch']))
            assert best['epoch']==r['epoch'] and best['threshold']==r['threshold']
        originals={r['fold']:r['initial_sha256'] for r in fr['selections'] if r['version']=='Original'}
        assert all(r['initial_sha256']==originals[r['fold']] for r in fr['selections'])
    assert checked==640
    patients=pd.concat(allpatients);main=pd.read_csv(pub/'main_results.csv').set_index(['model','version']).sort_index()
    recomputed=patients.groupby(['model','version'])[EV.CLASSIFICATION+EV.RANKING].mean().sort_index()
    assert np.allclose(recomputed,main[recomputed.columns],rtol=0,atol=1e-12,equal_nan=True)
    bootstrap=pd.read_csv(pub/'paired_pr_intervals.csv');maxerr=0.
    for model in lock['models']:
        l=patients[(patients.model==model)&(patients.version=='Original')].set_index('patient').sort_index()
        r=patients[(patients.model==model)&(patients.version=='PR')].set_index('patient').sort_index()
        assert l.index.equals(r.index) and np.array_equal(l.fold,r.fold)
        delta=r[recomputed.columns].to_numpy()-l[recomputed.columns].to_numpy()
        draws=np.random.default_rng(42).integers(0,80,size=(10000,80));dist=np.nanmean(delta[draws],axis=1)
        for j,m in enumerate(recomputed.columns):
            z=bootstrap[(bootstrap.model==model)&(bootstrap.metric==m)].iloc[0]
            want=np.array([EV.nanmean(delta[:,j]),*np.nanpercentile(dist[:,j],[2.5,97.5])]);got=np.array([z.delta,z.ci_low,z.ci_high])
            assert np.allclose(want,got,rtol=0,atol=1e-12,equal_nan=True)
            if np.isfinite(want).any():maxerr=max(maxerr,float(np.nanmax(np.abs(want-got))))
    for p in pub.glob('*.csv'):
        cols=pd.read_csv(p,nrows=0).columns
        assert not set(cols)&{'patient','patient_id','channel','channel_name','checkpoint'}
        assert not any('balanced' in k.lower() for k in cols)
    write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',configurations=8,patient_rows_verified=checked,channels_per_configuration=7635,all_five_outer_folds_complete=True,paired_initializations_identical=True,validation_only_selection_verified=True,independent_sklearn_check=True,bootstrap_draws=10000,bootstrap_seed=42,bootstrap_max_error=maxerr,public_CSV_privacy_check=True))
    write(pub/'PUBLIC_SHA256.json',{p.name:sha(p) for p in sorted(pub.iterdir()) if p.is_file() and p.name!='PUBLIC_SHA256.json'})
    print('ALL_EIGHT_INDEPENDENT_VERIFICATION_PASS',checked,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=RUNTIME);run(p.parse_args().runtime)
