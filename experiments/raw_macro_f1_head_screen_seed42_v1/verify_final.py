"""Independent sealed-array metric/identity/coverage verification; no inference."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
import final_stage as f
c=f.c


def main():
    binding,_,_=f.setup();terminal=c.jr(f.PUBLIC/'FINAL_TERMINAL.json')
    assert terminal['status']=='COMPLETE_VERIFIED_THREE_RAW_MACRO_PROCEDURES'
    seal=c.jr(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')==terminal['score_seal_sha256']
    before=c.jr(f.FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');assert before['input']==binding
    assert before['evaluation_source']==c.sha(c.HERE/'final_stage.py')
    assert seal['freeze']==c.sha(f.FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    meta,banks=c.io.identity_and_banks(labels=False)
    archive=c.Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert c.sha(archive)==c.read(c.io.BASE/'RECONSTRUCTION_PRIVATE.pkl')['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:
        assert np.array_equal(z['patient'],meta['patient']) and np.array_equal(z['channel'],meta['channel'])
        y=1-z['y']
    measured=pd.read_csv(f.PUBLIC/'OUTER_PATIENT_MEAN.csv').set_index('procedure')
    old=pd.DataFrame(c.read(f.FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl'))
    count=[];maximum=0.;reports=[]
    for procedure in f.PROCEDURES:
        rows=[];allix=[];ids=set()
        for fold in range(1,6):
            path=f.FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl'
            assert c.sha(path)==seal['cells'][f'{procedure}_{fold}'];q=c.read(path)
            assert q['freeze']==seal['freeze']
            ix=q['indices'];pn=q['score_nez'];prob=1-pn.astype(np.float64);pred=pn<q['threshold']
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test']
            assert set(meta['patient'][ix]).isdisjoint(banks[fold-1]['roles']['fit']|banks[fold-1]['roles']['validation'])
            allix.extend(ix)
            for pid in np.unique(meta['patient'][ix]):
                jj=meta['patient'][ix]==pid;yy=y[ix][jj];pp=prob[jj];vv=pred[jj];ch=meta['channel'][ix][jj]
                assert pid not in ids;ids.add(pid)
                f1=f1_score(yy,vv,pos_label=1,zero_division=0);f0=f1_score(yy,vv,pos_label=0,zero_division=0)
                # Independent explicit tie order: descending probability then channel.
                order=sorted(range(len(yy)),key=lambda j:(-float(pp[j]),str(ch[j])))
                first=next(j for j,k in enumerate(order) if yy[k])
                r=dict(macro_f1=.5*(f1+f0),f1=f1,nez_f1=f0,auroc=roc_auc_score(yy,pp),
                       auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),
                       mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(np.mean(vv)-np.mean(yy)))
                assert abs(r['macro_f1']-f1_score(yy,vv,average='macro',zero_division=0))<1e-12
                prior=old[(old.procedure==procedure)&(old.patient==pid)&(old.fold==fold)]
                assert len(prior)==1
                maximum=max(maximum,float(np.max(np.abs(np.array([r[m] for m in c.METRICS])-prior[c.METRICS].to_numpy()[0]))))
                rows.append(r)
        assert len(ids)==len(rows)==80 and len(allix)==len(set(allix))==7635
        assert set(allix)==set(range(7635))
        result=pd.DataFrame(rows)
        for m in c.METRICS:
            assert abs(float(result[m].mean())-float(measured.loc[procedure,m]))<1e-12
            count.append(dict(procedure=procedure,metric=m,finite_patients=int(np.isfinite(result[m]).sum()),total_patients=80))
        reports.append(dict(procedure=procedure,patients=80,unique_channels=7635,macro_f1=float(result.macro_f1.mean())))
    assert maximum<1e-12
    pd.DataFrame(count).to_csv(f.PUBLIC/'OUTER_METRIC_COVERAGE.csv',index=False)
    c.write(f.PUBLIC/'INDEPENDENT_FINAL_VERIFICATION.json',dict(status='PASS',reports=reports,patient_rows=240,
          archive_identity_exact=True,ownOuter_disjoint_from_ownFIT_Development=True,all80_patients_all7635_channels=True,
          all9_patient_metrics_independently_recomputed=True,maximum_patient_metric_error=maximum,
          no_training_or_inference_in_verifier=True,verifier_source_sha256=c.sha(c.HERE/'verify_final.py'),
          score_seal_sha256=c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),goal_point_established=terminal['goal_point_established']))
    print('MACRO_FINAL_INDEPENDENT_VERIFICATION_PASS',reports,flush=True)


if __name__=='__main__':main()
