"""Independent sealed-score/model/role verification; no training or inference."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
import final_stage as f
c=f.c


def main():
    binding,choices,_=f.setup();terminal=c.jr(f.PUBLIC/'FINAL_TERMINAL.json')
    assert terminal['status']=='COMPLETE_VERIFIED_FOUR_PRECOMPAND_PROCEDURES'
    freeze=c.jr(f.FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');fr=c.read(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl')
    assert freeze['input']==binding and fr['input']==binding
    assert freeze['model_freeze']==c.sha(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl')
    assert freeze['evaluation_source']==c.sha(c.HERE/'final_stage.py')
    seal=c.jr(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');assert seal['freeze']==c.sha(f.FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert seal['all_scores_sealed_before_fullOuter_truth'] and len(seal['cells'])==20
    for row in fr['rows']:
        assert c.sha(row['body'])==row['body_sha256']
        if row['matrix']:
            assert c.sha(row['matrix'])==row['matrix_sha256']
            state=f.torch.load(row['matrix'],map_location='cpu',weights_only=False)
            assert state['binding']['input']==binding and state['matrix'].shape==(24,24)
            assert state['binding']['body_checkpoint_sha256']==row['body_sha256']
            assert state['selection']['threshold']==row['threshold'] and state['epoch']==row['epoch']
        if row['procedure']=='SELECTED':assert row['arm']==choices[row['fold']]
    meta,banks=c.io.identity_and_banks(labels=False)
    archive=c.Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert c.sha(archive)==c.read(c.io.BASE/'RECONSTRUCTION_PRIVATE.pkl')['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:
        assert np.array_equal(z['patient'],meta['patient']) and np.array_equal(z['channel'],meta['channel']);y=1-z['y']
    private=pd.DataFrame(c.read(f.FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl'))
    public=pd.read_csv(f.PUBLIC/'OUTER_PATIENT_MEAN.csv').set_index('procedure')
    reports=[];maximum=0.
    for procedure in f.PROCEDURES:
        data=[];identities=set();indices=[]
        for fold in range(1,6):
            p=f.FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl';assert c.sha(p)==seal['cells'][f'{procedure}_{fold}']
            q=c.read(p);assert q['freeze']==seal['freeze'];ix=q['indices'];pn=q['score_nez'];prob=1-pn.astype(np.float64);pred=pn<q['threshold']
            row=next(r for r in fr['rows'] if r['fold']==fold and r['procedure']==procedure)
            assert q['body_sha256']==row['body_sha256'] and q['matrix_sha256']==row['matrix_sha256'] and q['threshold']==row['threshold']
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test']
            assert set(meta['patient'][ix]).isdisjoint(banks[fold-1]['roles']['fit']|banks[fold-1]['roles']['validation'])
            if procedure=='SELECTED':
                other=c.read(f.FINAL/'outer_scores_private'/f'{choices[fold]}_{fold}.pkl')
                assert np.array_equal(pn,other['score_nez']) and q['threshold']==other['threshold']
            indices.extend(ix)
            for pid in np.unique(meta['patient'][ix]):
                assert pid not in identities;identities.add(pid);loc=meta['patient'][ix]==pid
                yy=y[ix][loc];pp=prob[loc];pr=pred[loc];ch=meta['channel'][ix][loc]
                order=sorted(range(len(yy)),key=lambda j:(-float(pp[j]),str(ch[j])))
                first=next(j for j,k in enumerate(order) if yy[k])
                r=dict(macro_f1=f1_score(yy,pr,average='macro',zero_division=0),f1=f1_score(yy,pr,pos_label=1,zero_division=0),
                       nez_f1=f1_score(yy,pr,pos_label=0,zero_division=0),accuracy=accuracy_score(yy,pr),auroc=roc_auc_score(yy,pp),
                       auprc=average_precision_score(yy,pp),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pr.mean()-yy.mean()))
                old=private[(private.procedure==procedure)&(private.fold==fold)&(private.patient==pid)]
                assert len(old)==1
                maximum=max(maximum,float(np.max(np.abs(np.array([r[m] for m in c.METRICS])-old[c.METRICS].to_numpy()[0]))))
                data.append(r)
        assert len(identities)==len(data)==80 and len(indices)==len(set(indices))==7635 and set(indices)==set(range(7635))
        for m in c.METRICS:assert abs(float(np.mean([r[m] for r in data]))-float(public.loc[procedure,m]))<1e-12
        reports.append(dict(procedure=procedure,patients=80,channels=7635,macro_f1=float(np.mean([r['macro_f1'] for r in data]))))
    assert maximum<1e-12
    c.write(f.PUBLIC/'INDEPENDENT_FINAL_VERIFICATION.json',dict(status='PASS',procedures=4,logical_models=20,patient_rows=320,reports=reports,
          all_frozen_body_matrix_threshold_hashes_verified=True,selected_aliases_exact=True,archive_identity_and_ownOuter_roles_exact=True,
          all9_patient_metrics_independently_recomputed=True,maximum_metric_difference=maximum,
          verifier_source_sha256=c.sha(c.HERE/'verify_final.py'),score_seal_sha256=c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),
          no_training_or_inference=True,goal_point_established=terminal['goal_point_established']))
    print('PRECOMPAND_INDEPENDENT_FINAL_VERIFICATION_PASS',reports,flush=True)


if __name__=='__main__':main()
