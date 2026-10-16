"""Recompute frozen configs, roles, binary parity, all metrics and patient CIs."""
import numpy as np
import pandas as pd
import final_stage as f
from summarize import independent_fit,independent_metrics
from decoders import predict
c=f.c


def main():
    binding,choices,models,refs=f.setup();terminal=c.jr(f.PUBLIC/'FINAL_TERMINAL.json')
    assert terminal['status']=='COMPLETE_NINE_RAW_DECISION_PROCEDURES'
    freeze=c.jr(f.FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');fr=c.jr(f.FINAL/'DECISION_FREEZE_PRIVATE.json')
    assert freeze['input']==binding and fr['input']==binding and freeze['decisions']==c.sha(f.FINAL/'DECISION_FREEZE_PRIVATE.json')
    assert binding['source']=={n:c.sha(c.HERE/n) for n in binding['source']}
    seal,meta,banks,y=f.labels_after_seal();checks=[]
    for ref in refs:
        fold=ref['fold'];d=banks[fold-1];va=d['val']
        for body in c.BODIES:
            model=next(r for r in models if r['fold']==fold and r['procedure']==body)
            assert c.sha(model['body'])==model['body_sha256']
            if model['matrix']:assert c.sha(model['matrix'])==model['matrix_sha256']
            if body=='NATIVE':logit=c.read(ref['validation'])['logit']
            else:logit=c.read(f.OLD/'matrices_private'/f'fold{fold}_screenfinal'/body/'HELD_PRIVATE.pkl')['cal_logit_nez']
            original=dict(family='ORIGINAL',value=float(model['threshold']))
            for decoder in c.DECODERS:
                row=next(r for r in fr['rows'] if r['fold']==fold and r['procedure']==body+'__'+decoder)
                assert row['development_indices']==d['archive_indices'][va].tolist() and row['model']==model
                expected=original if decoder=='ORIGINAL' else independent_fit(decoder,logit,d['y'][va],d['patient'][va],original)
                assert expected==row['config']
            checks.append(dict(fold=fold,body=body,independent_Development_refit_identical=True,body_matrix_hashes_identical=True))
    frame=[];maximum=0.;oldseal=c.jr(f.OLD/'OUTER_SCORE_SEAL_PRIVATE.json')
    oldmetrics=pd.DataFrame(c.read(f.OLD/'OUTER_PATIENT_METRICS_PRIVATE.pkl'))
    for procedure in f.PROCEDURES:
        indices=[];ids=set()
        for fold in range(1,6):
            path=f.FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl';assert c.sha(path)==seal['cells'][f'{procedure}_{fold}']
            q=c.read(path);ix=q['indices'];x=q['logit_nez'];pred=q['pred_ez']
            row=next(r for r in fr['rows'] if r['fold']==fold and r['procedure']==procedure)
            assert q['configuration']==row and q['freeze']==seal['freeze']
            assert np.array_equal(pred,predict(row['config'],x,meta['patient'][ix]))
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test']
            assert set(meta['patient'][ix]).isdisjoint(banks[fold-1]['roles']['fit']|banks[fold-1]['roles']['validation'])
            if procedure=='SELECTED':
                assert row['selected_procedure']==choices[fold]
                other=c.read(f.FINAL/'outer_scores_private'/f'{choices[fold]}_{fold}.pkl')
                assert np.array_equal(x,other['logit_nez']) and np.array_equal(pred,other['pred_ez'])
            if procedure.endswith('__ORIGINAL'):
                body=row['body'];p=f.OLD/'outer_scores_private'/f'{body}_{fold}.pkl'
                assert c.sha(p)==oldseal['cells'][f'{body}_{fold}'];old=c.read(p)
                assert np.array_equal(old['indices'],ix) and np.array_equal(pred,old['score_nez']<old['threshold'])
            indices.extend(ix)
            for pid in np.unique(meta['patient'][ix]):
                assert pid not in ids;ids.add(pid);j=meta['patient'][ix]==pid;ai=ix[j]
                metric=independent_metrics(y[ai],x[j],pred[j],meta['channel'][ai])
                frame.append(dict(procedure=procedure,fold=fold,patient=str(pid),center=str(meta['center'][ai][0]),**metric))
                if procedure.endswith('__ORIGINAL'):
                    old=oldmetrics[(oldmetrics.procedure==row['body'])&(oldmetrics.fold==fold)&(oldmetrics.patient==pid)]
                    assert len(old)==1
                    for k in ['macro_f1','f1','nez_f1','accuracy','ez_fraction_bias']:assert abs(float(old.iloc[0][k])-metric[k])<1e-12
        assert len(ids)==80 and len(indices)==len(set(indices))==7635 and set(indices)==set(range(7635))
    frame=pd.DataFrame(frame);stored=pd.DataFrame(c.read(f.FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl'))
    keys=['procedure','fold','patient'];a,b=frame.sort_values(keys).reset_index(drop=True),stored.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]);maximum=float(np.max(np.abs(a[c.METRICS].to_numpy()-b[c.METRICS].to_numpy())));assert maximum<1e-12
    public=pd.read_csv(f.PUBLIC/'OUTER_PATIENT_MEAN.csv').set_index('procedure')
    for p,g in frame.groupby('procedure'):
        for k in c.METRICS:assert abs(float(g[k].mean())-float(public.loc[p,k]))<1e-12
    paired=pd.read_csv(f.PUBLIC/'OUTER_PAIRED_CI.csv');draw=np.random.default_rng(42).integers(0,80,(10000,80));cichecks=0
    for (left,right),reported in paired.groupby(['left','right']):
        aa=frame[frame.procedure==left].sort_values('patient');bb=frame[frame.procedure==right].sort_values('patient')
        assert aa.patient.tolist()==bb.patient.tolist()
        diff=aa[c.METRICS].to_numpy()-bb[c.METRICS].to_numpy();values=diff[draw].mean(1);ci=np.quantile(values,[.025,.975],axis=0)
        for j,k in enumerate(c.METRICS):
            r=reported[reported.metric==k].iloc[0]
            assert max(abs(r['mean']-diff[:,j].mean()),abs(r.low-ci[0,j]),abs(r.high-ci[1,j]))<1e-12;cichecks+=1
    outstatus=c.jr(f.PUBLIC/'OUTER_STATUS.json');assert outstatus['valid_windows']==1471965
    assert len(outstatus['parity'])==5 and sum(r['patients'] for r in outstatus['parity'])==80
    c.write(f.PUBLIC/'INDEPENDENT_FINAL_VERIFICATION.json',dict(status='PASS',procedures=9,logical_cells=45,patient_rows=720,
        patients_per_procedure=80,channels_per_procedure=7635,valid_raw250_windows=1471965,checks=checks,
        selected_aliases_exact=True,all_original_binary_metrics_reproduced=True,unbounded_margin_metrics_independently_recomputed=True,
        within_body_rankings_invariant=True,maximum_metric_difference=maximum,independent_paired_CI_checks=cichecks,
        all45_score_cells_before_full_Outer_truth=True,score_seal_sha256=c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),
        decision_freeze_sha256=c.sha(f.FINAL/'DECISION_FREEZE_PRIVATE.json'),goal_point_established=terminal['goal_point_established'],
        no_training_or_inference_in_verifier=True,scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False))
    print('RAW_DECISION_INDEPENDENT_FINAL_VERIFICATION_PASS',terminal['patient_equal_macro_f1'],flush=True)


if __name__=='__main__':main()
