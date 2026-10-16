"""Independent model/Dev/role/alias/metric/CI audit; never refit or rerun Outer."""
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
import final_stage as f
from model import Model
from admit import evaluate
c=f.c


def main():
    binding,choices,refs=f.setup();terminal=c.jr(f.PUBLIC/'FINAL_TERMINAL.json');assert terminal['status']=='COMPLETE_SEVEN_FRESH_RAW_PROCEDURES'
    fr=c.read(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl');freeze=c.jr(f.FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert fr['input']==binding and freeze['input']==binding and freeze['model_freeze']==c.sha(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl')
    assert binding['source']=={n:c.sha(c.HERE/n) for n in binding['source']}
    seal,meta,banks,y=f.sealed_labels();_,labeled=c.io.identity_and_banks(labels=True);store=c.RawStore();devchecks=[]
    for row in fr['rows']:
        if row['procedure']=='SELECTED':continue
        if row['kind']=='FIXED_REFERENCE':continue
        p=row['checkpoint'];assert c.sha(p)==row['checkpoint_sha256'];best=c.torch.load(p,map_location='cpu',weights_only=False)
        assert best['binding']['input']==binding and best['binding']['arm']==row['arm'] and best['epoch']==row['epoch']>=1
        assert best['selection']['threshold']==row['threshold'];model=Model(row['arm'],row['fold']).cuda()
        assert c.state_hash(model.state_dict())==row['initial_state_sha256']==best['binding']['initial_state_sha256']
        model.load_state_dict(best['model']);d=labeled[row['fold']-1];assert np.array_equal(d['archive_indices'],banks[row['fold']-1]['archive_indices'])
        q=c.read(f.FINAL/'models_private'/f"fold{row['fold']}_screenfinal"/row['arm']/'HELD_PRIVATE.pkl')
        assert q['binding']==best['binding'] and np.array_equal(q['cal_archive_indices'],d['archive_indices'][d['val']])
        assert np.array_equal(q['archive_indices'],d['archive_indices'][d['val']])
        cl=evaluate(model,d,d['val'],store);error=float(np.max(np.abs(cl-q['cal_logit_nez'])));assert error<=1e-6
        grid=np.round(np.arange(0,1.0001,.005),3)
        assert np.array_equal(c.probability(cl)[:,None]<grid,c.probability(q['cal_logit_nez'])[:,None]<grid)
        assert c.calibrate(d,d['val'],cl)==best['selection']
        devchecks.append(dict(fold=row['fold'],arm=row['arm'],maximum_Development_replay_error=error,all201_grid_decisions_exact=True,random_initial_state_exact=True))
        del model;gc.collect();c.torch.cuda.empty_cache()
    assert len(devchecks)==20
    for fold in range(1,6):
        assert len({r['initial_state_sha256'] for r in fr['rows'] if r['fold']==fold and r['procedure'] in c.TRAINED})==1
    rows=[];oldseal=c.jr(f.OLD/'OUTER_SCORE_SEAL_PRIVATE.json')
    for procedure in f.PROCEDURES:
        ids=set();indices=[]
        for fold in range(1,6):
            p=f.FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl';assert c.sha(p)==seal['cells'][f'{procedure}_{fold}'];q=c.read(p)
            row=next(r for r in fr['rows'] if r['fold']==fold and r['procedure']==procedure)
            assert q['configuration']==row and q['freeze']==seal['freeze'];ix=q['indices'];pn=q['score_nez'];pp=1-pn.astype(float);pred=pn<row['threshold']
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test']
            assert set(meta['patient'][ix]).isdisjoint(banks[fold-1]['roles']['fit']|banks[fold-1]['roles']['validation'])
            if procedure=='SELECTED':
                assert row['selected_arm']==choices[fold]
                other=c.read(f.FINAL/'outer_scores_private'/f'{choices[fold]}_{fold}.pkl')
                assert np.array_equal(pn,other['score_nez']) and row['threshold']==other['configuration']['threshold']
            elif row['kind']=='FIXED_REFERENCE':
                key=f"{row['old_procedure']}_{fold}";op=f.OLD/'outer_scores_private'/(key+'.pkl');assert c.sha(op)==oldseal['cells'][key]
                old=c.read(op);assert np.array_equal(pn,old['score_nez']) and np.array_equal(ix,old['indices']) and row['threshold']==old['threshold']
            indices.extend(ix)
            for pid in np.unique(meta['patient'][ix]):
                assert pid not in ids;ids.add(pid);j=meta['patient'][ix]==pid;ai=ix[j];yy=y[ai];p=pp[j];pr=pred[j];ch=meta['channel'][ai]
                order=sorted(range(len(yy)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if yy[i])
                metric=dict(macro_f1=f1_score(yy,pr,average='macro',zero_division=0),f1=f1_score(yy,pr,zero_division=0),nez_f1=f1_score(yy,pr,pos_label=0,zero_division=0),
                    accuracy=accuracy_score(yy,pr),auroc=roc_auc_score(yy,p),auprc=average_precision_score(yy,p),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pr.mean()-yy.mean()))
                rows.append(dict(procedure=procedure,fold=fold,patient=str(pid),center=str(meta['center'][ai][0]),**metric))
        assert len(ids)==80 and len(indices)==len(set(indices))==7635 and set(indices)==set(range(7635))
    frame=pd.DataFrame(rows);stored=pd.DataFrame(c.read(f.FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl'));keys=['procedure','fold','patient']
    a,b=frame.sort_values(keys).reset_index(drop=True),stored.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]);maximum=float(np.max(np.abs(a[c.METRICS].to_numpy()-b[c.METRICS].to_numpy())));assert maximum<1e-12
    public=pd.read_csv(f.PUBLIC/'OUTER_PATIENT_MEAN.csv').set_index('procedure')
    for arm,g in frame.groupby('procedure'):
        for m in c.METRICS:assert abs(float(g[m].mean())-float(public.loc[arm,m]))<1e-12
    draw=np.random.default_rng(42).integers(0,80,(10000,80));ci_checks=0
    for (left,right),reported in pd.read_csv(f.PUBLIC/'OUTER_PAIRED_CI.csv').groupby(['left','right']):
        a=frame[frame.procedure==left].sort_values('patient');b=frame[frame.procedure==right].sort_values('patient');assert a.patient.tolist()==b.patient.tolist()
        diff=a[c.METRICS].to_numpy()-b[c.METRICS].to_numpy();interval=np.quantile(diff[draw].mean(1),[.025,.975],axis=0)
        for j,m in enumerate(c.METRICS):
            r=reported[reported.metric==m].iloc[0]
            assert max(abs(r['mean']-diff[:,j].mean()),abs(r.low-interval[0,j]),abs(r.high-interval[1,j]))<1e-12;ci_checks+=1
    coverage=c.jr(f.PUBLIC/'OUTER_STATUS.json')['coverage'];assert sum(r['valid_windows'] for r in coverage)==1471965
    assert sum(r['patients'] for r in coverage)==80 and sum(r['channels'] for r in coverage)==7635
    c.write(f.PUBLIC/'INDEPENDENT_FINAL_VERIFICATION.json',dict(status='PASS',procedures=7,logical_cells=35,fresh_fits=20,patient_metric_rows=560,
        patients_per_procedure=80,channels_per_procedure=7635,valid_Raw250_windows=1471965,Development_checks=devchecks,
        all_four_initializers_exact_per_fold=True,source_role_and_archive_identity_exact=True,selected_aliases_exact=True,reference_predictions_unchanged=True,
        all9_patient_metrics_independent=True,maximum_metric_error=maximum,independent_paired_CI_checks=ci_checks,
        score_seal_sha256=c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),model_freeze_sha256=c.sha(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl'),
        primary='SELECTED',primary_point_over_065=terminal['primary_point_over_065'],no_training_or_Outer_inference_in_verifier=True,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False))
    print('FRESH_RAW_INDEPENDENT_FINAL_VERIFICATION_PASS',terminal['patient_equal_macro_f1'],flush=True)


if __name__=='__main__':main()
