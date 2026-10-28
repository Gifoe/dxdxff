"""Independent metric formulas and direct-index bootstrap; no parent evaluator."""
import csv
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from task_io import *

def independent_values(y,score,pred,channel):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=bool)
    tp=np.count_nonzero((y==1)&pred);tn=np.count_nonzero((y==0)&~pred);fp=np.count_nonzero((y==0)&pred);fn=np.count_nonzero((y==1)&~pred)
    f1=2*tp/(2*tp+fp+fn);f0=2*tn/(2*tn+fp+fn);order=sorted(range(len(y)),key=lambda i:(-float(score[i]),str(channel[i])))
    ranks=[j+1 for j,i in enumerate(order) if y[i]==1];assert ranks
    return dict(macro_f1=(f1+f0)/2,auroc=roc_auc_score(y,score),auprc=average_precision_score(y,score),f1=f1,nez_f1=f0,accuracy=(tp+tn)/len(y),mrr=1/min(ranks),hit_at_1=float(y[order[0]]==1),ez_fraction_bias=(fp-fn)/len(y))

def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8') as f:return list(csv.DictReader(f))

def verify(p):
    from evaluation import assert_frozen
    fr=assert_frozen();seal=jr(ROOT/'SCORE_SEAL_PRIVATE.json');rows=read(ROOT/'PATIENT_ENDPOINTS_PRIVATE.pkl')
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==p.rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as a:y=1-a['y']
    lookup={(r['model'],r['condition'],r['patient']):r for r in rows};checked=0
    for r in fr['cells']:
        k=key(r['model'],r['condition'],r['fold']);path=ROOT/'scores_private'/(k+'.pkl');assert sha(path)==seal['files'][k];q=read(path);ii=q['indices']
        for pid in np.unique(p.meta['patient'][ii]):
            jj=np.flatnonzero(p.meta['patient'][ii]==pid);v=independent_values(y[ii[jj]],q['ranking_anchor'][jj],q['prediction_ez'][jj],p.meta['channel'][ii[jj]])
            expected=lookup[r['model'],r['condition'],str(pid)]
            assert all(abs(float(v[m])-expected[m])<1e-12 for m in METRICS)
            if r['condition'].startswith('SCORE_'):
                old=lookup[r['model'],'NO_PR',str(pid)]
                assert all(expected[m]==old[m] for m in ['auroc','auprc','mrr','hit_at_1'])
            checked+=1
    assert checked==1680
    ids=sorted({r['patient'] for r in rows});groups={};matrix={}
    for i,pid in enumerate(ids):
        source=lookup[MODELS[0],'NO_PR',pid]['source'];groups.setdefault(source,[]).append(i)
    rng=np.random.default_rng(42);draw=[]
    for source in ['hup','lzu','multicenter','pediatric']:
        v=np.asarray(groups[source]);draw.append(v[rng.integers(0,len(v),size=(10000,len(v)))])
    draw=np.concatenate(draw,axis=1)
    for model in MODELS:
        for arm in ARMS:matrix[model,arm]=np.array([[lookup[model,arm,pid][m] for m in METRICS] for pid in ids])
    definitions=[]
    for model in MODELS:
        for arm in ARMS:definitions.append(('absolute',model,arm,matrix[model,arm]))
        for arm in ARMS:
            if arm!='FULL_PR':definitions.append(('FULL_MINUS',model,arm,matrix[model,'FULL_PR']-matrix[model,arm]))
        for arm in ['SCORE_CENTER','SCORE_Z']:definitions.append(('POST_MINUS_NOPR',model,arm,matrix[model,arm]-matrix[model,'NO_PR']))
    intervals={(r['kind'],r['model'],r['condition'],r['metric']):r for r in table('INTERVALS')};family={(r['model'],r['metric']):r for r in table('PRIMARY_FAMILY_INTERVALS')};nci=nfamily=0
    for kind,model,arm,value in definitions:
        sample=np.concatenate([value[draw[k:k+250]].mean(1) for k in range(0,10000,250)])
        nominal=np.quantile(sample,[.025,.975],axis=0);point=value.mean(0)
        for j,m in enumerate(METRICS):
            r=intervals[kind,model,arm,m];assert np.max(np.abs(np.array([float(r['mean']),float(r['low']),float(r['high'])])-[point[j],nominal[0,j],nominal[1,j]]))<1e-12;nci+=1
        if kind=='FULL_MINUS' and arm=='SCORE_Z':
            bounds=np.percentile(sample,[100*.05/12,100*(1-.05/12)],axis=0)
            for j,m in enumerate(METRICS):
                if m not in ['macro_f1','auroc']:continue
                r=family[model,m];assert max(abs(float(r['mean'])-point[j]),abs(float(r['low'])-bounds[0,j]),abs(float(r['high'])-bounds[1,j]))<1e-12;nfamily+=1
    assert nci==405 and nfamily==6
    for model in MODELS:
        assert np.array_equal(matrix['LINEAR_SVM','NO_PR'],matrix['LINEAR_SVM','GLOBAL'])
    for r in table('MAIN_RESULTS'):
        assert all(abs(float(r[m])-matrix[r['model'],r['condition']][:,j].mean())<1e-12 for j,m in enumerate(METRICS))
    for r in table('SOURCE_RESULTS'):
        rr=[x for x in rows if x['model']==r['model'] and x['condition']==r['condition'] and x['source']==r['source']]
        assert len(rr)==int(r['patients'])
        assert all(abs(float(r[m])-np.mean([x[m] for x in rr]))<1e-12 for m in METRICS)
    for r in table('FOLD_RESULTS'):
        rr=[x for x in rows if x['model']==r['model'] and x['condition']==r['condition'] and x['fold']==int(r['fold'])]
        assert len(rr)==int(r['patients']) and sum(x['channels'] for x in rr)==int(r['channels'])
        assert all(abs(float(r[m])-np.mean([x[m] for x in rr]))<1e-12 for m in METRICS)
    for r in table('SOURCE_SUMMARY'):
        values=np.array([[np.mean([x[m] for x in rows if x['model']==r['model'] and x['condition']==r['condition'] and x['source']==s]) for m in METRICS] for s in ['hup','lzu','multicenter','pediatric']])
        for j,m in enumerate(METRICS):
            assert abs(float(r['center_mean_'+m])-values[:,j].mean())<1e-12
            if m!='ez_fraction_bias':assert abs(float(r['worst_'+m])-min(values[:,j]))<1e-12
        assert abs(float(r['worst_absolute_ez_fraction_bias'])-np.abs(values[:,-1]).max())<1e-12
    for r in table('COVERAGE'):assert int(r['finite_patients'])==int(r['patients'])==80
    p.verify_protected()
    out=dict(status='PASS',logical_cells=105,new_main_fits=40,new_svm_oof_fits=30,reused_main_fits=30,alias_cells=5,postprocess_cells=30,unique_patients=80,channels_per_condition=7635,patient_condition_rows=1680,endpoint_values=15120,nominal_intervals=nci,primary_family_intervals=nfamily,output_ranking_invariance='EXACT',frozen_parents_unchanged=True)
    write(PUB/'INDEPENDENT_VERIFICATION.json',out);print('PR_REPR_INDEPENDENT_VERIFICATION_PASS',flush=True)
    return out
