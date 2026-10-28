"""Frozen sealed-score analyses. No model training and no raw-wave copying."""
import argparse,sys,zipfile,math
from pathlib import Path
sys.dont_write_bytecode=True
import trusted as t
import numpy as np
from scipy.special import expit
import analysis_math as m

MODELS=['LINEAR_SVM','TIMECONV','DRECHANNET'];ARMS=['NO_PR','FULL_PR','SCORE_Z']

def freeze(p,commit):
    lock=t.jr(t.MAIN/'SOURCE_LOCK.json');assert lock['source']==t.frozen_io.sources()
    assert t.jr(t.MAIN/'public/ADMISSION.json')['status']=='PASS'
    assert t.jr(t.PREVIEW/'public/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert t.jr(t.RAW_ROOT/'public/TimeConvCNN_OUTER_STATUS.json')['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
    q=t.read(t.MAIN/'SELECTED_MODELS_PRIVATE.pkl');assert q['binding']==t.frozen_io.digest(lock)
    cells=[r for r in q['cells'].values() if r['model'] in MODELS and r['condition'] in ARMS]
    assert len(cells)==45 and {(r['model'],r['condition'],r['fold']) for r in cells}=={(m,a,f) for m in MODELS for a in ARMS for f in range(1,6)}
    for r in cells:
        assert t.sha(r['checkpoint'])==r['checkpoint_sha256'] and t.sha(r['validation'])==r['validation_sha256']
    obj=dict(source=t.source_hashes(),source_commit=commit,parent_source_commit=lock['source_commit'],parent_source=lock['source'],parent_hashes=p.parent_protected,cells=cells,exploratory=True,core_timeconv_previously_complete=True)
    path=t.RUNTIME/'ABC_FREEZE_PRIVATE.json'
    if path.exists():assert t.jr(path)==obj
    else:t.write(path,obj)
    # Label-independent seizure sampling is fixed before any new A-C metrics.
    plan={str(q['patient']):dict(seizures=len(q['records']),subsets={str(k):m.seizure_subsets(str(q['patient']),len(q['records']),k) for k in [1,2,len(q['records'])]}) for q in p.rec['ledgers'] if len(q['records'])>=3}
    target=t.RUNTIME/'D_SEIZURE_SAMPLE_PLAN_PRIVATE.pkl'
    if target.exists():assert t.read(target)==plan
    else:t.save(target,plan)
    t.write(t.PUB/'manifest.json',dict(source_commit=commit,parent_source_commit=lock['source_commit'],source=obj['source'],logical_cells=45,patients=80,channels=7635,seizures=256,checkpoint_audit=[dict(model=r['model'],condition=r['condition'],fold=r['fold'],checkpoint_sha256=r['checkpoint_sha256'],validation_sha256=r['validation_sha256']) for r in cells],preview_score_seal_sha256=t.sha(t.PREVIEW/'SCORE_SEAL_PRIVATE.json'),seizure_plan_sha256=t.sha(target),D_eligible_patients=len(plan),timeconv_version='ICTAL_END_TO_END_ORIGINAL_AND_PR',D_pending=True))
    print('ABC_ALL45_SELECTIONS_FROZEN_D_SAMPLE_PLAN_FROZEN',len(plan),flush=True);return obj

def seal_scores(p,fr):
    import evaluation
    from math_core import score_transform
    seal={};preview=t.jr(t.PREVIEW/'SCORE_SEAL_PRIVATE.json')['files']
    for r in fr['cells']:
        model,arm,fold=r['model'],r['condition'],r['fold'];k=t.frozen_io.key(model,arm,fold)
        d=p.bank(fold);ii=np.flatnonzero(np.isin(p.meta['patient'],list(d['roles']['test'])))
        if model!='TIMECONV':
            path=t.PREVIEW/'scores_private'/(k+'.pkl');assert t.sha(path)==preview[k];q=t.read(path)
            assert q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold'] and np.array_equal(ii,q['indices'])
            out=dict(indices=ii,ranking_anchor=q['ranking_anchor'],prediction_ez=q['prediction_ez'],logit_ez=q['logit_ez'],provenance='VERIFIED_EARLY_PREVIEW_PARENT_SCORES')
        else:
            ref=evaluation.references(p,model,'NO_PR' if arm=='SCORE_Z' else arm,fold,d,ii)
            if arm=='SCORE_Z':
                z=score_transform(ref['logit_ez'],p.meta['patient'][ii],arm);prob=expit(z);pred=prob>=r['threshold']
                assert r['orientation']=='EZ_GE'
                out=dict(indices=ii,ranking_anchor=ref['ranking_anchor'],prediction_ez=pred,logit_ez=z,provenance='SEALED_ICTAL_NATIVE_LOGIT_SCORE_Z')
            else:out=dict(indices=ii,ranking_anchor=ref['ranking_anchor'],prediction_ez=ref['prediction_ez'],logit_ez=ref['logit_ez'],provenance='SEALED_ICTAL_PARENT_SCORE')
        out.update(model=model,condition=arm,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],freeze_sha256=t.sha(t.RUNTIME/'ABC_FREEZE_PRIVATE.json'))
        path=t.RUNTIME/'private/scores'/(k+'.pkl')
        if path.exists():assert t.read(path)['freeze_sha256']==out['freeze_sha256']
        else:t.save(path,out)
        seal[k]=t.sha(path)
    assert len(seal)==45;t.write(t.RUNTIME/'ABC_SCORE_SEAL_PRIVATE.json',dict(files=seal,all45_before_metrics=True))
    print('ABC_ALL45_SCORES_SEALED',flush=True);return seal

def endpoints(p,fr,seal):
    labels=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert t.sha(labels)==p.rec['binding']['archive']
    with np.load(labels,allow_pickle=True) as z:
        for k in ['patient','channel','center']:assert np.array_equal(z[k].astype(str),p.meta[k].astype(str))
        y=1-z['y']
    sources=t.source_map(p);rows=[];budget=[];curves=[]
    for r in fr['cells']:
        model,arm,fold=r['model'],r['condition'],r['fold'];k=t.frozen_io.key(model,arm,fold);path=t.RUNTIME/'private/scores'/(k+'.pkl');assert t.sha(path)==seal[k];q=t.read(path)
        ii=q['indices']
        for pid in np.unique(p.meta['patient'][ii]):
            jj=np.flatnonzero(p.meta['patient'][ii]==pid);yy=y[ii[jj]];ss=q['ranking_anchor'][jj];cc=p.meta['channel'][ii[jj]];pp=q['prediction_ez'][jj]
            a=m.metrics(yy,ss,pp,cc);b=m.metrics(yy,ss,pp,cc,True)
            assert all((not np.isfinite(a[v]) and not np.isfinite(b[v])) or abs(a[v]-b[v])<1e-12 for v in m.METRICS)
            common=dict(model=model,condition=arm,fold=fold,patient=str(pid),source=sources[str(pid)],channels=len(jj))
            rows.append(dict(common,**a))
            sorted_y=yy[m.order(ss,cc)];cumulative=np.cumsum(sorted_y);positives=int(yy.sum())
            for kind,value in m.BUDGETS:
                v=m.budget_values(yy,ss,cc,kind,value);kk=int(v['review_channels']);hits=int(cumulative[kk-1])
                assert v['nez_count']==kk-hits and abs(v['precision']-hits/kk)<1e-12
                if positives:assert abs(v['recall']-hits/positives)<1e-12
                budget.append(dict(common,budget_kind=kind,budget=value,**v))
            if arm!='SCORE_Z':
                vals=[0.]+[float(cumulative[max(1,int(math.ceil(q*len(yy))))-1]/positives) if positives else np.nan for q in np.arange(1,101)/100]
                curves.append(dict(common,recall=np.array(vals)))
    assert len(rows)==720 and len(budget)==4320 and len(curves)==480
    lookup={(r['model'],r['condition'],r['patient']):r for r in rows}
    for r in rows:
        if r['condition']=='SCORE_Z':assert all(r[v]==lookup[r['model'],'NO_PR',r['patient']][v] for v in m.RANK)
    t.csv_out('patient_endpoints',rows,True);t.csv_out('patient_budgets',budget,True);t.save(t.RUNTIME/'private/curve_endpoints.pkl',curves)
    # Replay all four supplied anchors using their actual verified parent table.
    reference=t.jr(t.PREVIEW/'public/INDEPENDENT_VERIFICATION.json');assert reference['status']=='PASS'
    import csv
    with (t.PREVIEW/'public/MAIN_RESULTS.csv').open(encoding='utf-8') as f:old=list(csv.DictReader(f))
    for r in old:
        if r['condition'] in ARMS:
            rr=[q for q in rows if q['model']==r['model'] and q['condition']==r['condition']]
            assert all(abs(np.mean([q[v] for q in rr])-float(r[v]))<1e-12 for v in ['auroc','auprc','macro_f1','f1','mrr','hit_at_1'])
    print('ABC_ALL720_PATIENT_ROWS_INDEPENDENTLY_VERIFIED',flush=True);return rows,budget,curves

def summary(rows,budgets,curves):
    ids=sorted({r['patient'] for r in rows});lookup={(r['model'],r['condition'],r['patient']):r for r in rows};source=np.array([lookup[MODELS[0],'NO_PR',pid]['source'] for pid in ids]);mat={(model,arm):np.array([[lookup[model,arm,pid][v] for v in m.METRICS] for pid in ids]) for model in MODELS for arm in ARMS}
    A=[];B=[];S=[];CM=[];BS=[];bins=[];C=[];CP=[];curve=[]
    for model in MODELS:
        for arm in ARMS:
            a=mat[model,arm];assert a.shape==(80,len(m.METRICS))
            for j,v in enumerate(m.METRICS):A.append(dict(model=model,condition=arm,metric=v,**m.interval(a[:,j],source)))
            points=[]
            for s in m.SOURCES:
                b=a[source==s];points.append(np.nanmean(b,0));S.append(dict(model=model,condition=arm,source=s,patients=len(b),**dict(zip(m.METRICS,np.nanmean(b,0).tolist()))))
            point=np.array(points);CM.append(dict(model=model,condition=arm,**{'center_mean_'+v:float(np.nanmean(point[:,j])) for j,v in enumerate(m.METRICS)},**{'worst_'+v:float(np.nanmin(point[:,j])) for j,v in enumerate(m.METRICS) if v!='ez_fraction_bias'},worst_absolute_ez_fraction_bias=float(np.nanmax(np.abs(point[:,-1])))))
        for other in ['NO_PR','SCORE_Z']:
            delta=mat[model,'FULL_PR']-mat[model,other]
            for j,v in enumerate(m.METRICS):
                family=6 if other=='SCORE_Z' and v in ['macro_f1','auroc'] else 12 if other=='NO_PR' and v in ['auroc','auprc','macro_f1','f1'] else None
                ci=m.interval(delta[:,j],source,family);ci=dict(adjusted_low=np.nan,adjusted_high=np.nan,family_size=0,**ci) if not family else ci
                good=delta[np.isfinite(delta[:,j]),j]
                B.append(dict(model=model,contrast='FULL_MINUS_'+other,metric=v,median=float(np.median(good)),positive_fraction=float(np.mean(good>1e-12)),zero_fraction=float(np.mean(np.abs(good)<=1e-12)),negative_fraction=float(np.mean(good< -1e-12)),**ci))
                if other=='NO_PR' and v in ['auroc','auprc','macro_f1','f1']:
                    for s in m.SOURCES:
                        d=delta[source==s,j];BS.append(dict(model=model,source=s,metric=v,positive_fraction=float(np.mean(d>1e-12)),zero_fraction=float(np.mean(np.abs(d)<=1e-12)),negative_fraction=float(np.mean(d< -1e-12)),**m.interval(d,source[source==s])))
                    for s in ['all']+m.SOURCES:
                        d=delta[:,j] if s=='all' else delta[source==s,j];counts,edges=np.histogram(d,bins=np.linspace(-1,1,41));assert counts.sum()==np.isfinite(d).sum()
                        bins.extend(dict(model=model,source=s,metric=v,bin_low=float(lo),bin_high=float(hi),count=int(n),patients=len(d)) for lo,hi,n in zip(edges[:-1],edges[1:],counts))
    blookup={(r['model'],r['condition'],r['patient'],r['budget_kind'],r['budget']):r for r in budgets}
    for model in MODELS:
        for kind,value in m.BUDGETS:
            arrays={a:np.array([[blookup[model,a,pid,kind,value][v] for v in ['recall','precision','hit','nez_count','lift','review_channels']] for pid in ids]) for a in ARMS}
            for a in ARMS:
                for j,v in enumerate(['recall','precision','hit','nez_count','lift','review_channels']):C.append(dict(model=model,condition=a,budget_kind=kind,budget=value,metric=v,**m.interval(arrays[a][:,j],source)))
            delta=arrays['FULL_PR']-arrays['NO_PR']
            for j,v in enumerate(['recall','precision','hit','nez_count','lift','review_channels']):
                fam=9 if kind=='fraction' and v=='recall' else None;ci=m.interval(delta[:,j],source,fam);ci=dict(adjusted_low=np.nan,adjusted_high=np.nan,family_size=0,**ci) if not fam else ci
                CP.append(dict(model=model,budget_kind=kind,budget=value,metric=v,**ci))
        for arm in ['NO_PR','FULL_PR']:
            rr={r['patient']:r for r in curves if r['model']==model and r['condition']==arm};arr=np.array([rr[pid]['recall'] for pid in ids])
            for j in range(101):curve.append(dict(model=model,condition=arm,review_fraction=j/100,**m.interval(arr[:,j],source)))
    for name,rr in [('TABLE_A',A),('TABLE_B',B),('TABLE_B_SOURCE',BS),('TABLE_C_SOURCE',S),('TABLE_C_SUMMARY',CM),('TABLE_D_REVIEW',C),('TABLE_D_PAIRED',CP),('COVERAGE_CURVE',curve),('PATIENT_DISTRIBUTION_BINS',bins)]:t.csv_out(name,rr)
    return dict(patient_rows=len(rows),budget_rows=len(budgets),curve_patient_rows=len(curves),primary_family_intervals=sum(r['family_size']==6 for r in B),B_family_intervals=sum(r['family_size']==12 for r in B),C_family_intervals=sum(r['family_size']==9 for r in CP))

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--source-commit',required=True);a=arg.parse_args();assert len(a.source_commit)==40
    t.RUNTIME.mkdir(parents=True,exist_ok=True);t.PUB.mkdir(exist_ok=True)
    p=t.parents();fr=freeze(p,a.source_commit);seal=seal_scores(p,fr);rows,b,c=endpoints(p,fr,seal);counts=summary(rows,b,c);p.verify_protected()
    t.write(t.PUB/'ABC_VERIFICATION.json',dict(status='PASS',patients=80,channels_per_condition=7635,logical_cells=45,ranking_invariance='EXACT',independent_endpoint_formulas=True,independent_direct_index_bootstrap=True,frozen_parents_unchanged=True,**counts))
    t.write(t.PUB/'STATUS.json',dict(status='ABC_COMPLETE_VERIFIED_D_PENDING',A='COMPLETE',B='COMPLETE',C='COMPLETE',D='PENDING',timeconv_core_models_complete=True,timeconv_additional_normalizer_training_untouched=True))
    files=[p for p in t.PUB.iterdir() if p.suffix in ['.csv','.json']];manifest={p.name:dict(sha256=t.sha(p),bytes=p.stat().st_size) for p in files}
    path=t.RUNTIME/'public_abc_export_v1.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'public/'+p.name)
        z.writestr('PUBLIC_MANIFEST.json',__import__('json').dumps(dict(files=manifest,private_data_included=False),sort_keys=True))
    print('ABC_COMPLETE_VERIFIED',t.sha(path),flush=True)

if __name__=='__main__':main()
