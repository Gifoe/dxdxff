"""Patient-equal tables and fixed paired within-source bootstrap."""
import csv
import numpy as np
from task_io import *

SOURCES=['hup','lzu','multicenter','pediatric']

def draws(ids,source):
    rng=np.random.default_rng(42);parts=[]
    for s,n in zip(SOURCES,[36,21,15,8]):
        ix=np.flatnonzero(source==s);assert len(ix)==n
        parts.append(ix[rng.integers(0,n,(10000,n))])
    return np.concatenate(parts,axis=1)

def matrices(rows):
    ids=sorted(set(r['patient'] for r in rows));assert len(ids)==80
    lookup={pid:i for i,pid in enumerate(ids)};source=np.empty(80,object);out={}
    for model in MODELS:
        for condition in ARMS:
            group=[r for r in rows if r['model']==model and r['condition']==condition]
            assert len(group)==80 and sum(r['channels'] for r in group)==7635
            a=np.empty((80,9))
            for r in group:
                i=lookup[r['patient']];source[i]=r['source'];a[i]=[r[k] for k in METRICS]
            assert np.isfinite(a).all();out[model,condition]=a
    return ids,source,out

def contrasts(mat):
    for model in MODELS:
        for arm in ARMS:
            yield 'absolute',model,arm,mat[model,arm]
        for arm in ARMS:
            if arm!='FULL_PR':yield 'FULL_MINUS',model,arm,mat[model,'FULL_PR']-mat[model,arm]
        for arm in ['SCORE_CENTER','SCORE_Z']:yield 'POST_MINUS_NOPR',model,arm,mat[model,arm]-mat[model,'NO_PR']

def report(rows):
    ids,source,mat=matrices(rows);ix=draws(ids,source);counts=np.zeros((10000,80),np.int16)
    for i in range(10000):counts[i]=np.bincount(ix[i],minlength=80)
    main=[];sources=[];summary=[];folds=[];coverage=[];ci=[];family=[]
    for model in MODELS:
        for arm in ARMS:
            a=mat[model,arm];point=a.mean(0)
            main.append(dict(model=model,condition=arm,patients=80,channels=7635,alias='NO_PR' if model=='LINEAR_SVM' and arm=='GLOBAL' else '',**dict(zip(METRICS,point.tolist()))))
            srcpoints=[]
            for s in SOURCES:
                b=a[source==s];v=b.mean(0);srcpoints.append(v);sources.append(dict(model=model,condition=arm,source=s,patients=len(b),**dict(zip(METRICS,v.tolist()))))
            sp=np.array(srcpoints)
            summary.append(dict(model=model,condition=arm,**{'center_mean_'+m:float(sp[:,j].mean()) for j,m in enumerate(METRICS)},**{'worst_'+m:float(sp[:,j].min()) for j,m in enumerate(METRICS) if m!='ez_fraction_bias'},worst_absolute_ez_fraction_bias=float(np.abs(sp[:,-1]).max())))
            for fold in range(1,6):
                rr=[r for r in rows if r['model']==model and r['condition']==arm and r['fold']==fold]
                folds.append(dict(model=model,condition=arm,fold=fold,patients=len(rr),channels=sum(r['channels'] for r in rr),**{m:float(np.mean([r[m] for r in rr])) for m in METRICS}))
            for j,m in enumerate(METRICS):coverage.append(dict(model=model,condition=arm,metric=m,patients=80,finite_patients=int(np.isfinite(a[:,j]).sum())))
    for kind,model,arm,value in contrasts(mat):
        samples=(counts.astype(float)@value)/80;lim=np.quantile(samples,[.025,.975],axis=0);point=value.mean(0)
        ci.extend(dict(kind=kind,model=model,condition=arm,metric=m,mean=float(point[j]),low=float(lim[0,j]),high=float(lim[1,j]),patients=80,draws=10000) for j,m in enumerate(METRICS))
        if kind=='FULL_MINUS' and arm=='SCORE_Z':
            bounds=np.quantile(samples,[.05/12,1-.05/12],axis=0)
            family.extend(dict(model=model,contrast='FULL_PR_MINUS_SCORE_Z',metric=m,mean=float(point[j]),low=float(bounds[0,j]),high=float(bounds[1,j]),family_size=6,confidence=1-.05/6) for j,m in enumerate(METRICS) if m in ['macro_f1','auroc'])
    assert len(ci)==405 and len(family)==6
    for name,value in [('MAIN_RESULTS',main),('SOURCE_RESULTS',sources),('SOURCE_SUMMARY',summary),('FOLD_RESULTS',folds),('COVERAGE',coverage),('INTERVALS',ci),('PRIMARY_FAMILY_INTERVALS',family)]:csv_out(name,value)
    write(PUB/'REPORT_STATUS.json',dict(status='AGGREGATES_COMPUTED_PENDING_INDEPENDENT_VERIFICATION',configurations=21,patients_per_configuration=80,channels_per_configuration=7635,nominal_intervals=405,family_intervals=6,unique_patients=80))
    return main,family

def findings():
    with (PUB/'MAIN_RESULTS.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    with (PUB/'PRIMARY_FAMILY_INTERVALS.csv').open(encoding='utf-8') as f:intervals=list(csv.DictReader(f))
    out=['# PR 表征归一化与输出后处理：完整结果','',
         '固定的三个模型、一个既有共同运行和原五折划分；全部80人、7635通道保留。以下是历史队列上的探索性补充对照，未按结果筛选运行或患者。','',
         'Macro-F1、AUROC、AP、EZ-F1、NEZ-F1、ACC、MRR、Top-1 为患者等权均值；表内以百分数显示。原JBHI图表未替换。','',
         '|模型|条件|Macro-F1|AUROC|AP|EZ-F1|NEZ-F1|ACC|MRR|Top-1|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:out.append('|'+r['model']+'|'+r['condition']+'|'+'|'.join(f'{100*float(r[m]):.2f}' for m in METRICS[:-1])+'|')
    out+=['','## Full PR 相对输出 Score-Z','', '|模型|指标|差值(pp)|六项调整区间(pp)|','|---|---|---:|---|']
    for r in intervals:out.append(f"|{r['model']}|{r['metric']}|{100*float(r['mean']):+.2f}|[{100*float(r['low']):+.2f}, {100*float(r['high']):+.2f}]|")
    out+=['','输出中心化/标准化保持冻结No PR排序，因此AUROC、AP、MRR和Top-1逐患者完全一致。分类指标使用各自验证集阈值。Global和No PR在SVM中是同一结果的显式别名。','',
          '完整的四来源、Center Mean、Worst、全部九项指标、405条名义区间和六条家族调整区间分别见CSV。EZ比例偏差的最差来源按绝对偏差定义。','',
          '区间跨零说明本对照未支持差异，不作等价解释。表征PR的排序贡献需由Full PR与Score-Z的排序差异支持；不能把输出阈值改善称为排序改善。']
    (PUB/'FINDINGS.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
