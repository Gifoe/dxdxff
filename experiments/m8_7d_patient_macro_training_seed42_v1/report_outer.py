"""Read full truth only after the entire frozen score seal is verified."""
from common import *

FINAL=ROOT/'final_stage_v1';PUBLIC=FINAL/'public'


def load_rows():
    seal=jr(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');fh=sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert seal['freeze']==fh and seal['all15_cells_sealed_before_full_truth'] and len(seal['files'])==15
    frozen=jr(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');assert frozen['selected'] in NEW
    for name,h in frozen['final_sources'].items():assert sha(HERE/name)==h
    for name,h in frozen['input']['source'].items():assert sha(HERE/name)==h
    for key,h in seal['files'].items():assert sha(FINAL/'scores_private'/(key+'.pkl'))==h
    context=read(FINAL/'METADATA_PRIVATE.pkl');meta=context['meta'];archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==context['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y_nez=q['y'].copy()
    assert len(y_nez)==7635 and int((1-y_nez).sum())==1743
    rows=[];confusion=[]
    for key,h in sorted(seal['files'].items()):
        arm,fold=key.rsplit('_',1);fold=int(fold);q=read(FINAL/'scores_private'/(key+'.pkl'));assert q['freeze']==fh;ix=q['indices']
        pn=q['score_nez'];pred=pn<q['threshold'];y=1-y_nez[ix];p=1-pn.astype(float)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        for pid in sorted(set(meta['patient'][ix])):
            j=meta['patient'][ix]==pid
            rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ix][j][0]),channels=int(j.sum()),
                **EV.classification(y[j],p[j],pred[j]),**EV.ranking(y[j],p[j],meta['channel'][ix][j])))
        confusion.append(dict(arm=arm,fold=fold,tp=int((pred&(y==1)).sum()),tn=int((~pred&(y==0)).sum()),fp=int((pred&(y==0)).sum()),fn=int((~pred&(y==1)).sum())))
    f=pd.DataFrame(rows)
    for arm,g in f.groupby('arm'):assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
    return f,pd.DataFrame(confusion)


def tables(frame):
    frame.groupby('arm')[METRICS].mean().reset_index().to_csv(PUBLIC/'MAIN_RESULTS.csv',index=False)
    ff=frame.groupby(['arm','fold'])[METRICS].mean().reset_index();ff.to_csv(PUBLIC/'FOLD_RESULTS.csv',index=False)
    ff.groupby('arm')[METRICS].std(ddof=1).reset_index().to_csv(PUBLIC/'FOLD_SD.csv',index=False)
    source=[];coverage=[];groups=[]
    for arm,g in frame.groupby('arm'):
        groups.append((dict(kind='absolute',left=arm,right=''),g))
        coverage.extend(dict(arm=arm,metric=m,finite_patients=int(np.isfinite(g[m]).sum()),total_patients=len(g)) for m in METRICS)
        for center,z in g.groupby('center'):
            source.append(dict(arm=arm,center=center,patients=len(z),suppressed=len(z)<5,**{m:float(z[m].mean()) if len(z)>=5 else np.nan for m in METRICS}))
    for left,right in [('T0_GLOBAL_NORMALIZED','C2_REFERENCE'),('T1_PATIENT_BALANCED','C2_REFERENCE'),('T1_PATIENT_BALANCED','T0_GLOBAL_NORMALIZED')]:
        q=frame[frame.arm==left].merge(frame[frame.arm==right],on=['fold','patient','center'],suffixes=('_l','_r'),validate='one_to_one');assert len(q)==80
        groups.append((dict(kind='paired',left=left,right=right),pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in METRICS}))))
    intervals=[dict(**identity,**r) for identity,g in groups for r in cluster(g.to_dict('records'))]
    for name,data in [('SOURCE_RESULTS',source),('METRIC_COVERAGE',coverage),('INTERVALS',intervals)]:pd.DataFrame(data).to_csv(PUBLIC/(name+'.csv'),index=False)


def main():
    f,confusion=load_rows();tables(f);confusion.groupby('arm')[['tp','tn','fp','fn']].sum().reset_index().to_csv(PUBLIC/'POOLED_CONFUSION_COUNTS.csv',index=False)
    save(FINAL/'PATIENT_METRICS_PRIVATE.pkl',f)
    selected=jr(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')['selected'];a=float(f[f.arm==selected].macro_f1.mean());b=float(f[f.arm=='C2_REFERENCE'].macro_f1.mean())
    old=pd.read_csv(BASE_ROOT/'final_stage_v1/public/MAIN_RESULTS.csv');base=old[old.arm=='C2_INTERACTION'].iloc[0]
    for m in METRICS:assert abs(float(f[f.arm=='C2_REFERENCE'][m].mean())-float(base[m]))<1e-12
    write(PUBLIC/'FINAL_DECISION.json',dict(status='POINT_TARGET_MET' if a>.64 and a>b else 'POINT_TARGET_NOT_MET',selected=selected,
        patient_equal_Macro_F1=a,control_Macro_F1=b,delta=a-b,target_strictly_greater_than=.64,patients=80,features=7,one_classification_head=True,
        outcome_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,verification_required=True))
    print('PATIENT_MACRO_TRAINING_OUTER_REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
