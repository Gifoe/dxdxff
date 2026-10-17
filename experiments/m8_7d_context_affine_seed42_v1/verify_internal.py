"""Actual Raw-through-assembled-model replay, frozen state and independent statistics."""
import gc
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from train import linear

def check_metrics(d,ix,logit,theta,rows):
    for r in rows:
        j=d['patient'][ix]==r['patient'];y=1-d['y'][ix][j];pn=probability(logit[j])
        p=1-pn.astype(float);pred=pn<theta;ch=d['channel'][ix][j]
        order=sorted(range(len(y)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if y[i])
        values=dict(macro_f1=f1_score(y,pred,average='macro',zero_division=0),f1=f1_score(y,pred,zero_division=0),
            nez_f1=f1_score(y,pred,pos_label=0,zero_division=0),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),
            accuracy=accuracy_score(y,pred),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pred.mean()-y.mean()))
        for k,v in values.items():assert abs(float(r[k])-v)<1e-12


def main():
    binding,core,fac,records,rec,banks,old=setup();status=jr(PUB/'TRAIN_STATUS.json')
    assert status['status']=='ALL10_NEW_PLUS5_REUSED_MODELS_FROZEN' and status['body_updates']==0
    fr=jr(ROOT/'MODEL_FREEZE_PRIVATE.json');assert fr['input']==binding and len(fr['cells'])==15
    seal=jr(ROOT/'REPRESENTATION_SEAL_PRIVATE.json');assert seal['input']==binding
    for name,h in seal['patients'].items():assert sha(ROOT/name)==h
    assert len(seal['patients'])==320
    actual=pd.read_csv(PUB/'DEVELOPMENT_CELLS.csv');rows=[];checks=[];inputs=store();maximum=0.
    for arm in ARMS:
        for fold in range(1,6):
            d=bank(fold,banks);va=d['val'];ss=scale(fold);bankq=cached(fold,d)
            original,bodyr,bodystate,bodyq=body(old,fold,core,fac,records);before=state_hash(original.state_dict())
            r=next(z for z in fr['cells'] if z['arm']==arm and z['fold']==fold)
            assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
            assert set(d['patient']).isdisjoint(d['roles']['test'])
            if arm==ARMS[0]:
                state=bodystate;q=bodyq
                out=previous.predict(original,d,va,inputs,ss)
            else:
                assert r['body_checkpoint_sha256']==bodyr['checkpoint_sha256'] and sha(r['body_checkpoint'])==r['body_checkpoint_sha256']
                state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);q=read(r['validation'])
                assert q['binding']==state['binding'] and q['binding']['input']==binding
                assert q['binding']['representation_seal']==sha(ROOT/'REPRESENTATION_SEAL_PRIVATE.json')
                head=make(arm,linear(bankq),fold);assert state_hash(head.state_dict())==q['binding']['initial'];head.load_state_dict(state['model'])
                assert state_hash(head.readout.state_dict())==q['binding']['readout']==state_hash(original.head[2].state_dict())
                cached_out=predict(head,bankq['hidden'],d,va);full=MODEL.Fused(original,head).cuda().eval()
                assert sum(isinstance(z,torch.nn.Linear) and z.out_features==1 for z in full.modules())==1
                out=previous.predict(full,d,va,inputs,ss)
                assert np.max(np.abs(out-cached_out))<=1e-6
                grid=np.round(np.arange(0,1.0001,.005),3)
                assert np.array_equal(probability(out)[:,None]<grid,probability(cached_out)[:,None]<grid)
                del head,full
            assert state_hash(original.state_dict())==before
            assert np.array_equal(q['archive_indices'],d['archive_indices'][va])
            error=float(np.max(np.abs(out-q['logit'])));assert error<=1e-6
            grid=np.round(np.arange(0,1.0001,.005),3)
            assert np.array_equal(probability(out)[:,None]<grid,probability(q['logit'])[:,None]<grid)
            selected=core.select_threshold(d['y'][va],probability(out),d['patient'][va])
            for k,v in selected.items():assert abs(float(v)-float(state[k]))<1e-12
            part=patient_rows(d,va,out,state['threshold'],arm,fold);check_metrics(d,va,out,state['threshold'],part);rows+=part
            pub=actual[(actual.arm==arm)&(actual.fold==fold)].iloc[0]
            assert pub.checkpoint_sha256==r['checkpoint_sha256'] and pub.selected_epoch==state['epoch'] and abs(pub.threshold-state['threshold'])<1e-12
            for k in METRICS:assert abs(float(pub[k])-float(np.mean([z[k] for z in part])))<1e-12
            maximum=max(maximum,error)
            checks.append(dict(arm=arm,fold=fold,maximum_replay_error=error,all201_threshold_decisions_exact=True,epoch=int(state['epoch']),body_readout_unchanged=True,actual_Raw_end_to_end=True))
            del original;gc.collect();torch.cuda.empty_cache()
            print('CONTEXT_AFFINE_INDEPENDENT_CELL_PASS',arm,fold,flush=True)
    f=pd.DataFrame(rows);stored=pd.DataFrame(read(ROOT/'DEVELOPMENT_PATIENTS_PRIVATE.pkl'));keys=['arm','fold','patient']
    a,b=f.sort_values(keys).reset_index(drop=True),stored.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    selection=choose(f);candidate=jr(ROOT/'CANDIDATE_PRIVATE.json')
    assert candidate['selection']==selection and candidate['model_freeze_sha256']==sha(ROOT/'MODEL_FREEZE_PRIVATE.json')
    f.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/'DEVELOPMENT_MAIN.csv',index=False)
    ff=f.groupby(['arm','fold'])[METRICS].mean().reset_index();ff.to_csv(PUB/'DEVELOPMENT_FOLD.csv',index=False)
    ff.groupby('arm')[METRICS].std(ddof=1).reset_index().to_csv(PUB/'DEVELOPMENT_FOLD_SD.csv',index=False)
    sources=[];coverage=[];intervals=[];checks_ci=0;error_ci=0.;groups=[]
    for arm,g in f.groupby('arm'):
        assert len(g)==65 and g.patient.nunique()==47;groups.append((dict(kind='absolute',left=arm,right=''),g))
        for center,z in g.groupby('center'):
            n=z.patient.nunique();sources.append(dict(arm=arm,center=center,appearances=len(z),unique_patients=n,suppressed=n<5,
                **{k:float(z[k].mean()) if n>=5 else np.nan for k in METRICS}))
        coverage.extend([dict(arm=arm,metric=k,finite_appearances=int(np.isfinite(g[k]).sum()),total_appearances=len(g),
            unique_patients=g.patient.nunique()) for k in METRICS])
    for left,right in [(ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[2],ARMS[1])]:
        q=f[f.arm==left].merge(f[f.arm==right],on=['fold','patient','center'],suffixes=('_l','_r'),validate='one_to_one');assert len(q)==65
        diff=pd.DataFrame(dict(patient=q.patient,**{k:q[k+'_l']-q[k+'_r'] for k in METRICS}))
        groups.append((dict(kind='paired',left=left,right=right),diff))
    for identity,g in groups:
        rr=cluster(g.to_dict('records'));ids=sorted(g.patient.unique())
        sums=np.stack([g[g.patient==p][METRICS].sum().to_numpy(float) for p in ids]);n=np.array([int((g.patient==p).sum()) for p in ids])
        draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
        samples=sums[draw].sum(1)/n[draw].sum(1)[:,None];ci=np.quantile(samples,[.025,.975],axis=0)
        for j,row in enumerate(rr):
            err=max(abs(row['mean']-float(g[METRICS[j]].mean())),abs(row['low']-ci[0,j]),abs(row['high']-ci[1,j]));assert err<1e-12
            error_ci=max(error_ci,err);checks_ci+=1;intervals.append(dict(**identity,**row))
    assert len(f)==195 and checks_ci==54
    for name,data in [('DEVELOPMENT_SOURCE',sources),('METRIC_COVERAGE',coverage),('DEVELOPMENT_INTERVALS',intervals)]:
        pd.DataFrame(data).to_csv(PUB/(name+'.csv'),index=False)
    assert len(pd.read_csv(PUB/'TRAIN_HISTORY.csv'))==status['epochs']==int(actual.epochs.sum())
    for name,h in binding['source'].items():assert sha(HERE/name)==h
    assert sha(BASE_SEAL)==binding['protected_Outer_seal']
    write(PUB/'CANDIDATE_SELECTION.json',dict(status='SELECTED_BEFORE_ANY_NEW_OUTER',**selection,
        model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),candidate_sha256=sha(ROOT/'CANDIDATE_PRIVATE.json'),new_Outer_used=False))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checkpoint_checks=checks,maximum_replay_error=maximum,
        patient_metric_rows=len(f),independent_metric_checks=len(f)*9,independent_interval_checks=checks_ci,
        maximum_interval_difference=error_ci,all7_features_unchanged=True,new_Outer_passes=0))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_CONTEXT_AFFINE_INTERNAL',actual_new_fits=10,reused_controls=5,
        epochs=status['epochs'],selected=selection['selected'],final_stage_warranted=selection['new_qualified'],
        new_Outer_passes=0,goal_established=False,independent_confirmation=False))
    print('CONTEXT_AFFINE_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
