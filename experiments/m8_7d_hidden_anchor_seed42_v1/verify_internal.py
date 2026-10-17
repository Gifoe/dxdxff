"""Independently replay all25cells, warm transfer/frozen tensors, metrics and CIs."""
import gc
import itertools
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from train import initialize,evaluate
from admit import selected_state


def metrics(d,ix,out,theta,part):
    for row in part:
        j=d['patient'][ix]==row['patient'];y=1-d['y'][ix][j];pn=probability(out[j]);p=1-pn.astype(float);pred=pn<theta;ch=d['channel'][ix][j]
        order=sorted(range(len(y)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if y[i])
        values=dict(macro_f1=f1_score(y,pred,average='macro',labels=[0,1],zero_division=0),f1=f1_score(y,pred,zero_division=0),nez_f1=f1_score(1-y,1-pred.astype(int),zero_division=0),
            auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),accuracy=accuracy_score(y,pred),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pred.mean()-y.mean()))
        for m,v in values.items():assert abs(float(row[m])-v)<1e-12


def main():
    binding,core,fac,records,rec,banks,old=setup();status=jr(PUB/'TRAIN_STATUS.json');assert status['status']=='ALL20_NEW_PLUS5_REUSED_HIDDEN_ANCHOR_FROZEN'
    fr=jr(ROOT/'MODEL_FREEZE_PRIVATE.json');assert fr['input']==binding and len(fr['cells'])==25
    actual=pd.read_csv(PUB/'DEVELOPMENT_CELLS.csv');inputs=store();rows=[];checks=[];maximum=0.;random={}
    for arm in ARMS:
        for fold in range(1,6):
            d=bank(fold,banks);va=d['val'];ss=scale(fold);r=next(r for r in fr['cells'] if r['arm']==arm and r['fold']==fold)
            assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
            if arm=='C2_REFERENCE':_,state,_,out,error=reference(old,fold,core,fac,records,d,inputs,ss)
            else:
                q=read(r['validation']);state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
                assert state['binding']==q['binding'] and q['binding']['input']==binding and q['binding']['arm']==arm and state['epoch']>=1
                assert np.array_equal(q['archive_indices'],d['archive_indices'][va])
                warm=ROOT/'models_private/P_WARM_ONLY'/f'fold{fold}'/'BEST_PRIVATE.pt' if arm.startswith(('H1','H2')) else None
                m,random_hash,warm_sha=initialize(core,fac,records,fold,arm,warm)
                assert state_hash(m.state_dict())==q['binding']['initial'] and random_hash==q['binding']['random_initializer'] and warm_sha==q['binding']['warm_checkpoint_sha256']
                random.setdefault(fold,[]).append(random_hash)
                assert state_hash(selected_state(m,tuple(q['unchanged_prefixes'])))==q['unchanged_state_sha256']
                m.load_state_dict(state['model']);assert state_hash(selected_state(m,tuple(q['unchanged_prefixes'])))==q['unchanged_state_sha256']
                out=evaluate(m,d,va,inputs,ss,state['binding']['phase']);error=float(np.max(np.abs(out-q['logit'])));assert error<=1e-6
                grid=np.round(np.arange(0,1.0001,.005),3);assert np.array_equal(probability(out)[:,None]<grid,probability(q['logit'])[:,None]<grid)
                if arm=='P_WARM_ONLY':assert torch.count_nonzero(m.raw_projection.weight)==0
                if arm=='H2_WARM_FIXED':assert sum(p.numel() for p in m.parameters() if p.requires_grad)==12424
                del m;gc.collect();torch.cuda.empty_cache()
            sel=core.select_threshold(d['y'][va],probability(out),d['patient'][va]);assert all(abs(float(v)-float(state[k]))<1e-12 for k,v in sel.items())
            part=patient_rows(d,va,out,state['threshold'],arm,fold);metrics(d,va,out,state['threshold'],part);rows+=part
            pub=actual[(actual.arm==arm)&(actual.fold==fold)].iloc[0];assert pub.checkpoint_sha256==r['checkpoint_sha256'] and pub.selected_epoch==state['epoch'] and abs(pub.threshold-state['threshold'])<1e-12
            for m in METRICS:assert abs(float(pub[m])-float(np.mean([p[m] for p in part])))<1e-12
            maximum=max(maximum,error);checks.append(dict(arm=arm,fold=fold,maximum_replay_error=error,all201_decisions_exact=True,epoch=state['epoch'],transfer_and_frozen_state_verified=True))
            print('HIDDEN_ANCHOR_INDEPENDENT_CELL_PASS',arm,fold,flush=True)
    assert all(len(z)==4 and len(set(z))==1 for z in random.values())
    f=pd.DataFrame(rows);stored=pd.DataFrame(read(ROOT/'DEVELOPMENT_PATIENTS_PRIVATE.pkl'));keys=['arm','fold','patient']
    a,b=f.sort_values(keys).reset_index(drop=True),stored.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and np.max(np.abs(a[METRICS].to_numpy()-b[METRICS].to_numpy()))<1e-12
    selection=choose(f);candidate=jr(ROOT/'CANDIDATE_PRIVATE.json');assert candidate['selection']==selection and candidate['model_freeze_sha256']==sha(ROOT/'MODEL_FREEZE_PRIVATE.json')
    f.groupby('arm')[METRICS].mean().reset_index().to_csv(PUB/'DEVELOPMENT_MAIN.csv',index=False)
    ff=f.groupby(['arm','fold'])[METRICS].mean().reset_index();ff.to_csv(PUB/'DEVELOPMENT_FOLD.csv',index=False)
    ff.groupby('arm')[METRICS].std(ddof=1).reset_index().to_csv(PUB/'DEVELOPMENT_FOLD_SD.csv',index=False)
    sources=[];coverage=[];groups=[]
    for arm,g in f.groupby('arm'):
        assert len(g)==65 and g.patient.nunique()==47;groups.append((dict(kind='absolute',left=arm,right=''),g))
        for center,z in g.groupby('center'):
            n=z.patient.nunique();sources.append(dict(arm=arm,center=center,appearances=len(z),unique_patients=n,suppressed=n<5,**{m:float(z[m].mean()) if n>=5 else np.nan for m in METRICS}))
        coverage.extend(dict(arm=arm,metric=m,finite_appearances=int(np.isfinite(g[m]).sum()),total_appearances=len(g),unique_patients=g.patient.nunique()) for m in METRICS)
    for left,right in itertools.combinations(ARMS,2):
        # Report the later registered arm minus the earlier one, all ten pairs.
        left,right=right,left;q=f[f.arm==left].merge(f[f.arm==right],on=['fold','patient','center'],suffixes=('_l','_r'),validate='one_to_one');assert len(q)==65
        groups.append((dict(kind='paired',left=left,right=right),pd.DataFrame(dict(patient=q.patient,**{m:q[m+'_l']-q[m+'_r'] for m in METRICS}))))
    intervals=[];ci_checks=0;ci_error=0.
    for identity,g in groups:
        rr=cluster(g.to_dict('records'));ids=sorted(g.patient.unique());sums=np.stack([g[g.patient==p][METRICS].sum().to_numpy(float) for p in ids]);n=np.array([int((g.patient==p).sum()) for p in ids]);draw=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
        ci=np.quantile(sums[draw].sum(1)/n[draw].sum(1)[:,None],[.025,.975],axis=0)
        for j,row in enumerate(rr):
            error=max(abs(row['mean']-float(g[METRICS[j]].mean())),abs(row['low']-ci[0,j]),abs(row['high']-ci[1,j]));assert error<1e-12
            ci_error=max(ci_error,error);ci_checks+=1;intervals.append(dict(**identity,**row))
    assert ci_checks==135
    for name,data in [('DEVELOPMENT_SOURCE',sources),('METRIC_COVERAGE',coverage),('DEVELOPMENT_INTERVALS',intervals)]:pd.DataFrame(data).to_csv(PUB/(name+'.csv'),index=False)
    hist=pd.read_csv(PUB/'TRAIN_HISTORY.csv');assert len(hist)==status['epochs']==int(actual.epochs.sum())
    assert int((hist.phase=='phys').sum())==status['warm_epochs'] and int((hist.phase=='joint').sum())==status['joint_epochs']
    write(PUB/'CANDIDATE_SELECTION.json',dict(status='SELECTED_BEFORE_ANY_NEW_OUTER',**selection,new_Outer_used=False,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),candidate_sha256=sha(ROOT/'CANDIDATE_PRIVATE.json')))
    write(PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json',dict(status='PASS',checkpoint_checks=checks,maximum_replay_error=maximum,patient_metric_rows=len(f),independent_metric_checks=len(f)*9,
        independent_interval_checks=ci_checks,maximum_interval_difference=ci_error,all7_features_unchanged=True,all_stage_transfer_and_frozen_states_verified=True,new_Outer_passes=0))
    write(PUB/'INTERNAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_HIDDEN_ANCHOR_INTERNAL',actual_new_fits=20,warm_fits=5,joint_fits=15,reused_controls=5,warm_epochs=status['warm_epochs'],joint_epochs=status['joint_epochs'],
        epochs=status['epochs'],selected=selection['selected'],final_stage_warranted=selection['new_qualified'],new_Outer_passes=0,goal_established=False,independent_confirmation=False))
    print('HIDDEN_ANCHOR_INTERNAL_VERIFIED_COMPLETE',flush=True)


if __name__=='__main__':main()
