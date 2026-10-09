"""Frozen selected development artifacts, paired cluster draws and diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from base import sha,json_write,state_hash,patient_metrics,METRICS,select_threshold
from prepare import binding


def aggregate(frame,by):
    metrics=frame.groupby(by,sort=True)[METRICS].mean().reset_index()
    counts=frame.groupby(by).agg(patient_fold_cells=('patient','size'),unique_patients=('patient','nunique')).reset_index()
    return metrics.merge(counts,on=by,validate='one_to_one')


def bootstrap(patients):
    ids=sorted(patients.patient.unique()); draws=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    rows=[]; distributions=[]
    for left,right in [('W1','A0'),('W2','A0'),('W2','W1')]:
        pair=patients[patients.arm==left].merge(patients[patients.arm==right],on=['patient','fold','center'],suffixes=('_l','_r'),validate='one_to_one')
        assert len(pair)==65
        for metric in METRICS:
            diff=pair[metric+'_l'].to_numpy()-pair[metric+'_r'].to_numpy()
            temp=pd.DataFrame({'patient':pair.patient,'difference':diff}).groupby('patient').difference.agg(['sum','count']).reindex(ids)
            samples=temp['sum'].to_numpy()[draws].sum(1)/temp['count'].to_numpy()[draws].sum(1)
            lo,hi=np.quantile(samples,[.025,.975])
            rows.append({'comparison':left+'-'+right,'metric':metric,'delta':float(np.nanmean(diff)),
                'ci_low':float(lo),'ci_high':float(hi),'bootstrap_fraction_positive':float((samples>0).mean()),
                'draws':10000,'seed':42,'cluster_patients':47,'paired_patient_fold_cells':65})
            for scope,values in [('patient_fold_cells',diff),('unique_patient_mean',temp['sum'].to_numpy()/temp['count'].to_numpy())]:
                distributions.append({'comparison':left+'-'+right,'metric':metric,'scope':scope,'units':len(values),
                    'mean':float(np.nanmean(values)),'q05':float(np.nanquantile(values,.05)),
                    'q25':float(np.nanquantile(values,.25)),'median':float(np.nanmedian(values)),
                    'q75':float(np.nanquantile(values,.75)),'q95':float(np.nanquantile(values,.95)),
                    'fraction_improved':float((values>0).mean()),'fraction_tied':float((values==0).mean()),'fraction_worse':float((values<0).mean())})
    return pd.DataFrame(rows),pd.DataFrame(distributions)


def verify(root,protocol):
    pub=root/'public'; status=json.loads((pub/'RUN_STATUS.json').read_text())
    assert status['status']=='ALL_10_STUDENTS_COMPLETE' and status['binding']==binding(protocol)
    patients=[]; channels=[]; checks=[]
    for fold in range(1,6):
        path=root/f'fold{fold}/DEVELOPMENT_PRIVATE.pt'; d=torch.load(path,weights_only=False)
        assert d['binding']==binding(protocol)
        base=pd.read_csv(root/f'fold{fold}/A0_CHANNEL_PRIVATE.csv'); channels.append(base)
        pm=patient_metrics(base.y_nez.to_numpy(),base.score_nez.to_numpy(),base.patient.to_numpy(),base.center.to_numpy(),float(base.threshold.iloc[0]))
        pm['arm']='A0'; pm['fold']=fold; patients.append(pm)
        for arm in ['W1','W2']:
            cell=root/f'fold{fold}'/arm; meta=json.loads((cell/'COMPLETE_PRIVATE.json').read_text())
            assert sha(cell/'LAST_PRIVATE.pt')==meta['last_sha256'] and sha(cell/'BEST_PRIVATE.pt')==meta['best_sha256']
            last=torch.load(cell/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            best=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert last['binding']==best['binding']==meta['binding']
            assert json.loads(meta['binding']['source'])=={**binding(protocol),'sealed':sha(path)}
            assert state_hash(last['best']['model'])==state_hash(best['model'])
            assert state_hash(torch.load(cell/'INITIAL_PRIVATE.pt',weights_only=False))==meta['binding']['initial_sha256']
            assert len(last['history'])==last['epoch'] and 'rng' in last and 'optimizer' in last
            c=pd.read_csv(cell/'PREDICTIONS_PRIVATE.csv')
            pair=c.merge(base,on=['patient','channel','center','y_nez','fold'],validate='one_to_one',suffixes=('_new','_base'))
            assert len(pair)==len(base)==len(c) and np.isfinite(c.score_nez).all() and c.score_nez.between(0,1).all()
            assert (c.threshold==meta['threshold']).all()
            selected=select_threshold(c.y_nez.to_numpy(),c.score_nez.to_numpy(),c.patient.to_numpy())
            assert selected['threshold']==meta['threshold']
            assert abs(selected['macro_f1']-best['macro_f1'])<1e-12
            if arm=='W2':assert np.allclose(c[[f'member{k}_p_nez' for k in range(4)]].mean(axis=1),c.score_nez,rtol=0,atol=1e-7)
            m=patient_metrics(c.y_nez.to_numpy(),c.score_nez.to_numpy(),c.patient.to_numpy(),c.center.to_numpy(),meta['threshold'])
            old=pd.read_csv(cell/'PATIENT_PRIVATE.csv'); joined=m.merge(old,on=['patient','center'],suffixes=('_n','_o'),validate='one_to_one')
            assert all(np.allclose(joined[k+'_n'],joined[k+'_o'],rtol=0,atol=1e-12,equal_nan=True) for k in METRICS)
            m['arm']=arm; m['fold']=fold; patients.append(m); channels.append(c)
            checks.append({'arm':arm,'fold':fold,'selected_epoch':meta['selected_epoch'],'threshold':meta['threshold'],
                'checkpoint_sha256':meta['best_sha256'],'completed_epochs':last['epoch'],'source_input_initial_bound':True,
                'private_prediction_sha256':sha(cell/'PREDICTIONS_PRIVATE.csv')})
    patients=pd.concat(patients,ignore_index=True); channels=pd.concat(channels,ignore_index=True)
    assert patients.patient.nunique()==47 and patients.groupby('arm').size().eq(65).all()
    json_write(pub/'ARTIFACT_INTEGRITY_AUDIT.json',{'status':'PASS','cells':checks,'outer_evaluation':False,'selected_predictions_metrics_hashes_verified':True})
    return patients,channels


def diagnostics(channels,pub):
    decisions=[]; rankrows=[]; fixed=[]; base=channels[channels.arm=='A0']
    for arm in ['W1','W2']:
        c=channels[channels.arm==arm]
        pair=c.merge(base,on=['patient','channel','center','fold','y_nez'],suffixes=('_new','_base'),validate='one_to_one')
        for scope,groups in [('overall',[('ALL',pair)]),('fold',list(pair.groupby('fold'))),('center',list(pair.groupby('center')))]:
            for unit,g in groups:
                y=g.y_nez.to_numpy(); original=g.score_nez_base.to_numpy()>=g.threshold_base.to_numpy()
                for mode,tau in [('own_selected',g.threshold_new.to_numpy()),('fixed_A0',g.threshold_base.to_numpy())]:
                    new=g.score_nez_new.to_numpy()>=tau; correct_old=original==y; correct_new=new==y
                    oldtp=(y==0)&~original; newtp=(y==0)&~new; oldfp=(y==1)&~original; newfp=(y==1)&~new
                    decisions.append({'arm':arm,'scope':scope,'unit':str(unit),'threshold':mode,'channels':len(g),
                        'corrected':int((~correct_old&correct_new).sum()),'spoiled':int((correct_old&~correct_new).sum()),
                        'net_corrected':int(correct_new.sum()-correct_old.sum()),'EZ_TP_gained':int((newtp&~oldtp).sum()),
                        'EZ_TP_lost':int((oldtp&~newtp).sum()),'FP_added':int((newfp&~oldfp).sum()),'FP_removed':int((oldfp&~newfp).sum())})
        for (fold,pid),g in pair.groupby(['fold','patient']):
            old=g.score_nez_base.to_numpy(); new=g.score_nez_new.to_numpy(); shift=new-old
            i,j=np.triu_indices(len(g),1); b=np.sign(old[i]-old[j]); a=np.sign(new[i]-new[j])
            rankrows.append({'arm':arm,'fold':fold,'patient':pid,'center':str(g.center.iloc[0]),
                'mean_probability_shift':float(shift.mean()),'within_patient_shift_std':float(shift.std()),
                'mean_absolute_shift':float(np.abs(shift).mean()),
                'correlation_with_A0':float(np.corrcoef(old,new)[0,1]) if old.std()>1e-12 and new.std()>1e-12 else np.nan,
                'pair_order_relation_changed_fraction':float((a!=b).mean()),'strict_rank_reversal_fraction':float((a*b<0).mean())})
        for fold,g in c.groupby('fold'):
            pm=patient_metrics(g.y_nez.to_numpy(),g.score_nez.to_numpy(),g.patient.to_numpy(),g.center.to_numpy(),float(g.A0_threshold.iloc[0]))
            pm['arm']=arm; pm['fold']=fold; fixed.append(pm)
    pd.DataFrame(decisions).to_csv(pub/'ERROR_CORRECTION_AUDIT.csv',index=False)
    rf=pd.DataFrame(rankrows); values=[k for k in rf if k not in ['arm','fold','patient','center']]
    rows=[]
    for by,scope in [(['arm'],'overall'),(['arm','fold'],'fold'),(['arm','center'],'center')]:
        r=rf.groupby(by)[values].mean().reset_index(); r['scope']=scope
        if len(by)>1:r=r.rename(columns={by[1]:'unit'})
        else:r['unit']='ALL'
        rows.append(r)
    pd.concat(rows,ignore_index=True).to_csv(pub/'RANKING_SCORE_AUDIT.csv',index=False)
    return pd.concat(fixed,ignore_index=True)


def members(channels,patients,pub):
    c=channels[channels.arm=='W2']; metrics=[]; diversity=[]
    def with_metadata(m,fold,member,mode,threshold):
        # One concat avoids repeated native BlockManager.insert calls. Metrics
        # and row order are unchanged; this is only diagnostic table assembly.
        m=m.drop(columns=['fold'],errors='ignore').reset_index(drop=True)
        metadata=pd.DataFrame({'fold':[fold]*len(m),'member':[member]*len(m),
            'threshold_mode':[mode]*len(m),'threshold':[threshold]*len(m)})
        return pd.concat([m,metadata],axis=1)
    for fold,g in c.groupby('fold'):
        y=g.y_nez.to_numpy(); pid=g.patient.to_numpy(); center=g.center.to_numpy(); tau=float(g.threshold.iloc[0])
        for k in range(4):
            score=g[f'member{k}_p_nez'].to_numpy()
            for mode,t in [('parent_ensemble_threshold',tau),('posthoc_member_threshold_diagnostic',select_threshold(y,score,pid)['threshold'])]:
                m=patient_metrics(y,score,pid,center,t)
                metrics.append(with_metadata(m,fold,str(k),mode,t))
        m=patients[(patients.arm=='W2')&(patients.fold==fold)].copy()
        metrics.append(with_metadata(m,fold,'ensemble','parent_ensemble_threshold',tau))
        for p,gp in g.groupby('patient'):
            probs=gp[[f'member{k}_p_nez' for k in range(4)]].to_numpy(); binary=probs>=tau
            pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
            row={'fold':fold,'patient':p,'center':str(gp.center.iloc[0]),'member_probability_variance':float(probs.var(axis=1).mean()),
                'mean_pair_binary_disagreement':float(np.mean([(binary[:,i]!=binary[:,j]).mean() for i,j in pairs]))}
            for i,j in pairs:row[f'correlation_{i}_{j}']=float(np.corrcoef(probs[:,i],probs[:,j])[0,1]) if probs[:,i].std()>1e-12 and probs[:,j].std()>1e-12 else np.nan
            diversity.append(row)
    frame=pd.concat(metrics,ignore_index=True); rows=[]
    for by,scope in [(['member','threshold_mode'],'overall'),(['member','threshold_mode','fold'],'fold')]:
        r=aggregate(frame,by); r['scope']=scope; rows.append(r)
    pd.concat(rows,ignore_index=True).to_csv(pub/'MEMBER_DIVERSITY_SUMMARY.csv',index=False)
    dv=pd.DataFrame(diversity); vals=[k for k in dv if k not in ['fold','patient','center']]; dr=[]
    for by,scope in [([], 'overall'),(['fold'],'fold'),(['center'],'center')]:
        r=dv.groupby(by)[vals].mean().reset_index() if by else pd.DataFrame([dv[vals].mean().to_dict()])
        r['scope']=scope
        if by:r=r.rename(columns={by[0]:'unit'})
        else:r['unit']='ALL'
        dr.append(r)
    pd.concat(dr,ignore_index=True).to_csv(pub/'MEMBER_PAIR_DIVERSITY.csv',index=False)
    # Oracle here is diagnostic only; it cannot replace or prune any member.
    out=[]
    for mode in frame.threshold_mode.unique():
        individual=frame[(frame.member!='ensemble')&(frame.threshold_mode==mode)].groupby('member')[METRICS].mean()
        ens=patients[patients.arm=='W2'][METRICS].mean()
        for metric in METRICS:
            best=float(individual[metric].max()); out.append({'threshold_mode':mode,'metric':metric,'ensemble':float(ens[metric]),
                'best_single_member_diagnostic':best,'ensemble_minus_best':float(ens[metric]-best),'deployed_member_selection':False})
    pd.DataFrame(out).to_csv(pub/'ENSEMBLE_VS_BEST_MEMBER_DIAGNOSTIC.csv',index=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2); pub=a.runtime/'public'; patients,channels=verify(a.runtime,a.protocol)
    summary=aggregate(patients,['arm']); folds=aggregate(patients,['arm','fold']); centers=aggregate(patients,['arm','center']); boot,dist=bootstrap(patients)
    for name,frame in [('VALIDATION_SUMMARY',summary),('VALIDATION_BY_FOLD',folds),('VALIDATION_BY_CENTER',centers),('PAIRED_BOOTSTRAP',boot),('PATIENT_IMPROVEMENT_DISTRIBUTION',dist)]:frame.to_csv(pub/(name+'.csv'),index=False)
    fixed=diagnostics(channels,pub); members(channels,patients,pub)
    aggregate(pd.concat([fixed,patients[patients.arm=='A0']],ignore_index=True),['arm']).to_csv(pub/'FIXED_A0_THRESHOLD_DIAGNOSTIC.csv',index=False)
    worst=centers.loc[centers.groupby('arm').macro_f1.idxmin()]; worst.to_csv(pub/'WORST_CENTER_METRICS.csv',index=False)
    s=summary.set_index('arm'); delta=lambda metric,l,r:float(s.loc[l,metric]-s.loc[r,metric])
    f=folds.pivot(index='fold',columns='arm',values='macro_f1'); improve0=int((f.W2>f.A0).sum()); improve1=int((f.W2>f.W1).sum())
    conditions={'W2_A0_macro_ge_0_015':delta('macro_f1','W2','A0')>=.015,'W2_W1_macro_ge_0_005':delta('macro_f1','W2','W1')>=.005,
        'EZF1_nondecline':delta('ez_f1','W2','A0')>=0,'AP_nondecline':delta('ez_auprc','W2','A0')>=0,
        'four_of_five_A0':improve0>=4,'three_of_five_W1':improve1>=3,'implementation_and_isolation':True}
    passed=all(conditions.values()); target=delta('macro_f1','W2','A0')>=.03
    supported=all(float(boot[(boot.comparison==c)&(boot.metric=='macro_f1')].ci_low.iloc[0])>0 for c in ['W2-A0','W2-W1'])
    if passed and supported:terminal='TABM_TARGET_GAIN_SUPPORTED' if target else 'TABM_INCREMENTAL_GAIN_SUPPORTED'
    elif delta('macro_f1','W2','A0')>0 and delta('macro_f1','W2','W1')<=0 and delta('macro_f1','W1','A0')>0:terminal='CAPACITY_ONLY_GAIN'
    elif delta('macro_f1','W2','A0')>0:terminal='TABM_SMALL_INCONCLUSIVE_GAIN'
    else:terminal='TABM_NOT_SUPPORTED'
    json_write(pub/'VALIDATION_GATE.json',{'pass':passed,'terminal':terminal,'conditions':conditions,'failed_conditions':[k for k,v in conditions.items() if not v],
        'W2_A0_macro_delta':delta('macro_f1','W2','A0'),'W2_W1_macro_delta':delta('macro_f1','W2','W1'),
        'W2_A0_EZF1_delta':delta('ez_f1','W2','A0'),'W2_A0_AP_delta':delta('ez_auprc','W2','A0'),
        'W2_A0_improved_folds':improve0,'W2_W1_improved_folds':improve1,'target_gain_ge_0_030':target,
        'both_primary_delta_CI_lower_positive':supported,'no_outer_evaluation_authorized':True,'development_not_independent_confirmation':True})
    json_write(pub/'RUN_STATUS.json',{'status':'COMPLETE_DEVELOPMENT_GATE_'+('PASS' if passed else 'FAILED'),'terminal':terminal,
        'new_student_models':10,'A0_retrained':False,'outer_evaluation':False,'validation_appearances':65,'unique_patients':47,
        'bootstrap_draws':10000,'binding':binding(a.protocol),'finalizer_sha256':sha(Path(__file__))})
    print(summary.to_string(index=False),flush=True); print('TABM_AGGREGATION_COMPLETE',terminal,flush=True)


if __name__=='__main__':main()
