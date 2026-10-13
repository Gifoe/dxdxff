"""Frozen validation comparison and mechanism interventions; gated B3-only outer."""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint,patient_rows


def rank_change(p,q):
    ij=np.triu_indices(len(p),1)
    return float(np.mean(np.sign(p[ij[0]]-p[ij[1]])!=np.sign(q[ij[0]]-q[ij[1]]))) if len(ij[0]) else 0.


def mechanism_decision(summary,effects):
    t=summary.set_index('arm');base=t.loc['B3','ez_auroc']-t.loc['B0','ez_auroc'];capacity=t.loc['B3','ez_auroc']-t.loc['B2','ez_auroc']
    sh=effects[(effects.comparison=='B3 minus B3_SHUFFLED')&(effects.metric=='ez_auroc')].iloc[0]
    if base<=0:return 'PMA_PREDICTIVE_INCREMENT_NOT_SUPPORTED'
    if capacity<=0:return 'NOT_DISTINGUISHABLE_FROM_GENERIC_RESIDUAL_CAPACITY'
    if sh.delta<=0 or sh.ci_low<=0:return 'MAGNITUDE_SPECIFIC_EXPLANATION_UNSUPPORTED'
    return 'DEVELOPMENT_MAGNITUDE_CORRESPONDENCE_SIGNAL_EXPLORATORY'


def development(a,o):
    lock,ref,c,core,original,trainer,models,banks,binding=o;pub=a.runtime/'public'
    assert not (pub/'RUN_STATUS.json').exists(),'Completed development evaluation cannot rerun'
    freeze=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert freeze['binding']==binding
    fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json');pre=json.loads((pub/'PRETRAIN_AUDIT.json').read_text())
    allrows=[];intervention=[];usage=[];files={}
    for fold,d0 in enumerate(banks,1):
        bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
        d=torch.load(bp,map_location='cpu',weights_only=False);va=d['val'];x=torch.tensor(d['x'][va],device='cuda')
        for arm in lock['arms']:
            rec=next(r for r in freeze['selections'] if r['arm']==arm and r['fold']==fold)
            cp=checkpoint(a,arm,fold);assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256']
            ck=torch.load(cp,map_location='cpu',weights_only=False);tau=ck['threshold']
            if arm=='B0':
                path=bp.parent/'B0_VAL_PRIVATE.pt';assert c.sha(path)==pre['folds'][fold-1]['B0_validation_predictions_sha256']
                cell=torch.load(path,map_location='cpu',weights_only=False);scores=cell['scores']
            else:
                m=models.constructor(original.SourceA0,arm,d['stats_normalizer'])().cuda().eval();m.load_state_dict(ck['model'],strict=True)
                with torch.inference_mode():ll=m(x);scores=torch.sigmoid(ll).cpu().numpy()
                path=bp.parent/f'{arm}_VAL_PRIVATE.pt';assert not path.exists()
                core.torch_write(path,{'binding':fs,'indices':va,'scores':scores,'logits':ll.cpu().numpy(),'threshold':tau,'checkpoint_sha256':c.sha(cp)})
            rows=patient_rows(c,d,va,scores,tau,arm,fold);assert len(rows)==13
            assert abs(np.mean([r['macro_f1'] for r in rows])-ck['macro_f1'])<1e-12
            assert abs(np.mean([r['ez_f1'] for r in rows])-ck['ez_f1'])<1e-12
            allrows.extend(rows);files[f'{arm}_{fold}']=c.sha(path)
            if arm!='B3':continue
            with torch.inference_mode():
                full,h,delta=m.detail(x);s=m.normalized_stats(x)
            permutation=np.arange(len(va));rng=np.random.default_rng(42+1009*fold)
            for pat in sorted(set(d['patient'][va])):
                ix=np.flatnonzero(d['patient'][va]==pat);permutation[ix]=rng.permutation(ix)
            inv=np.argsort(permutation);shuffled=s[permutation]
            assert torch.equal(shuffled[inv],s)
            with torch.inference_mode():
                ls,hs,_=m.detail(x,stats_override=shuffled);ld,hd,_=m.detail(x,disable=True)
                base=m.network[4](h).squeeze(-1);delta_logit=torch.nn.functional.linear(delta,m.network[4].weight).squeeze(-1)
            assert torch.equal(h,hs) and torch.equal(h,hd) and torch.equal(ld,base)
            assert np.array_equal(scores,torch.sigmoid(full).cpu().numpy())
            data={'binding':fs,'indices':va,'permutation':permutation,'full_scores':scores,
                'shuffled_scores':torch.sigmoid(ls).cpu().numpy(),'disabled_scores':torch.sigmoid(ld).cpu().numpy(),
                'threshold':tau,'x_and_h_unchanged':True,'stats_multiset_preserved':True}
            core.torch_write(bp.parent/'MECHANISM_PRIVATE.pt',data)
            files[f'mechanism_{fold}']=c.sha(bp.parent/'MECHANISM_PRIVATE.pt')
            for variant,key in [('B3_SHUFFLED','shuffled_scores'),('B3_DISABLED','disabled_scores')]:
                intervention.extend(patient_rows(c,d,va,data[key],tau,variant,fold))
            hn=h.norm(dim=-1).cpu().numpy();dn=delta.norm(dim=-1).cpu().numpy();dl=delta_logit.cpu().numpy();bl=base.cpu().numpy();da=delta.cpu().numpy()
            for pat in sorted(set(d['patient'][va])):
                k=np.flatnonzero(d['patient'][va]==pat);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')]
                cor=float(np.corrcoef(bl[k],dl[k])[0,1]) if np.std(bl[k])>0 and np.std(dl[k])>0 else np.nan
                usage.append({'patient':pat,'fold':fold,'channels':len(k),'mean_hidden_norm':float(hn[k].mean()),
                    'mean_correction_norm':float(dn[k].mean()),'within_patient_correction_norm_std':float(dn[k].std()),
                    'correction_norm_q05':float(np.percentile(dn[k],5)),'correction_norm_median':float(np.median(dn[k])),
                    'correction_norm_q95':float(np.percentile(dn[k],95)),'mean_correction_to_hidden_ratio':float((dn[k]/np.maximum(hn[k],1e-12)).mean()),
                    'near_zero_fraction':float((dn[k]<1e-8).mean()),'delta_logit_mean':float(dl[k].mean()),
                    'residual_activation_mean':float(da[k].mean()),'residual_activation_std':float(da[k].std()),
                    'residual_activation_q05':float(np.percentile(da[k],5)),'residual_activation_q95':float(np.percentile(da[k],95)),
                    'delta_logit_std':float(dl[k].std()),'base_logit_delta_logit_correlation':cor,
                    'disabled_pair_rank_change_fraction':rank_change(scores[k],data['disabled_scores'][k]),
                    'shuffled_pair_rank_change_fraction':rank_change(scores[k],data['shuffled_scores'][k]),
                    'disabled_top1_change':float(np.argmin(scores[k])!=np.argmin(data['disabled_scores'][k])),
                    'shuffled_top1_change':float(np.argmin(scores[k])!=np.argmin(data['shuffled_scores'][k]))})
    df=pd.DataFrame(allrows);mi=pd.DataFrame(intervention);uu=pd.DataFrame(usage)
    assert len(df)==260 and df.patient.nunique()==47 and len(mi)==130 and len(uu)==65
    core.torch_write(a.runtime/'DEVELOPMENT_METRICS_PRIVATE.pt',allrows)
    core.torch_write(a.runtime/'MECHANISM_METRICS_PRIVATE.pt',intervention)
    core.torch_write(a.runtime/'USAGE_METRICS_PRIVATE.pt',usage)
    c.write_json(a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json',{'binding':fs,'files':files,'outer_accessed':False})
    summary=df.groupby('arm')[c.METRICS].mean().reset_index()
    summary['parameters']=summary.arm.map(lock['parameters']);summary['unit']='validation_selected_development_65_appearances_47_IDs'
    summary.to_csv(pub/'DEVELOPMENT_SUMMARY.csv',index=False)
    df.groupby(['arm','patient'])[c.METRICS].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
    for field,file in [('fold','DEVELOPMENT_BY_FOLD.csv'),('center','DEVELOPMENT_BY_CENTER.csv')]:
        rows=[{'arm':arm,field:v,'appearances':len(part),'unique_patients':part.patient.nunique(),**part[c.METRICS].mean().to_dict()} for (arm,v),part in df.groupby(['arm',field])]
        pd.DataFrame(rows).to_csv(pub/file,index=False)
    indexed={arm:df[df.arm==arm].set_index(['fold','patient']).sort_index() for arm in lock['arms']}
    effects=[]
    def add_comparison(left,right,left_frame,right_frame):
        assert np.array_equal(left_frame.index,right_frame.index)
        delta=left_frame[c.METRICS].to_numpy()-right_frame[c.METRICS].to_numpy();patient=np.asarray(left_frame.index.get_level_values('patient'))
        boot=c.cluster_bootstrap(patient,delta,10000)
        for j,key in enumerate(c.METRICS):effects.append({'comparison':left+' minus '+right,'metric':key,'delta':np.nanmean(delta[:,j]),
            'ci_low':np.nanpercentile(boot[:,j],2.5),'ci_high':np.nanpercentile(boot[:,j],97.5),'cluster':'unique47_IDs_retaining65_appearances',
            'draws':10000,'seed':42,'thresholds_reoptimized':False})
    for left,right in lock['comparisons']:add_comparison(left,right,indexed[left],indexed[right])
    pd.DataFrame(effects).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
    mech=[]
    for right in ['B3_SHUFFLED','B3_DISABLED']:
        rr=mi[mi.arm==right].set_index(['fold','patient']).sort_index();before=len(effects)
        add_comparison('B3',right,indexed['B3'],rr);mech.extend(effects[before:])
    mm=pd.DataFrame(mech);mm.to_csv(pub/'MAGNITUDE_SHUFFLE_CONTROL.csv',index=False)
    pd.concat([df[df.arm=='B3'],mi]).groupby('arm')[c.METRICS].mean().reset_index().to_csv(pub/'MECHANISM_SUMMARY.csv',index=False)
    usage_cols=[k for k in uu if k not in ['patient','fold','channels']]
    usage_out=[]
    for fold,part in [(0,uu)]+list(uu.groupby('fold')):
        usage_out.append({'fold':fold,'patient_appearances':len(part),'unique_patients':part.patient.nunique(),
            **part[usage_cols].mean().to_dict(),'between_patient_mean_correction_std':float(part.mean_correction_norm.std(ddof=0))})
    pd.DataFrame(usage_out).to_csv(pub/'PMA_CORRECTION_AUDIT.csv',index=False)
    t=summary.set_index('arm');folds=df.groupby(['arm','fold']).ez_auroc.mean().unstack('arm')
    wins=int((folds.B3>folds.B0).sum())
    criteria={
        'B3_minus_B0_AUROC_ge_0.010':bool(t.loc['B3','ez_auroc']-t.loc['B0','ez_auroc']>=.010),
        'B3_minus_B2_AUROC_ge_0.005':bool(t.loc['B3','ez_auroc']-t.loc['B2','ez_auroc']>=.005),
        'B3_AP_not_lower_than_B0':bool(t.loc['B3','ez_auprc']>=t.loc['B0','ez_auprc']),
        'B3_MacroF1_delta_ge_minus_0.005':bool(t.loc['B3','macro_f1']-t.loc['B0','macro_f1']>=-.005),
        'B3_EZF1_delta_ge_minus_0.005':bool(t.loc['B3','ez_f1']-t.loc['B0','ez_f1']>=-.005),
        'positive_AUROC_folds_ge4':wins>=4,
        'engineering_integrity_numerical_checks':True}
    assert setup(a)[-1]==binding
    c.write_json(pub/'VALIDATION_GATE.json',{'pass':all(criteria.values()),'criteria':criteria,'positive_AUROC_folds':wins,
        'failed_criteria':[k for k,v in criteria.items() if not v],'outer_authorized':all(criteria.values()),'mechanism_terminal':mechanism_decision(summary,mm)})
    c.write_json(pub/'RUN_STATUS.json',{'status':'DEVELOPMENT_AND_MECHANISM_COMPLETE','formal_models':4,'new_training_cells':15,
        'reused_B0_cells':5,'validation_appearances':65,'unique_validation_patients':47,'new_outer_accessed':False,
        'scientific_status':lock['scientific_status'],'outer_pending_gate':all(criteria.values())})
    print('PMA_DEVELOPMENT_AND_MECHANISM_COMPLETE',json.dumps(criteria),flush=True)


def outer(a,o):
    lock,ref,c,core,original,trainer,models,banks,binding=o;pub=a.runtime/'public'
    assert json.loads((pub/'VALIDATION_GATE.json').read_text())['pass'],'Gate failed: no outer access allowed'
    assert json.loads((pub/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
    assert not (pub/'OUTER_STATUS.json').exists()
    assert not (a.runtime/'FROZEN_BEFORE_OUTER.json').exists(),'Preserve existing outer attempt; inspect before any resume'
    frozen=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert frozen['binding']==binding
    c.write_json(a.runtime/'FROZEN_BEFORE_OUTER.json',{'binding':binding,'development_freeze_sha256':c.sha(a.runtime/'DEVELOPMENT_FREEZE.json'),
        'gate_sha256':c.sha(pub/'VALIDATION_GATE.json'),'verification_sha256':c.sha(pub/'INDEPENDENT_VERIFICATION.json'),
        'authorized_outer_models':['B3'],'B1_B2_outer_not_authorized':True,'selections':frozen['selections']})
    fs=c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json')
    with np.load(a.features,allow_pickle=True) as f:raw={k:f[k] for k in f.files if k!='y'}
    sys.path.insert(0,str(a.source/'d2_ictal_outer_seed42_v1/code'))
    from evaluate_outer import frozen_transform
    cells=[];files={};seen=[]
    # B3 only; the current source-free B0 outer predictions are reused afterwards.
    for fold in range(1,6):
        rec=next(r for r in frozen['selections'] if r['arm']=='B3' and r['fold']==fold);cp=checkpoint(a,'B3',fold)
        assert c.sha(cp)==rec['checkpoint_sha256'];d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        s=raw['split_fold']==fold;ids=raw['split_patient'][s&(raw['split_role']=='test')];ix=np.flatnonzero(np.isin(raw['patient'],ids))
        ck=torch.load(cp,map_location='cpu',weights_only=False);m=models.constructor(original.SourceA0,'B3',d['stats_normalizer'])().eval();m.load_state_dict(ck['model'])
        x=frozen_transform(raw['x'][ix],raw['patient'][ix],d['pre'],core)
        with torch.inference_mode():logits=m(torch.from_numpy(x));scores=torch.sigmoid(logits).numpy()
        path=a.runtime/f'fold{fold}/B3_OUTER_PRIVATE.pt';assert not path.exists()
        core.torch_write(path,{'binding':fs,'indices':ix,'scores':scores,'threshold':ck['threshold']});files[str(fold)]=c.sha(path);seen.extend(ix.tolist());cells.append((ix,scores,ck['threshold']))
    assert sorted(seen)==list(range(7635))
    c.write_json(a.runtime/'OUTER_PREDICTION_SEAL.json',{'binding':fs,'files':files,'all_B3_scores_before_y':True})
    with np.load(a.features,allow_pickle=True) as f:y=f['y']
    oldseal=json.loads((a.previous/'OUTER_PREDICTIONS_SEAL.json').read_text())['files'];rows=[]
    for fold,(ix,score,tau) in enumerate(cells,1):
        oldpath=a.previous/f'no_source/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt';assert c.sha(oldpath)==oldseal['no_source_'+str(fold)]
        old=torch.load(oldpath,map_location='cpu',weights_only=False);assert np.array_equal(ix,old['indices'])
        d={'patient':raw['patient'],'channel':raw['channel'],'center':raw['center'],'y':y}
        rows.extend(patient_rows(c,d,ix,score,tau,'B3',fold));rows.extend(patient_rows(c,d,ix,old['scores'],old['threshold'],'B0',fold))
    df=pd.DataFrame(rows);core.torch_write(a.runtime/'OUTER_METRICS_PRIVATE.pt',rows)
    df.groupby('arm')[c.METRICS].mean().reset_index().to_csv(pub/'OUTER_SUMMARY.csv',index=False)
    b3=df[df.arm=='B3'].set_index('patient').sort_index();b0=df[df.arm=='B0'].set_index('patient').sort_index()
    delta=b3[c.METRICS].to_numpy()-b0[c.METRICS].to_numpy();boot=c.cluster_bootstrap(b3.index,delta,10000)
    pd.DataFrame([{'metric':k,'delta':delta[:,j].mean(),'ci_low':np.percentile(boot[:,j],2.5),'ci_high':np.percentile(boot[:,j],97.5)} for j,k in enumerate(c.METRICS)]).to_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False)
    c.write_json(pub/'OUTER_STATUS.json',{'status':'ONE_GATED_B3_OUTER_COMPLETE','patients':80,'channels':7635,'B1_B2_outer_evaluated':False,
        'scientific_status':'EXPLORATORY_REPEATED_OUTER_ON_HISTORICALLY_VIEWED_COHORT'})


if __name__=='__main__':
    is_outer='--outer' in sys.argv
    if is_outer:sys.argv.remove('--outer')
    a=args();o=setup(a);(outer if is_outer else development)(a,o)
