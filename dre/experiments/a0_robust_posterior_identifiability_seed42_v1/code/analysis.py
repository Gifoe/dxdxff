"""FIT-first diagnostic phases; only admitted arms have a VAL inference path."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance,ks_2samp
from posterior import fit_floor,normalize,iqr,fit_density,infer,decode,calibration,DensityInvalid,log_densities
from common import metric,METRICS,write_json,digest

ARMS={'D1_R0':('A0','R0'),'D1_R1':('A0','R1'),'D3_R0':('D2','R0'),'D3_R1':('D2','R1')}
SOURCES=['HUP','LZU','multicenter','pediatric']

def frozen_inputs(bank,family,scope):
    d,full,of=bank; ix=d['train'] if scope=='FIT_OOF' else d['val']
    logits=(d['a0_oof'] if family=='A0' else of['logits']) if scope=='FIT_OOF' else (d['a0_logits'][ix] if family=='A0' else full['logits'][ix])
    return {'logits':np.asarray(logits,dtype=np.float64),'patient':d['patient'][ix],
        'g':d['g'][ix],'channel':d['channel'][ix],'y':d['y'][ix]}

def normalization_rows(data,rep,floor,fold,arm,scope,include_class=True):
    x=data['logits']; p=data['patient']; groups=data['g']; z=np.empty_like(x); eps=[]; clipped=np.zeros(len(x),bool)
    for pid in sorted(set(p)):
        ix=np.flatnonzero(p==pid); v=x[ix]; zz,r=normalize(v,rep,floor); z[ix]=zz
        mask=np.zeros(len(ix),bool) if rep=='R0' else np.abs((v-np.median(v))/max(r,floor))>8
        clipped[ix]=mask
        eps.append({'patient':pid,'g':int(groups[ix[0]]),'iqr':r,'median':float(np.median(v)),
            'floor':rep=='R1' and r<floor,'clip':float(mask.mean()),'constant':bool(np.ptp(v)==0),
            'raw_spread':float(np.ptp(v)),'max_abs_raw':float(np.abs(v).max()),'max_abs_z':float(np.abs(zz).max())})
    rows=[]
    for source in [-1,0,1,2,3]:
        take=np.ones(len(x),bool) if source==-1 else groups==source
        ee=[e for e in eps if source==-1 or e['g']==source]
        if not take.any():continue
        rows.append({'fold':fold,'arm':arm,'scope':scope,'source':'all' if source==-1 else SOURCES[source],
            'patients':len(ee),'channels':int(take.sum()),'FIT_IQR_floor':floor if rep=='R1' else None,
            'raw_IQR_min':min(e['iqr'] for e in ee),'raw_IQR_q10':float(np.quantile([e['iqr'] for e in ee],.10)),
            'raw_IQR_median':float(np.median([e['iqr'] for e in ee])),'raw_IQR_q90':float(np.quantile([e['iqr'] for e in ee],.90)),
            'low_IQR_fraction':float(np.mean([e['iqr']<=1e-5 for e in ee])),
            'floor_activation_fraction':float(np.mean([e['floor'] for e in ee])),
            'clip_fraction':float(clipped[take].mean()),'patient_clip_mean':float(np.mean([e['clip'] for e in ee])),
            'patient_clip_q90':float(np.quantile([e['clip'] for e in ee],.90)),'patient_clip_max':max(e['clip'] for e in ee),
            'constant_patients':sum(e['constant'] for e in ee),'raw_abs_max':float(np.abs(x[take]).max()),
            'transformed_abs_max':float(np.abs(z[take]).max()),
            'before_clip_abs_max':float(max(np.abs((x[p==e['patient']]-e['median'])/max(e['iqr'],floor)).max() for e in ee)) if rep=='R1' else float(np.abs(x[take]).max()),
            'raw_q01':float(np.quantile(x[take],.01)),'raw_q99':float(np.quantile(x[take],.99)),
            'z_q01':float(np.quantile(z[take],.01)),'z_q99':float(np.quantile(z[take],.99)),
            'nonfinite_scores':int((~np.isfinite(z[take])).sum())})
    tails=[]
    for cls in ([0,1] if include_class else []):
        classrows=[]
        for e in eps:
            v=z[(p==e['patient'])&(data['y']==cls)]
            if len(v):classrows.append({'low':e['iqr']<=1e-5,'first':float(v.mean()),'second':float(np.mean(v*v))})
        n=len(classrows)
        tails.append({'fold':fold,'arm':arm,'scope':scope,'class':['EZ','NEZ'][cls],'class_bearing_patients':n,
            'class_mean':np.mean([r['first'] for r in classrows]) if n else np.nan,
            'low_IQR_patient_count':sum(r['low'] for r in classrows),
            'low_IQR_first_moment_contribution':sum(r['first'] for r in classrows if r['low'])/n if n else np.nan,
            'low_IQR_second_moment_contribution':sum(r['second'] for r in classrows if r['low'])/n if n else np.nan,
            'max_single_patient_second_moment':max((r['second'] for r in classrows),default=np.nan),
            'R1_second_moment_theoretical_bound':64 if rep=='R1' else None})
    return z,eps,rows,tails

def safe_density(data,rep,floor):
    try:
        den=fit_density(data['logits'],data['y'],data['patient'],data['g'],set(data['patient']),rep,floor)
        vals=[den['global']]+den['sources']
        assert all(np.isfinite([v['mu_ez'],v['mu_nez'],v['variance'],v['pi']]).all() and v['variance']>0 and v['mu_ez']<v['mu_nez'] for v in vals)
        return den,True,'none'
    except DensityInvalid as e:return e.audit,False,'global_ordering_or_class_count_failed'

def diagnostics(logits,channels,source,density,rep,floor):
    q,m=infer(logits,source,density,rep,floor); qf,f=infer(logits,source,density,rep,floor,fixed=True)
    pred,dc=decode(q,channels,logits); fixed,df=decode(qf,channels,logits)
    order=np.lexsort((channels.astype(str),logits))
    assert np.all(np.diff(q[order])<=1e-13) and np.all(np.diff(qf[order])<=1e-13)
    assert np.isfinite(m['logodds']).all() and np.isfinite(m['ll_gain'])
    return q,qf,pred,fixed,m,dc

def fit_phase(banks,lock,pub):
    fits={}; normal=[]; floors=[]; tails=[]; density_rows=[]; parameters=[]; loo=[]; loo_audit=[]
    for arm,(family,rep) in ARMS.items():
        fits[arm]=[]
        for fold,bank in enumerate(banks,1):
            data=frozen_inputs(bank,family,'FIT_OOF'); ids=sorted(set(data['patient']))
            floor=None; floor_valid=True
            if rep=='R1':
                try:floor=fit_floor(data['logits'],data['patient'],ids)
                except ValueError:floor_valid=False
            if not floor_valid:
                fits[arm].append({'valid':False,'floor':None,'density':None,'reason':'no_positive_IQR'})
                density_rows.append({'fold':fold,'arm':arm,'valid':False,'global_ordered':False,'reason':'no_positive_IQR'})
                floors.append({'fold':fold,'family':family,'representation':rep,'FIT_patients':len(ids),'floor':None,
                    'positive_IQR_patients':0,'label_blind':True,'fit_scope':'outer_FIT_OOF_only'})
                loo_audit.append({'fold':fold,'arm':arm,'expected_pseudo_targets':len(ids),'valid_LOO_episodes':0,
                    'invalid_LOO_episodes':len(ids),'query_labels_excluded':True,'query_floor_excluded':True,
                    'upstream_fully_nested':False,'LOO_class_order_failure':0,'LOO_clip_quality_exceedances':0})
                continue
            z,eps,nr,tr=normalization_rows(data,rep,floor,fold,arm,'FIT_OOF'); normal+=nr; tails+=tr
            floors.append({'fold':fold,'family':family,'representation':rep,'FIT_patients':len(ids),'floor':floor,
                'positive_IQR_patients':sum(e['iqr']>0 for e in eps),'label_blind':True,'fit_scope':'outer_FIT_OOF_only'})
            den,ordered,reason=safe_density(data,rep,floor)
            clip_ok=rep=='R0' or (nr[0]['clip_fraction']<=lock['numerical_gate']['R1_total_clip_max'] and max(r['clip_fraction'] for r in nr[1:])<=lock['numerical_gate']['R1_source_clip_max'])
            finite=True; diagnostic=[]
            if ordered:
                for pid in ids:
                    ix=np.flatnonzero(data['patient']==pid)
                    q,qf,pe,pf,m,dc=diagnostics(data['logits'][ix],data['channel'][ix],int(data['g'][ix[0]]),den,rep,floor)
                    diagnostic.append({'extreme_q':float(np.mean((q<=1e-6)|(q>=1-1e-6))),'converged':m['converged'],
                        'boundary':m['near_boundary'],'fallback':m['fallback']})
                finite=all(r['converged'] for r in diagnostic)
            valid=bool(ordered and clip_ok and finite)
            if not clip_ok:reason='R1_clipping_quality_failed'
            if not finite:reason='MAP_nonconvergence'
            gm=den['global']; moments=den['global_moments']
            density_rows.append({'fold':fold,'arm':arm,'valid':valid,'reason':reason,'global_ordered':ordered,'clip_quality_pass':clip_ok,
                'FIT_patients':len(ids),'class_patients_EZ':moments['0']['n'],'class_patients_NEZ':moments['1']['n'],
                'mu_EZ':gm['mu_ez'],'mu_NEZ':gm['mu_nez'],'gap':gm['mu_nez']-gm['mu_ez'],'variance':gm['variance'],
                'standardized_separation':(gm['mu_nez']-gm['mu_ez'])/np.sqrt(gm['variance']),
                'extreme_posterior_fraction':np.mean([r['extreme_q'] for r in diagnostic]) if diagnostic else None,
                'MAP_convergence_fraction':np.mean([r['converged'] for r in diagnostic]) if diagnostic else None,
                'source_fallback_patient_fraction':np.mean([r['fallback'] for r in diagnostic]) if diagnostic else None,
                'finite_density_checks':bool(ordered),'low_IQR_fraction':nr[0]['low_IQR_fraction']})
            parameters.append({'fold':fold,'arm':arm,'valid':valid,'FIT_floor':floor,'density':den})
            fits[arm].append({'valid':valid,'floor':floor,'density':den,'reason':reason})
            counts={'fold':fold,'arm':arm,'expected_pseudo_targets':len(ids),'valid_LOO_episodes':0,'invalid_LOO_episodes':0,
                'query_labels_excluded':True,'query_floor_excluded':True,'upstream_fully_nested':False,
                'LOO_class_order_failure':0,'LOO_clip_quality_exceedances':0}
            # Invalid full densities are not applied. Valid but clipping-inadmissible
            # fits may have FIT-only diagnostics, never a full VAL arm.
            if ordered:
                for pid in ids:
                    take=data['patient']!=pid; qi=np.flatnonzero(~take)
                    train={k:v[take] for k,v in data.items()}; target_logits=data['logits'][qi]
                    local_floor=None
                    try:
                        if rep=='R1':local_floor=fit_floor(train['logits'],train['patient'],set(train['patient']))
                        ld,ok,why=safe_density(train,rep,local_floor)
                        if not ok:
                            counts['invalid_LOO_episodes']+=1; counts['LOO_class_order_failure']+=1; continue
                        if rep=='R1':
                            _,_,lnr,_=normalization_rows(train,rep,local_floor,fold,arm,'LOO_DENSITY_FIT',include_class=False)
                            if lnr[0]['clip_fraction']>.05 or max(r['clip_fraction'] for r in lnr[1:])>.10:
                                # The locked quality gate is the full outer-FIT
                                # check. LOO clipping is descriptive, not a new gate.
                                counts['LOO_clip_quality_exceedances']+=1
                        # No target y is passed to floor, density, MAP or decoder.
                        q,qf,pe,pf,m,dc=diagnostics(target_logits,data['channel'][qi],int(data['g'][qi[0]]),ld,rep,local_floor)
                        y=data['y'][qi] # Reveal only after both decisions are frozen.
                        mm=metric(y,target_logits,.5,pred=~pe); mf=metric(y,target_logits,.5,pred=~pf)
                        actual=float(np.mean(y==0)); counts['valid_LOO_episodes']+=1
                        loo.append({'fold':fold,'arm':arm,'patient':pid,'MAP_macro_f1':mm['macro_f1'],'fixed_macro_f1':mf['macro_f1'],
                            'MAP_ez_f1':mm['ez_f1'],'fixed_ez_f1':mf['ez_f1'],'ez_auprc':mm['ez_auprc'],'ez_auroc':mm['ez_auroc'],
                            'MAP_prevalence_MAE':abs(m['pi']-actual),'prior_prevalence_MAE':abs(m['prior']-actual),
                            'predicted_fraction_error':abs(pe.mean()-actual),'fixed_predicted_fraction_error':abs(pf.mean()-actual),
                            'MAP_prior_displacement':m['pi']-m['prior'],'abs_MAP_prior_displacement':abs(m['pi']-m['prior']),
                            'converged':m['converged'],'near_boundary':m['near_boundary'],'extreme_k':dc['k'] in [0,len(qi)],
                            'source_fallback':m['fallback'],'floor':local_floor})
                    except ValueError:
                        counts['invalid_LOO_episodes']+=1
            else:counts['invalid_LOO_episodes']=len(ids)
            loo_audit.append(counts)
            print('FIT_CHECK_AND_LOO',arm,fold,'numeric',valid,'LOO',counts['valid_LOO_episodes'],len(ids),flush=True)
    write_json(pub/'SOURCE_GAUSSIAN_PARAMETERS.json',{'fits':parameters,'fitting':'full outer FIT OOF only','family_unchanged':True})
    pd.DataFrame(normal).to_csv(pub/'ROBUST_NORMALIZATION_AUDIT.csv',index=False)
    pd.DataFrame(floors).to_csv(pub/'FIT_IQR_FLOOR_BY_FOLD.csv',index=False)
    pd.DataFrame(tails).to_csv(pub/'SCORE_TAIL_INFLUENCE_AUDIT.csv',index=False)
    pd.DataFrame(density_rows).to_csv(pub/'FIT_DENSITY_IDENTIFIABILITY.csv',index=False)
    pd.DataFrame(loo_audit).to_csv(pub/'PATIENT_LOO_DENSITY_AUDIT.csv',index=False)
    lf=pd.DataFrame(loo); utility=[]; gates={}
    for arm in ARMS:
        sub=lf[lf.arm==arm] if len(lf) else pd.DataFrame()
        expected=sum(r['expected_pseudo_targets'] for r in loo_audit if r['arm']==arm)
        full=len(sub)==expected and all(r['valid_LOO_episodes']==r['expected_pseudo_targets'] for r in loo_audit if r['arm']==arm)
        numeric=all(r['valid'] for r in fits[arm]) and len(fits[arm])==5
        fg=[]
        for fold in range(1,6):
            group=sub[sub.fold==fold] if len(sub) else pd.DataFrame()
            row={'arm':arm,'fold':fold,'scope':'FIT_LOO_SCREENING_NOT_NESTED','valid_cells':len(group),
                'expected_cells':next(r['expected_pseudo_targets'] for r in loo_audit if r['arm']==arm and r['fold']==fold),
                'full_group':len(group)==next(r['expected_pseudo_targets'] for r in loo_audit if r['arm']==arm and r['fold']==fold)}
            if len(group):row.update(group.drop(columns=['fold','arm','patient']).mean().to_dict())
            if len(group):row['MAP_minus_fixed_macro']=float((group.MAP_macro_f1-group.fixed_macro_f1).mean())
            utility.append(row); fg.append(row)
        mean=sub.drop(columns=['fold','arm','patient']).mean().to_dict() if len(sub) else {}
        delta=float((sub.MAP_macro_f1-sub.fixed_macro_f1).mean()) if len(sub) else None
        prevalence=mean.get('prior_prevalence_MAE',np.nan)-mean.get('MAP_prevalence_MAE',np.nan)
        severe=not full or any(r.get('near_boundary',1)>.20 or r.get('extreme_k',1)>.20 or r.get('converged',0)<1 for r in fg)
        conditions={'numeric_all_5':numeric,'all_LOO_episodes_valid':full,
            'MAP_macro_gain_at_least_0p005':full and delta>=.005,
            'at_least_3_positive_folds':full and sum(r.get('MAP_minus_fixed_macro',-np.inf)>0 for r in fg)>=3,
            'EZ_F1_nondecline':full and mean['MAP_ez_f1']>=mean['fixed_ez_f1'],
            'prevalence_MAE_gain_at_least_0p005':full and prevalence>=.005,'no_severe_collapse':not severe}
        gates[arm]={'admitted':all(conditions.values()),'numerically_valid':numeric,'conditions':conditions,
            'MAP_minus_fixed_macro':delta if full else None,'prior_MAE_minus_MAP_MAE':prevalence if full else None,
            'positive_folds':sum(r.get('MAP_minus_fixed_macro',-np.inf)>0 for r in fg) if full else None,
            'valid_LOO_cells':len(sub),'expected_LOO_cells':expected,'full_LOO_screening':full}
        total={'arm':arm,'fold':'all','scope':'FIT_LOO_SCREENING_NOT_NESTED','valid_cells':len(sub),'expected_cells':expected,'full_group':full}
        if full:total.update(mean); total.update(MAP_minus_fixed_macro=delta,prior_MAE_minus_MAP_MAE=prevalence)
        utility.append(total)
    pd.DataFrame(utility).to_csv(pub/'FIT_MAP_VS_FIXED_PRIOR.csv',index=False)
    write_json(pub/'FIT_POSTERIOR_UTILITY_GATE.json',{'arms':gates,'selected_using_VAL':False,'fully_nested_independence':False,
        'density_parameters_digest':digest(parameters),'admission_frozen_before_VAL':True})
    return fits,gates,loo

def validation_phase(banks,fits,gates,controls,pub):
    rows=list(controls); diag=[]; errors=[]; private=[]
    for arm,(family,rep) in ARMS.items():
        if not gates[arm]['admitted']:continue
        for fold,bank in enumerate(banks,1):
            data=frozen_inputs(bank,family,'VAL_FULL_FIT'); d,full,of=bank; frozen=fits[arm][fold-1]
            assert frozen['valid']; den=frozen['density']; floor=frozen['floor']
            for pid in sorted(set(data['patient'])):
                qi=np.flatnonzero(data['patient']==pid); qi=qi[np.argsort(data['channel'][qi].astype(str),kind='stable')]
                logits=data['logits'][qi]; channel=data['channel'][qi]; source=int(data['g'][qi[0]])
                q,qf,pe,pf,m,dc=diagnostics(logits,channel,source,den,rep,floor)
                # Only now are retrospective VAL labels revealed to evaluation.
                y=data['y'][qi]; global_ix=d['val'][qi]
                score=d['a0_scores'][global_ix] if family=='A0' else full['scores'][global_ix]
                for name,pred in [(arm,pe),(arm+'_FIXED_PRIOR',pf)]:
                    mm=metric(y,score,.5,pred=~pred)
                    base=metric(y,score,.5)
                    for k in ['ez_auprc','ez_auroc','ez_mrr','top1_is_ez']:assert np.isclose(mm[k],base[k],rtol=0,atol=1e-12,equal_nan=True)
                    rows.append({'fold':fold,'patient':pid,'center':str(d['center'][global_ix[0]]),'method':name,**mm})
                b,e=calibration(q,y); bf,ef=calibration(qf,y); actual=float(np.mean(y==0))
                diag.append({'fold':fold,'patient':pid,'method':arm,'brier':b,'ece':e,'fixed_brier':bf,'fixed_ece':ef,
                    'pi_MAE':abs(m['pi']-actual),'prior_MAE':abs(m['prior']-actual),'MAP_prior_displacement':m['pi']-m['prior'],
                    'predicted_fraction':float(pe.mean()),'actual_fraction':actual,'predicted_fraction_MAE':abs(pe.mean()-actual),
                    'near_boundary':m['near_boundary'],'extreme_k':dc['k'] in [0,len(qi)],'converged':m['converged']})
                basepred=d['a0_scores'][global_ix]>=d['A0_threshold']; new=~pe
                errors.append({'method':arm,'TP_gained':int(((y==0)&basepred&~new).sum()),'TP_lost':int(((y==0)&~basepred&new).sum()),
                    'FP_added':int(((y==1)&basepred&~new).sum()),'FP_removed':int(((y==1)&~basepred&new).sum()),
                    'errors_recovered':int(((basepred!=y)&(new==y)).sum()),'correct_spoiled':int(((basepred==y)&(new!=y)).sum())})
                private.append({'arm':arm,'fold':fold,'patient':pid,'channel':channel,'q':q,'fixed_q':qf,'predicted_EZ':pe,'fixed_EZ':pf})
    return rows,diag,errors,private

def transport(banks,fits,gates,pub):
    rows=[]
    for arm,(family,rep) in ARMS.items():
        for fold,bank in enumerate(banks,1):
            floor=fits[arm][fold-1]['floor']
            if rep=='R1' and floor is None:continue
            scopes={}
            for scope in ['FIT_OOF','VAL_FULL_FIT']:
                data=frozen_inputs(bank,family,scope)
                unlabeled={k:v for k,v in data.items() if k!='y'}
                z,eps,nr,_=normalization_rows(unlabeled,rep,floor,fold,arm,scope,include_class=False)
                scopes[scope]=(data,z,eps,nr)
            for group in [-1,0,1,2,3]:
                collect=[]
                for scope,(data,z,eps,nr) in scopes.items():
                    take=np.ones(len(z),bool) if group==-1 else data['g']==group
                    ee=[e for e in eps if group==-1 or e['g']==group]
                    if not len(ee):collect=[]; break
                    collect.append((z[take],ee,data['logits'][take]))
                if not collect:continue
                (zf,ef,rawf),(zv,ev,rawv)=collect
                rows.append({'fold':fold,'arm':arm,'source':'all' if group==-1 else SOURCES[group],
                    'FIT_patients':len(ef),'VAL_patients':len(ev),'FIT_floor':floor,
                    'FIT_patient_median_mean':np.mean([e['median'] for e in ef]),'VAL_patient_median_mean':np.mean([e['median'] for e in ev]),
                    'FIT_patient_IQR_mean':np.mean([e['iqr'] for e in ef]),'VAL_patient_IQR_mean':np.mean([e['iqr'] for e in ev]),
                    'FIT_raw_spread_mean':np.mean([e['raw_spread'] for e in ef]),'VAL_raw_spread_mean':np.mean([e['raw_spread'] for e in ev]),
                    'patient_median_Wasserstein':wasserstein_distance([e['median'] for e in ef],[e['median'] for e in ev]),
                    'patient_IQR_Wasserstein':wasserstein_distance([e['iqr'] for e in ef],[e['iqr'] for e in ev]),
                    'FIT_floor_fraction':np.mean([e['floor'] for e in ef]),'VAL_floor_fraction':np.mean([e['floor'] for e in ev]),
                    'FIT_clip_patient_mean':np.mean([e['clip'] for e in ef]),'VAL_clip_patient_mean':np.mean([e['clip'] for e in ev]),
                    'FIT_z_mean':float(zf.mean()),'VAL_z_mean':float(zv.mean()),'FIT_z_std':float(zf.std()),'VAL_z_std':float(zv.std()),
                    'z_Wasserstein':wasserstein_distance(zf,zv),'z_KS_statistic':ks_2samp(zf,zv,method='asymp').statistic,
                    'FIT_z_q05':float(np.quantile(zf,.05)),'VAL_z_q05':float(np.quantile(zv,.05)),
                    'FIT_z_q95':float(np.quantile(zf,.95)),'VAL_z_q95':float(np.quantile(zv,.95)),
                    'channel_distribution_descriptive_not_independent_test':True,'VAL_used_for_fitting_or_admission':False})
    pd.DataFrame(rows).to_csv(pub/'OOF_TO_VAL_TRANSPORT_AUDIT.csv',index=False)
    return rows
