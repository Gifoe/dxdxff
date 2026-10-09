"""Isolated fold processes: frozen formal scorer, four legal teachers, FIT posterior."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
from common import sha,digest,write_json,metric,METRICS
from model import SourceA0,SOURCES
from posterior import fit_density,infer,decode,calibration,normalize,log_densities

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--fold',type=int,required=True)
    p.add_argument('--phase',choices=['formal','teachers','posterior'],required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
    sys.path.insert(0,str(a.source/'pr_uncertainty_aware_supervision_seed42_v1/code'))
    from uas_core import PRMLP,prepare,oof_plan,validate_exclusion,seed_all,state_hash,torch_write
    from train import train
    torch.set_num_threads(2); root=a.runtime/f'fold{a.fold}'; pub=a.runtime/'public'
    lock=json.loads(a.protocol.read_text()); tests=json.loads((pub/'DECODER_UNIT_TEST_AUDIT.json').read_text())
    assert tests['status']=='PASS' and tests['protocol_sha256']==sha(a.protocol)
    code={q.name:sha(q) for q in Path(__file__).parent.glob('*.py')}; assert code==tests['code_sha256']
    path=root/'BANK_PRIVATE.pt'; d=torch.load(path,map_location='cpu',weights_only=False)
    assert d['binding']['protocol']==sha(a.protocol) and d['binding']['code']==code
    bind=digest({'bank':sha(path),'protocol':sha(a.protocol),'tests':sha(pub/'DECODER_UNIT_TEST_AUDIT.json'),'code':code})
    fit=set(d['patient'][d['train']]); val=set(d['patient'][d['val']]); assert fit.isdisjoint(val)
    if a.phase=='formal':
        model,best,hist=train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],root/'D2',42+1009*a.fold,bind,a.device)
        with torch.no_grad():
            x=torch.as_tensor(d['x'],device=a.device); g=torch.as_tensor(d['g'],device=a.device)
            base,delta=model.parts(x,g); logits=base+delta; scores=torch.sigmoid(logits).cpu().numpy()
        artifact={'binding':bind,'checkpoint_sha256':sha(root/'D2/BEST_PRIVATE.pt'),'logits':logits.cpu().numpy(),
            'base':base.cpu().numpy(),'delta':delta.cpu().numpy(),'scores':scores,'threshold':best['threshold'],'epoch':best['epoch'],
            'history':hist,'rank_singular_values':torch.linalg.svdvals(model.V).cpu().detach().tolist()}
        torch_write(root/'D2_FROZEN_PRIVATE.pt',artifact)
        write_json(root/'FORMAL_COMPLETE.json',{'status':'PASS','binding':bind,'checkpoint_sha256':artifact['checkpoint_sha256'],'epoch':best['epoch'],'threshold':best['threshold']})
        print('FORMAL_COMPLETE',a.fold,flush=True); return
    full=torch.load(root/'D2_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert full['binding']==bind and full['checkpoint_sha256']==sha(root/'D2/BEST_PRIVATE.pt')
    if a.phase=='teachers':
        tr=d['train']; fx=d['raw_x'][tr]; fy=d['y'][tr]; fp=d['patient'][tr]; fg=d['g'][tr]; fc=d['channel'][tr]
        oof=np.full(len(tr),np.nan); coverage=np.zeros(len(tr),int); audits=[]
        for plan in d['plans']:
            validate_exclusion(plan['train'],plan['validation'],plan['query'],fit)
            assert not set(plan['train']+plan['validation']+plan['query'])&val
            ti=np.flatnonzero(np.isin(fp,plan['train'])); vi=np.flatnonzero(np.isin(fp,plan['validation'])); qi=np.flatnonzero(np.isin(fp,plan['query']))
            tx,pre=prepare(fx,fp,ti); assert pre['fit_patient_ids']==plan['train']
            pb=digest({q:pre[q].tolist() for q in ['mean','scale','var','imputer_statistics']})
            tb=bind+digest(plan)+pb; cell=root/f'D2_teacher{plan["group"]}'
            model,best,hist=train(tx,fy,fp,fg,ti,vi,cell,plan['seed'],tb,a.device)
            with torch.no_grad():oof[qi]=model(torch.as_tensor(tx[qi],device=a.device),torch.as_tensor(fg[qi],device=a.device)).cpu().numpy()
            coverage[qi]+=1
            audits.append({'fold':a.fold,'family':'D2','group':plan['group'],'checkpoint_sha256':sha(cell/'BEST_PRIVATE.pt'),
                'teacher_code_hash':digest(code),'preprocessor_hash':pb,'query_identity_hash':digest(list(zip(fp[qi],fc[qi]))),
                'query_channels':len(qi),'train_patients':len(plan['train']),'selection_patients':len(plan['validation']),'query_patients':len(plan['query']),
                'query_train_overlap':0,'query_selection_overlap':0,'outer_overlap':0,'selected_epoch':best['epoch'],'threshold':best['threshold'],
                'source_fit_patient_counts':[len(set(fp[ti][fg[ti]==k])) for k in range(4)],'completed_epochs':len(hist)})
        assert (coverage==1).all() and np.isfinite(oof).all()
        torch_write(root/'D2_OOF_PRIVATE.pt',{'binding':bind,'logits':oof,'audits':audits,'patient':fp,'channel':fc})
        write_json(root/'TEACHERS_COMPLETE.json',{'status':'PASS','binding':bind,'teachers':audits,'once_per_fit_channel':True})
        print('TEACHERS_COMPLETE',a.fold,flush=True); return
    of=torch.load(root/'D2_OOF_PRIVATE.pt',map_location='cpu',weights_only=False); assert of['binding']==bind
    tr=d['train']; va=d['val']; fp=d['patient'][tr]; fg=d['g'][tr]; fy=d['y'][tr]
    assert np.array_equal(of['patient'],fp) and np.array_equal(of['channel'],d['channel'][tr])
    densities={}; fitrows=[]; paramrows=[]
    for name,logits in [('D1',d['a0_oof']),('D3',of['logits'])]:
        den=fit_density(logits,fy,fp,fg,fit); densities[name]=den
        for pat in sorted(fit):
            ix=np.flatnonzero(fp==pat); ix=ix[np.argsort(d['channel'][tr][ix].astype(str),kind='stable')]
            q,diag=infer(logits[ix],int(fg[ix[0]]),den); qfixed,df=infer(logits[ix],int(fg[ix[0]]),den,True)
            ez,dc=decode(q,d['channel'][tr][ix],logits[ix]); ezf,_=decode(qfixed,d['channel'][tr][ix],logits[ix]); z,_=normalize(logits[ix])
            le,ln=log_densities(z,den['sources'][int(fg[ix[0]])]); ge,gn=log_densities(z,den['global'])
            brier,ece=calibration(q,fy[ix]); bf,ef=calibration(qfixed,fy[ix]); truth=float(np.mean(fy[ix]==0))
            row={'method':name,'fold':a.fold,'scope':'FIT_OOF_FITTED_DIAGNOSTIC','patient':pat,'center':str(d['center'][tr][ix[0]]),
                'pi':diag['pi'],'prior':diag['prior'],'actual':truth,'predicted':float(ez.mean()),'pi_abs_error':abs(diag['pi']-truth),
                'prior_abs_error':abs(diag['prior']-truth),'brier':brier,'ece':ece,'fixed_brier':bf,'fixed_ece':ef,
                'll_gain':diag['ll_gain'],'converged':diag['converged'],'near_boundary':diag['near_boundary'],'fallback':diag['fallback'],
                'low_iqr':diag['low_iqr'],'separation':diag['separation'],'extreme_k':dc['k'] in [0,len(ix)],
                'source_class_loglik_minus_global':float(np.mean(np.where(fy[ix]==0,le-ge,ln-gn))),
                'macro_map':metric(fy[ix],1-q,.5,pred=~ez,order_score=logits[ix])['macro_f1'],
                'macro_fixed':metric(fy[ix],1-qfixed,.5,pred=~ezf,order_score=logits[ix])['macro_f1']}
            fitrows.append(row)
        paramrows.append({'method':name,'fold':a.fold,'density':den})
    # Seal the fitted model and falsification outputs BEFORE formal VAL inference.
    torch_write(root/'POSTERIOR_FROZEN_PRIVATE.pt',{'binding':bind,'scorer_checkpoint':full['checkpoint_sha256'],
        'densities':densities,'fit_falsification':fitrows,'parameters':paramrows,'all_fitting_patients':sorted(fit)})
    density_hash=sha(root/'POSTERIOR_FROZEN_PRIVATE.pt')
    rows=[]; diagrows=[]; errors=[]; ranks=[]; source_rows=[]; distribution=[]; channels=[]
    for scope,ix in [('FIT_OOF',tr),('VAL_FULL_FIT_SCORER',va)]:
        for name,logits in [('A0',d['a0_oof'] if scope=='FIT_OOF' else d['a0_logits'][va]),('D2',of['logits'] if scope=='FIT_OOF' else full['logits'][va])]:
            for cls,label in [(None,'all'),(0,'EZ'),(1,'NEZ')]:
                vv=logits if cls is None else logits[d['y'][ix]==cls]
                distribution.append({'fold':a.fold,'family':name,'scope':scope,'class':label,'channels':len(vv),
                    'mean':float(vv.mean()),'std':float(vv.std()),'q05':float(np.quantile(vv,.05)),'median':float(np.median(vv)),'q95':float(np.quantile(vv,.95))})
    for pat in sorted(val):
        ix=va[d['patient'][va]==pat]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
        y=d['y'][ix]; channel=d['channel'][ix]; group=int(d['g'][ix[0]]); center=str(d['center'][ix[0]])
        score0=d['a0_scores'][ix]; log0=d['a0_logits'][ix]; score2=full['scores'][ix]; log2=full['logits'][ix]
        b0=score0>=d['A0_threshold']; decisions={'D0':b0,'D2':score2>=full['threshold']}; scorer={'D0':score0,'D2':score2}; posterior_info={}
        for name,logits,original_score in [('D1',log0,score0),('D3',log2,score2)]:
            q,diag=infer(logits,group,densities[name]); qf,fd=infer(logits,group,densities[name],True)
            ez,dc=decode(q,channel,logits); ezf,df=decode(qf,channel,logits)
            decisions[name]=~ez; scorer[name]=original_score
            mpost=metric(y,1-q,.5,pred=~ez,order_score=-diag['logodds'])
            mraw=metric(y,original_score,.5,pred=~ez)
            for m in ['ez_auprc','ez_auroc','ez_mrr','top1_is_ez']:assert np.isclose(mpost[m],mraw[m],rtol=0,atol=1e-12,equal_nan=True)
            brier,ece=calibration(q,y); bf,ef=calibration(qf,y); truth=float(np.mean(y==0))
            diagrows.append({'fold':a.fold,'method':name,'patient':pat,'center':center,'scope':'VAL','pi':diag['pi'],'prior':diag['prior'],
                'actual':truth,'predicted':float(ez.mean()),'pi_abs_error':abs(diag['pi']-truth),'prior_abs_error':abs(diag['prior']-truth),
                'brier':brier,'ece':ece,'fixed_brier':bf,'fixed_ece':ef,'ll_gain':diag['ll_gain'],'converged':diag['converged'],
                'near_boundary':diag['near_boundary'],'fallback':diag['fallback'],'low_iqr':diag['low_iqr'],'separation':diag['separation'],
                'extreme_k':dc['k'] in [0,len(ix)],'k_minus_expected':dc['k']-dc['expected_count'],'objective':dc['objective'],
                'macro_map':mpost['macro_f1'],'macro_fixed':metric(y,original_score,.5,pred=~ezf)['macro_f1'],
                'source_class_loglik_minus_global':np.nan})
            posterior_info[name]={'q':q,'diag':diag,'decoder':dc,'fixed_q':qf,'fixed_pred':~ezf}
        for name in ['D0','D1','D2','D3']:
            m=metric(y,scorer[name],.5,pred=decisions[name]); row={'fold':a.fold,'patient':pat,'center':center,'method':name,**m}; rows.append(row)
            changed=decisions[name]!=b0; before=b0==y; after=decisions[name]==y
            errors.append({'fold':a.fold,'patient':pat,'center':center,'method':name,
                'TP_gained':int(((y==0)&b0&~decisions[name]).sum()),'TP_lost':int(((y==0)&~b0&decisions[name]).sum()),
                'FP_added':int(((y==1)&b0&~decisions[name]).sum()),'FP_removed':int(((y==1)&~b0&decisions[name]).sum()),
                'errors_recovered':int((~before&after).sum()),'correct_spoiled':int((before&~after).sum()),'changed':int(changed.sum())})
            s=scorer[name]; aa=np.sign(score0[:,None]-score0); bb=np.sign(s[:,None]-s); upper=np.triu(np.ones_like(aa,bool),1)
            ranks.append({'fold':a.fold,'patient':pat,'center':center,'method':name,'pair_reversal_fraction':float(np.mean((aa*bb<0)[upper])) if upper.any() else 0.,
                'changed_rank_positions':float(np.mean(np.argsort(score0,kind='stable')!=np.argsort(s,kind='stable'))),
                'corresponding_scorer_rank_change':0. if name in ['D1','D3'] else np.nan})
        delta=full['delta'][ix]; base=full['base'][ix]
        source_rows.append({'fold':a.fold,'patient':pat,'center':center,'mean_delta':float(delta.mean()),'mean_abs_delta':float(np.abs(delta).mean()),
            'std_delta':float(delta.std()),'max_abs_delta':float(np.abs(delta).max()),'mean_abs_base':float(np.abs(base).mean()),
            'delta_base_energy_ratio':float(np.mean(delta**2)/max(np.mean(base**2),1e-12)),
            'effective_rank':int(np.sum(np.asarray(full['rank_singular_values'])>1e-6))})
        channels.append({'patient':pat,'fold':a.fold,'channel':channel,'y':y,'decisions':decisions,'posterior':posterior_info,
            'D0_score':score0,'D2_score':score2})
    torch_write(root/'VALIDATION_PRIVATE.pt',{'binding':bind,'density_sha256':density_hash,'scorer_checkpoint':full['checkpoint_sha256'],
        'rows':rows,'diagnostics':diagrows,'FIT_falsification':fitrows,'errors':errors,'ranks':ranks,'source':source_rows,
        'distribution':distribution,'parameters':paramrows,'channels':channels})
    write_json(root/'POSTERIOR_COMPLETE.json',{'status':'PASS','binding':bind,'density_sha256':density_hash,'D2_D3_same_scorer_sha256':full['checkpoint_sha256'],
        'validation_sha256':sha(root/'VALIDATION_PRIVATE.pt'),'FIT_falsification_before_VAL':True,'outer_evaluation':False})
    print('POSTERIOR_MATCHED_VAL_COMPLETE',a.fold,flush=True)

if __name__=='__main__':main()
