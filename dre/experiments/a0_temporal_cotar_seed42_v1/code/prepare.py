"""Seal FIT/VAL only; replay frozen A0 and regenerate eval-logit OOF teachers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from base import (sha,json_write,torch_write,prepare,PRMLP,oof_plan,patient_metrics,METRICS)
from features import normalize


def binding(protocol):
    return {**{n:sha(Path(__file__).parent/n) for n in ['base.py','model.py','features.py','prepare.py','train.py','run.py']},
            'protocol':sha(protocol)}


def describe(logit,y,scope,fold):
    rows=[]
    for label,ix in [('all',np.ones(len(y),bool)),('EZ',y==0),('NEZ',y==1)]:
        v=logit[ix]
        rows.append({'fold':fold,'scope':scope,'class':label,'channels':len(v),
            'mean':float(v.mean()),'std':float(v.std()),'q05':float(np.quantile(v,.05)),
            'q50':float(np.median(v)),'q95':float(np.quantile(v,.95)),
            'mean_absolute_margin':float(np.abs(v).mean())})
    return rows


def main():
    p=argparse.ArgumentParser()
    for n in ['runtime','protocol','prior','a0_runtime']:p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    a=p.parse_args(); root=a.runtime; pub=root/'public'; lock=json.loads(a.protocol.read_text())
    torch.set_num_threads(2)
    coverage=json.loads((pub/'TEMPORAL_COVERAGE_AUDIT.json').read_text()); assert coverage['pass']
    bankpath=root/'TEMPORAL_PRIVATE.pt'; assert sha(bankpath)==coverage['private_temporal_bank_sha256']
    temporal=torch.load(bankpath,weights_only=False)['banks']; bind=binding(a.protocol)
    export=a.prior/'gate/FEATURES_PRIVATE.npz'; assert sha(export)==lock['private_88D_export_sha256']
    # Hash-verified source export contains all-cohort fields; seal development
    # arrays before fitting anything or generating a new prediction.
    with np.load(export,allow_pickle=True) as d: arrays={k:d[k] for k in d.files}
    teacher_audit=json.loads((a.prior/'OOF_TEACHER_AUDIT.json').read_text()); assert teacher_audit['status']=='PASS'
    previous=json.loads((a.a0_runtime/'public/A0_REPRODUCTION.json').read_text()); assert previous['status']=='PASS'
    selection=json.loads((a.a0_runtime/'public/SELECTION_AUDIT.json').read_text())
    teacher_rows=[]; reproductions=[]; normalizations=[]; shifts=[]
    for fold in range(1,6):
        s=arrays['split_fold']==fold
        fit=set(arrays['split_patient'][s&(arrays['split_role']=='fit')].astype(str))
        val=set(arrays['split_patient'][s&(arrays['split_role']=='validation')].astype(str))
        assert fit.isdisjoint(val)
        ix=np.flatnonzero(np.isin(arrays['patient'],list(fit|val)))
        x,y,pid,ch,ce=[arrays[k][ix] for k in ['x','y','patient','channel','center']]
        pid=pid.astype(str); ch=ch.astype(str); ce=ce.astype(str)
        tr=np.flatnonzero(np.isin(pid,list(fit))); va=np.flatnonzero(np.isin(pid,list(val)))
        xx,pre=prepare(x,pid,tr); assert set(pre['fit_patient_ids'])==fit
        path=a.a0_runtime/f'fold{fold}/A0/BEST_PRIVATE.pt'
        sel=next(r for r in selection['selections'] if r['fold']==fold and r['arm']=='A0')
        assert sha(path)==sel['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=PRMLP().eval(); model.load_state_dict(saved['model'])
        for param in model.parameters(): param.requires_grad_(False)
        with torch.no_grad(): full=model(torch.from_numpy(xx)).numpy(); score=torch.sigmoid(torch.from_numpy(full[va])).numpy()
        ref=pd.read_csv(a.a0_runtime/f'fold{fold}/A0/VALIDATION_CHANNEL_PRIVATE.csv')
        frame=pd.DataFrame({'patient':pid[va],'channel':ch[va],'y_nez':y[va],'score':score})
        joined=frame.merge(ref,on=['patient','channel','y_nez'],validate='one_to_one')
        assert len(joined)==len(va)==len(ref)
        drift=float(np.abs(joined.score-joined.score_nez).max()); assert drift<=2e-7
        tau=sel['threshold']; pm=patient_metrics(y[va],score,pid[va],ce[va],tau)
        old=pd.read_csv(a.a0_runtime/f'fold{fold}/A0/VALIDATION_PATIENT_PRIVATE.csv')
        assert np.allclose(pm[METRICS].mean(),old[METRICS].mean(),rtol=0,atol=1e-12)
        pm['arm']='T0'; pm['fold']=fold
        foldroot=root/f'fold{fold}'; foldroot.mkdir(exist_ok=True)
        pm.to_csv(foldroot/'T0_PATIENT_PRIVATE.csv',index=False)
        torch_write(foldroot/'A0_FROZEN_PRIVATE.pt',{'checkpoint_hash':sha(path),'state':saved['model'],
            'preprocessor':pre,'requires_grad':False,'threshold':tau})
        # Reuse the original hard-label teachers, but make deterministic eval
        # logits afresh. MC-dropout artifacts are not residual training inputs.
        oof=np.full(len(tr),np.nan); seen=np.zeros(len(tr),int)
        fx,fy,fp,fc=x[tr],y[tr],pid[tr],ch[tr]
        for plan in oof_plan(fit,fold):
            ti=np.flatnonzero(np.isin(fp,plan['train'])); query=np.flatnonzero(np.isin(fp,plan['query']))
            assert set(fp[query]).isdisjoint(plan['train']) and set(fp[query]).isdisjoint(plan['validation'])
            tx,tp=prepare(fx,fp,ti); assert tp['fit_patient_ids']==plan['train']
            teacherpath=a.prior/f'fold{fold}/teacher{plan["group"]}/BEST_PRIVATE.pt'
            record=next(r for r in teacher_audit['folds'] if r['fold']==fold and r['teacher_group']==plan['group'])
            assert sha(teacherpath)==record['checkpoint_sha256']
            old_oof=torch.load(a.prior/f'fold{fold}/teacher{plan["group"]}/OOF_PRIVATE.pt',weights_only=False)
            assert old_oof['source']['plan']==plan
            assert np.array_equal(old_oof['patient'],fp[query]) and np.array_equal(old_oof['channel'],fc[query])
            assert np.array_equal(old_oof['preprocessor']['mean'],tp['mean'])
            assert np.array_equal(old_oof['preprocessor']['scale'],tp['scale'])
            ts=torch.load(teacherpath,map_location='cpu',weights_only=False)
            assert ts['epoch']==record['selected_epoch']
            teacher=PRMLP().eval(); teacher.load_state_dict(ts['model'])
            with torch.no_grad(): logits=teacher(torch.from_numpy(tx[query])).numpy()
            oof[query]=logits; seen[query]+=1
            teacher_rows.append({'fold':fold,'group':plan['group'],'checkpoint_sha256':sha(teacherpath),
                'train_patients':len(plan['train']),'selection_patients':len(plan['validation']),
                'query_patients':len(plan['query']),'query_channels':len(query),
                'query_train_overlap':0,'query_selection_overlap':0,'outer_overlap':0,
                'preprocessor_exact_original_teacher':True,'deterministic_eval_logits_not_MC':True})
        assert np.isfinite(oof).all() and (seen==1).all()
        relevant={p:temporal[p] for p in sorted(fit|val)}
        for p,b in relevant.items(): assert np.array_equal(np.asarray(b['channels']),ch[pid==p])
        norm,ns=normalize(relevant,fit)
        base=full.copy(); base[tr]=oof
        sealed={'binding':bind,'patient':pid,'channel':ch,'center':ce,'y':y,'train':tr,'val':va,
            'base_logits':base.astype(np.float32),'full_A0_logits':full.astype(np.float32),
            'T0_threshold':tau,'banks':norm,'normalizer_private':ns,'A0_checkpoint_hash':sha(path)}
        torch_write(foldroot/'DEVELOPMENT_PRIVATE.pt',sealed)
        reproductions.append({'fold':fold,'max_probability_drift':drift,'threshold':tau,
            'selected_A0_epoch':sel['epoch'],'checkpoint_sha256':sha(path),'metrics':pm[METRICS].mean().to_dict()})
        normalizations.append({'fold':fold,'fit_patients':len(fit),'validation_patients':len(val),
            'fit_tokens':ns['tokens'],'outer_rows_in_bank':0,'input_sha256':sha(foldroot/'DEVELOPMENT_PRIVATE.pt'),
            'T1_T2_use_same_file':True,'finite_all':True,'statistics_fit_only':True})
        shifts+=describe(oof,fy,'FIT_OOF',fold)+describe(full[tr],fy,'FIT_IN_SAMPLE_DIAGNOSTIC',fold)+describe(full[va],y[va],'VAL_FROZEN_A0',fold)
        print('SEALED_REPLAY_OOF_COMPLETE',fold,flush=True)
    mean=np.mean([r['metrics']['macro_f1'] for r in reproductions]); assert abs(mean-.6380797828499001)<1e-12
    json_write(pub/'A0_REPRODUCTION.json',{'status':'PASS','folds':reproductions,'development_macro_f1':float(mean),
        'T0_retrained':False,'new_outer_predictions':False,'parameters':8817})
    json_write(pub/'OOF_A0_AUDIT.json',{'status':'PASS','teachers':20,'reused_checkpoints':True,
        'generated_logits_eval_mode':True,'one_prediction_per_FIT_channel':True,'fold_groups':teacher_rows})
    json_write(pub/'TEMPORAL_NORMALIZATION_AUDIT.json',{'status':'PASS','folds':normalizations,
        'relative_label_blind':True,'model_input_has_no_patient_center_label':True,'A0_preprocessing_unchanged':True,
        'outer_labels_materialized_only_in_original_source_export_before_sealing':True})
    pd.DataFrame(shifts).to_csv(pub/'A0_OOF_DISTRIBUTION_AUDIT.csv',index=False)
    json_write(pub/'RUN_STATUS.json',{'status':'SEALED_REPLAY_OOF_COMPLETE','outer_evaluation':False,'binding':bind})


if __name__=='__main__':main()
