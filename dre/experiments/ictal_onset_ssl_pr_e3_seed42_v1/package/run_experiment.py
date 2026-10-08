"""Fixed 80-patient five-fold E0-E3 study; optional gated E4.

All model and threshold decisions use FIT/VAL patients only.  Test patients
are read only after every allowed variant and selection is committed.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from model import IctalLocalization, patient_bce, vicreg

VARIANTS={'E0':(False,False,False),'E1':(False,True,False),'E2':(True,False,False),'E3':(True,True,False),'E4':(True,True,True),'E5_SHAM':(True,True,False)}


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


class PatientFiles:
    def __init__(self,folder:Path,mf:dict,device):
        self.folder=Path(folder);self.meta=mf['cohort'];self.device=torch.device(device)
    def get(self,pid,with_label=True):
        path=self.folder/self.meta[pid]['file']
        with np.load(path,allow_pickle=False) as dat:
            pair=torch.tensor(dat['pair'],device=self.device,dtype=torch.float32)
            present=torch.tensor(dat['present'],device=self.device,dtype=torch.bool)
            y=torch.tensor(dat['labels_ez'],device=self.device,dtype=torch.float32) if with_label else None
        if pair.ndim!=4 or pair.shape[1]!=present.shape[1] or pair.shape[0]!=present.shape[0]:
            raise ValueError('patient tensor contract failed')
        return pair,present,y


def metric_one(labels_ez,score_ez,threshold_ez):
    yt=np.asarray(labels_ez,dtype=int);sc=np.asarray(score_ez,dtype=float)
    if len(yt)!=len(sc) or len(yt)<1:raise ValueError('bad scores')
    pred=(sc>=threshold_ez).astype(int)
    # Rank within the patient; rank 1 is the highest EZ score.
    rank=np.argsort(-sc,kind='stable')
    pos=np.flatnonzero(yt[rank]==1)
    mrr=(1.0/(pos[0]+1)) if len(pos) else float('nan')
    top1=float(yt[rank[0]]==1) if len(rank) else float('nan')
    res={
      'mrr':float(mrr),'top1':top1,
      'macro_f1':float(f1_score(yt,pred,labels=[0,1],average='macro',zero_division=0)),
      'ez_f1':float(f1_score(yt,pred,labels=[1],average='binary',pos_label=1,zero_division=0)),
      'n_channels':int(len(yt)),'ez_count':int(yt.sum()),
      'ez_ap':float(average_precision_score(yt,sc)) if yt.sum()>0 and yt.sum()<len(yt) else float('nan'),
      'ez_auroc':float(roc_auc_score(yt,sc)) if yt.sum()>0 and yt.sum()<len(yt) else float('nan'),
    }
    return res


def score_patient(model,loader,pid):
    pair,present,y=loader.get(pid,with_label=True)
    model.eval()
    with torch.no_grad():
        logits,valid=model(pair,present)
    mask=valid.detach().cpu().numpy().astype(bool)
    labels=y.detach().cpu().numpy()[mask].astype('int8')
    scores=torch.sigmoid(-logits[valid]).detach().cpu().numpy()
    return {'id':pid,'labels_ez':labels,'scores_ez':scores,'center':loader.meta[pid]['center']}


def pool(rows,threshold):
    entries=[metric_one(row['labels_ez'],row['scores_ez'],threshold) for row in rows]
    return {key:float(np.nanmean([e[key] for e in entries])) for key in ('macro_f1','ez_f1','ez_ap','ez_auroc','mrr','top1')}


def fit_threshold(rows):
    # Single validation-selected global threshold per fold; NO per-patient labels at inference.
    thresholds=np.round(np.linspace(.05,.95,37),6)
    candidate=[(float(pool(rows,float(t))['macro_f1']),-abs(float(t)-.5),float(t)) for t in thresholds]
    _,_,best=max(candidate)
    return best


def run_ssl(fold,loader,fit_ids,out_dir,epochs,seed,learning_rate=3e-4,shuffle_transition=False):
    set_seed(seed)
    anchor=IctalLocalization(use_pr=False)
    encoder=anchor.encoder.to(loader.device)
    opt=torch.optim.AdamW(encoder.parameters(),lr=learning_rate,weight_decay=1e-4)
    history=[]
    if epochs<=0:raise ValueError('SSL must have at least one pretraining epoch')
    for epoch in range(1,epochs+1):
        encoder.train();rows=[]
        order=np.random.permutation(fit_ids)
        for offset in range(0,len(order),2):
            all_v1=[];all_v2=[]
            for pid in order[offset:offset+2]:
                pair,present,_=loader.get(pid,with_label=False)
                # Balanced SSL patient sampling; selection is label-blind.
                eligible=torch.nonzero(present.any(0),as_tuple=False).flatten()
                if len(eligible)>48:
                    eligible=eligible[torch.randperm(len(eligible),device=pair.device)[:48]]
                    pair=pair[:,eligible];present=present[:,eligible]
                if int(present.sum())<2:continue
                if shuffle_transition:
                    # Sham SSL control: permute ictal segments across unlabeled
                    # channels/seizures within this patient, breaking true pair identity.
                    pair=pair.clone()
                    ict=pair[:,:,1,:]
                    values=ict[present].clone()
                    shuffled=values[torch.randperm(len(values),device=values.device)]
                    ict[present]=shuffled
                    pair[:,:,1,:]=ict
                all_v1.append(encoder(pair,present,augment=True)[present])
                all_v2.append(encoder(pair,present,augment=True)[present])
            if not all_v1:continue
            view1=torch.cat(all_v1,0);view2=torch.cat(all_v2,0)
            if len(view1)<4:continue
            opt.zero_grad(set_to_none=True)
            loss,parts=vicreg(view1,view2)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite SSL loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.)
            opt.step();rows.append([float(loss.detach()),*parts])
        if not rows:raise RuntimeError('no valid SSL patients')
        means=np.mean(rows,axis=0)
        history.append({'fold':fold,'epoch':epoch,'total':float(means[0]),'inv':float(means[1]),
                        'var':float(means[2]),'cov':float(means[3])})
    file=out_dir/f'fold_{fold}_ssl_{"sham" if shuffle_transition else "true"}_encoder.pt'
    torch.save(encoder.state_dict(),file)
    print(f'[SSL fold {fold} {"SHAM" if shuffle_transition else "REAL"}] epochs={epochs} last_loss={history[-1]["total"]:.5f}',flush=True)
    return file,history


def train_one(fold,variant,loader,fit_ids,val_ids,output,epochs,seed,ssl_file=None):
    has_ssl,use_pr,use_attn=VARIANTS[variant]
    set_seed(seed)
    model=IctalLocalization(use_pr=use_pr,use_attention=use_attn).to(loader.device)
    if has_ssl:
        if ssl_file is None:raise ValueError('SSL state required')
        # Explicit loading of the SAME SSL encoder for E2/E3/E4.
        model.encoder.load_state_dict(torch.load(ssl_file,map_location=loader.device,weights_only=True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
    best=None;history=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for pid in np.random.permutation(fit_ids):
            pair,present,y=loader.get(pid,with_label=True)
            optimizer.zero_grad(set_to_none=True)
            logits,mask=model(pair,present)
            # Full available channel set always used for PR, including label-blind context.
            loss=patient_bce(logits,y,mask)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite supervised loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:raise ValueError('no fit patients')
        val_rows=[score_patient(model,loader,pid) for pid in val_ids]
        val_metrics=pool(val_rows,.5)
        primary=val_metrics['ez_ap'] if np.isfinite(val_metrics['ez_ap']) else -1.
        key=(primary,val_metrics['macro_f1'],-epoch)
        history.append({'fold':fold,'variant':variant,'epoch':epoch,'fit_bce':float(np.mean(losses)),
                        'val_ap':val_metrics['ez_ap'],'val_f1_0p5':val_metrics['macro_f1']})
        if best is None or key>best[0]:
            best=(key,copy.deepcopy(model.state_dict()),epoch)
    assert best is not None
    model.load_state_dict(best[1]);model.eval()
    val_rows=[score_patient(model,loader,pid) for pid in val_ids]
    threshold=fit_threshold(val_rows)
    selected=pool(val_rows,threshold)
    checkpoint=output/f'fold_{fold}_{variant}.pt'
    torch.save({'state_dict':model.state_dict(),'variant':variant,'fold':fold,
                'selected_epoch':best[2],'ez_threshold':threshold,
                'selection':'validation mean patient EZ-AP, tie F1@0.5',
                'val_metrics':selected},checkpoint)
    print(f'[SUP fold {fold} {variant}] selected epoch={best[2]} val_f1={selected["macro_f1"]:.4f} val_AP={selected["ez_ap"]:.4f}',flush=True)
    return {'fold':fold,'variant':variant,'checkpoint':str(checkpoint),'checkpoint_sha256':sha_file(checkpoint),
            'val_metrics':selected,'threshold':threshold,'selected_epoch':best[2]},history


def gated_attention_allowed(records):
    by={(e['fold'],e['variant']):e for e in records}
    d_f1=[];d_ap=[]
    for fold in range(1,6):
        a=by[(fold,'E1')]['val_metrics'];b=by[(fold,'E3')]['val_metrics']
        d_f1.append(b['macro_f1']-a['macro_f1']);d_ap.append(b['ez_ap']-a['ez_ap'])
    result={'mean_val_delta_f1_E3_minus_E1':float(np.mean(d_f1)),
            'mean_val_delta_ap_E3_minus_E1':float(np.nanmean(d_ap)),
            'positive_f1_folds':int(sum(x>0 for x in d_f1)),
            'passed': bool(np.mean(d_f1)>=.02 and np.nanmean(d_ap)>=0 and sum(x>0 for x in d_f1)>=4),
            'predeclared_gate':'>=0.02 macro F1; EZ-AP nondecreasing; >=4/5 positive F1 folds'}
    return result


def summarize_test(rows,output):
    summary=[]
    by_var={name:[r for r in rows if r['variant']==name] for name in VARIANTS}
    for variant,subset in by_var.items():
        if not subset:continue
        summary.append({'variant':variant,**{key:float(np.nanmean([r[key] for r in subset])) for key in ('macro_f1','ez_f1','ez_ap','ez_auroc','mrr','top1')},'n_patients':len(subset)})
    # Paired bootstrap resampling of patient rows (not window/electrode units).
    lookup={(r['variant'],r['patient']):r for r in rows}
    pairs=[];rng=np.random.default_rng(42)
    for a,b in [('E3','E1'),('E1','E0'),('E2','E0'),('E3','E2'),('E4','E3'),('E3','E5_SHAM')]:
        patients=sorted({r['patient'] for r in by_var[a]}&{r['patient'] for r in by_var[b]})
        if not patients:continue
        delta=np.array([lookup[(a,p)]['macro_f1']-lookup[(b,p)]['macro_f1'] for p in patients])
        draw=rng.integers(0,len(delta),size=(2000,len(delta)))
        lo,hi=np.quantile(delta[draw].mean(axis=1),[.025,.975])
        pairs.append({'contrast':a+'-'+b,'delta_patient_macro_f1':float(delta.mean()),
                      'bootstrap_95_lower':float(lo),'bootstrap_95_upper':float(hi),'n_patients':len(patients)})
    (output/'test_summary.json').write_text(json.dumps({'variant_summary':summary,'paired_bootstrap':pairs},indent=2))
    return summary,pairs


def write_csv(path,rows):
    if not rows:return
    names=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)


def run(args):
    src=Path(args.export);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((src/'audit.json').read_text())
    if audit['status'] not in ('PASS_SOURCE_CODE_ALIGNMENT_ONLY','PASS_INDEPENDENT_ONSET_PROVENANCE'):
        raise RuntimeError('data audit is not admitted')
    if audit['status']!='PASS_INDEPENDENT_ONSET_PROVENANCE' and not args.allow_source_only:
        raise RuntimeError('independent onset provenance not verified; explicitly pass --allow-source-only')
    mf=json.loads((src/'manifest.json').read_text());patients=PatientFiles(src,mf,args.device)
    set_seed(args.seed)
    selections=[];hist=[];ssl_hist=[];ssl_by_fold={}
    for entry in sorted(mf['folds'],key=lambda x:x['fold_idx']):
        fold=entry['fold_idx'];fit_ids=entry['fit_subjects'];val_ids=entry['validation_subjects']
        ssl_file,logs=run_ssl(fold,patients,fit_ids,output,args.ssl_epochs,args.seed+fold)
        ssl_by_fold[fold]=ssl_file;ssl_hist.extend(logs)
        for variant in ('E0','E1','E2','E3'):
            sel,logs=train_one(fold,variant,patients,fit_ids,val_ids,output,args.supervised_epochs,
                               args.seed+fold*100,ssl_file=ssl_file)
            selections.append(sel);hist.extend(logs)
    gate=gated_attention_allowed(selections)
    print('[VALIDATION GATE]',gate,flush=True)
    if args.try_e4_if_passed and gate['passed']:
        for entry in sorted(mf['folds'],key=lambda x:x['fold_idx']):
            f=entry['fold_idx'];sel,logs=train_one(f,'E4',patients,entry['fit_subjects'],entry['validation_subjects'],
                                                  output,args.supervised_epochs,args.seed+f*100,
                                                  ssl_file=ssl_by_fold[f])
            selections.append(sel);hist.extend(logs)
    gate['E4_eligible_and_executed']=bool(args.try_e4_if_passed and gate['passed'])
    if args.run_sham_if_passed and gate['passed']:
        for entry in sorted(mf['folds'],key=lambda x:x['fold_idx']):
            f=entry['fold_idx']
            sham_file,sham_logs=run_ssl(f,patients,entry['fit_subjects'],output,args.ssl_epochs,args.seed+f,shuffle_transition=True)
            ssl_hist.extend([{**row,'condition':'sham'} for row in sham_logs])
            sel,logs=train_one(f,'E5_SHAM',patients,entry['fit_subjects'],entry['validation_subjects'],
                               output,args.supervised_epochs,args.seed+f*100,ssl_file=sham_file)
            selections.append(sel);hist.extend(logs)
    gate['E5_SHAM_eligible_and_executed']=bool(args.run_sham_if_passed and gate['passed'])
    (output/'validation_gate.json').write_text(json.dumps(gate,indent=2))
    (output/'validation_selections.json').write_text(json.dumps(selections,indent=2))
    write_csv(output/'ssl_history.csv',ssl_hist)
    write_csv(output/'supervised_history.csv',hist)
    # Freeze all scores, thresholds and selected checkpoints BEFORE reading test.
    lock={'protocol':'seed42_80_e0_e3','audit_sha256':sha_file(src/'audit.json'),
          'data_manifest_sha256':sha_file(src/'manifest.json'),'selections':selections,
          'attention_gate':gate,'outer_test_exposed_historically':True}
    (output/'score_selection_lock.json').write_text(json.dumps(lock,indent=2))
    if not args.evaluate_outer:
        (output/'RUN_STATUS.json').write_text(json.dumps({'status':'VALIDATION_COMPLETE_TEST_NOT_ACCESSED',
                 'data_status':audit['status'],'E4_eligible':gate['passed']},indent=2))
        return gate
    rows=[]
    for fold_info in mf['folds']:
        fold=fold_info['fold_idx']
        for sel in [s for s in selections if s['fold']==fold]:
            file=Path(sel['checkpoint'])
            if sha_file(file)!=sel['checkpoint_sha256']:raise RuntimeError('checkpoint changed after lock')
            name=sel['variant'];_,pr,att=VARIANTS[name]
            model=IctalLocalization(pr,att).to(patients.device)
            model.load_state_dict(torch.load(file,map_location=patients.device,weights_only=True)['state_dict'])
            threshold=sel['threshold']
            for sid in fold_info['test_subjects']:
                rec=score_patient(model,patients,sid)
                metric=metric_one(rec['labels_ez'],rec['scores_ez'],threshold)
                rows.append({'variant':name,'fold':fold,'patient':sid,'center':rec['center'],**metric})
    summary,pairs=summarize_test(rows,output)
    write_csv(output/'outer_per_patient_private.csv',rows)
    fold_summary=[];center_summary=[]
    for variant in sorted({r['variant'] for r in rows}):
        vr=[r for r in rows if r['variant']==variant]
        for f in range(1,6):
            sub=[r for r in vr if r['fold']==f]
            fold_summary.append({'variant':variant,'fold':f,'n_patients':len(sub),**{k:float(np.nanmean([row[k] for row in sub])) for k in ('macro_f1','ez_f1','ez_ap','ez_auroc','mrr','top1')}})
        for c in sorted({r['center'] for r in vr}):
            sub=[r for r in vr if r['center']==c]
            center_summary.append({'variant':variant,'center':c,'n_patients':len(sub),**{k:float(np.nanmean([row[k] for row in sub])) for k in ('macro_f1','ez_f1','ez_ap','ez_auroc','mrr','top1')}})
    write_csv(output/'outer_by_fold.csv',fold_summary)
    write_csv(output/'outer_by_center.csv',center_summary)
    status={'status':'EXPLORATORY_OUTER_COMPLETE' if audit['status']!='PASS_INDEPENDENT_ONSET_PROVENANCE' else 'PREDECLARED_OUTER_COMPLETE_HISTORICAL_TEST_EXPOSED',
            'n_variants':len(set(s['variant'] for s in selections)),'n_folds':5,'data_audit':audit['status'],
            'evaluation_protocol':'patient-held-out; all variant/checkpoint/threshold decisions fixed before outer test',
            'E4_triggered':gate['E4_eligible_and_executed'],'n_patient_variant_rows':len(rows)}
    (output/'RUN_STATUS.json').write_text(json.dumps(status,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--export',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--ssl-epochs',type=int,default=25)
    p.add_argument('--supervised-epochs',type=int,default=30)
    p.add_argument('--try-e4-if-passed',action='store_true')
    p.add_argument('--run-sham-if-passed',action='store_true')
    p.add_argument('--evaluate-outer',action='store_true')
    p.add_argument('--allow-source-only',action='store_true')
    a=p.parse_args()
    print(json.dumps(run(a),indent=2))


if __name__=='__main__':main()
