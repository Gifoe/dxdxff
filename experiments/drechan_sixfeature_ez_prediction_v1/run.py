"""Admission -> FIT-only tuning -> freeze all570 -> seal all570 -> report."""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
import core as c


def table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.partial');pd.DataFrame(rows).to_csv(tmp,index=False,float_format='%.17g');tmp.replace(path)


def admission(root):
    assert not (root/'private/INPUT_FREEZE.json').exists(),'Preserve prior admission; diagnose before rerun'
    d=c.load_data();pub=root/'public';pub.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('frozen_PR_reference',c.PR_PARENT)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    rows=[];variation=[];role_counts=[]
    for f,rr in enumerate(d['roles'],1):
        tr=rr['fit'];pid=d['patient'];raw=d['x']
        pre=c.fit_pre(raw[tr],pid[tr],True);actual=c.transform(raw,pid,pre)
        expected,_=ref.prepare(d['raw14'],pid,tr)
        assert np.array_equal(actual.astype(np.float32),expected[:,c.KEEP]),'Current DREChannNet PR mismatch'
        for role,ix in rr.items():
            assert np.array_equal(c.transform(raw[ix],pid[ix],pre),actual[ix]),'PR must be local to complete patient, not other patients/batch'
        # Changing held patients cannot change FIT preprocessing or FIT features.
        altered=raw.copy();held=np.concatenate([rr['validation'],rr['test']]);altered[held]*=1e7
        after=c.fit_pre(altered[tr],pid[tr],True);assert c.digest(pre)==c.digest(after)
        for arm in c.ARMS:
            cols=c.columns(arm);z=actual[:,cols];change=raw.copy();other=np.setdiff1d(np.arange(12),cols);change[:,other]+=1e6
            zz=c.transform(change,pid,c.fit_pre(change[tr],pid[tr],True))[:,cols]
            assert np.array_equal(z,zz),'Inactive features may not affect a single-feature input'
        for j in range(12):
            sds=np.array([np.std(raw[pid==p,j]) for p in np.unique(pid)])
            variation.append(dict(fold=f,coordinate_index=j,R14_index=c.KEEP[j],patients=80,
                nonconstant_patients=int((sds>0).sum()),constant_patients=int((sds==0).sum()),
                supported_PR_patients=int((sds>1e-8*pre['units'][j]).sum())))
        splits=c.inner_splits(pid[tr],f)
        for it,iv in splits:
            w=c.weights(d['y'][tr[it]],pid[tr[it]])
            for p in np.unique(pid[tr[it]]):
                ii=pid[tr[it]]==p
                for y in [0,1]:np.testing.assert_allclose(w[ii&(d['y'][tr[it]]==y)].sum(),len(it)/(2*len(np.unique(pid[tr[it]]))),atol=1e-10,rtol=0)
        rows.append(dict(fold=f,PR_exact_current_float32=True,all14_branch_inputs_isolated=True,
                         all_roles_disjoint=True,inner_splits=3,held_patients_never_fit_preprocessor=True))
        role_counts.append(dict(fold=f,**{role+'_patients':len(set(pid[ix])) for role,ix in rr.items()},
                               **{role+'_channels':len(ix) for role,ix in rr.items()}))
    binding=dict(source=c.sources(),inputs=c.EXPECTED,python=__import__('platform').python_version(),
                 numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    assert sklearn.__version__=='1.7.2'
    c.write(root/'private/INPUT_FREEZE.json',binding)
    audit=dict(status='PASS',patients=80,channels=7635,EZ_channels=1743,NEZ_channels=5892,seizures=256,
        valid_channel_windows=1471965,features=6,coordinates=12,roles=role_counts,
        source_patients={z:len(set(d['patient'][d['source']==z])) for z in ['hup','lzu','multicenter','pediatric']},
        folds=rows,binding=binding,historical_models_reused=0,LR_formal_fits_at_admission=0,
        raw_extraction=False,neural_imports=False,GPU_used=False,CPU_affinity=[2],
        seven_single_seizure_patients_retained=True,original_A_unmodified=True)
    c.write(pub/'INPUT_AUDIT.json',audit);c.write(pub/'ADMISSION.json',dict(status='PASS',folds=5,PR_parent_exact=True,LR_fits=0))
    table(pub/'coordinate_variation.csv',variation)
    print('PREDICTION_ADMISSION_PASS 5roles,12coordinates,currentPRexact,0formalLRfits',flush=True)


def setup(root):
    binding=c.jr(root/'private/INPUT_FREEZE.json');assert binding['source']==c.sources()
    assert c.jr(root/'public/ADMISSION.json')['status']=='PASS'
    return c.load_data(),binding


def model_pack(root,key,bind,x,y,pid,C,pre,meta):
    path=root/'private/models'/(key+'.pkl')
    if path.exists():
        pack=c.read(path);assert pack['binding']==bind;return path,pack
    tick=time.perf_counter();model,state=c.fit_lr(x,y,pid,C)
    pack=dict(binding=bind,model=model,pre=pre,fit=state,meta=meta,seconds=time.perf_counter()-tick)
    c.save(path,pack);return path,pack


def train(root):
    d,binding=setup(root);pub=root/'public'
    commit=os.environ.get('PREDICTION_SOURCE_COMMIT','');assert re.fullmatch('[a-f0-9]{40}',commit),'Pre-fit source remote verification required'
    push=dict(commit=commit,source=c.sources(),input_freeze_sha256=c.sha(root/'private/INPUT_FREEZE.json'))
    pp=pub/'SOURCE_PUSH_BEFORE_FIT.json'
    if pp.exists():assert c.jr(pp)==push
    else:c.write(pp,push)
    selections=[];cv_rows=[];fit_audit=[];candidate_files={}
    freeze_path=root/'private/MODEL_SELECTION.pkl'
    if freeze_path.exists():
        assert len(c.read(freeze_path)['cells'])==570;return
    for f,rr in enumerate(d['roles'],1):
        tr=rr['fit'];cal=rr['validation'];xf=d['x'][tr];yf=d['y'][tr];pf=d['patient'][tr]
        splits=c.inner_splits(pf,f);full={}
        for pr in [True,False]:
            pre=c.fit_pre(xf,pf,pr)
            full[pr]=(pre,c.transform(xf,pf,pre),c.transform(d['x'][cal],d['patient'][cal],pre))
        for arm in c.ARMS:
            pr=not arm.endswith('_NO_PR');cols=c.columns(arm);scores={}
            for C in c.CS:
                oof=np.full(len(tr),np.nan)
                for j,(it,iv) in enumerate(splits):
                    pre=c.fit_pre(xf[it],pf[it],pr)
                    train_x=c.transform(xf[it],pf[it],pre)[:,cols]
                    held_x=c.transform(xf[iv],pf[iv],pre)[:,cols]
                    meta=dict(kind='INNER_CV',fold=f,arm=arm,C=C,inner_fold=j,permutation=-1)
                    bind=dict(input_sha=c.digest(binding),**meta,fit_indices_sha=c.array_sha(tr[it]),held_indices_sha=c.array_sha(tr[iv]))
                    key=f'cv_{f}_{arm}_c{C:g}_i{j}'
                    path,pack=model_pack(root,key,bind,train_x,yf[it],pf[it],C,pre,meta)
                    oof[iv]=c.predict(pack['model'],held_x)
                    candidate_files[key]=c.sha(path)
                    fit_audit.append(dict(**meta,**pack['fit'],seconds=pack['seconds'],warnings=';'.join(pack['fit']['warnings'])))
                scores[C]=c.patient_auc(yf,oof,pf)
                cv_rows.append(dict(fold=f,arm=arm,C=C,FIT_patients=len(set(pf)),OOF_patient_AUROC=scores[C]))
            best=c.choose_c(scores);pre,zf,zc=full[pr]
            meta=dict(kind='SELECTED',fold=f,arm=arm,C=best,inner_fold=-1,permutation=-1)
            bind=dict(input_sha=c.digest(binding),**meta,fit_indices_sha=c.array_sha(tr))
            path,pack=model_pack(root,f'selected_{f}_{arm}',bind,zf[:,cols],yf,pf,best,pre,meta)
            prob=c.predict(pack['model'],zc[:,cols]);choice=c.threshold(d['y'][cal],prob,d['patient'][cal])
            selections.append(dict(**meta,path=str(path),sha256=c.sha(path),**choice))
            fit_audit.append(dict(**meta,**pack['fit'],seconds=pack['seconds'],warnings=';'.join(pack['fit']['warnings'])))
            selected_count=sum(r['kind']=='SELECTED' for r in selections)
            c.write(pub/'TRAIN_STATUS.json',dict(status='TRAINING',selected_cells=selected_count,selected_total=70,
                                               completed_unique_LR_fits=len(fit_audit),formal_target=1410))
            print('SELECTED_LR_COMPLETE',f,arm,'C',best,'cells',selected_count,70,flush=True)
        ref=next(r for r in selections if r['fold']==f and r['arm']=='ALL_SIX_PR');pre,zf,zc=full[True]
        for b in range(100):
            yperm=c.permuted_labels(yf,pf,f,b)
            meta=dict(kind='PERMUTED',fold=f,arm='ALL_SIX_PR',C=ref['C'],inner_fold=-1,permutation=b)
            bind=dict(input_sha=c.digest(binding),**meta,fit_indices_sha=c.array_sha(tr),true_selected_model_sha=ref['sha256'])
            path,pack=model_pack(root,f'null_{f}_{b:03d}',bind,zf,yperm,pf,ref['C'],pre,meta)
            prob=c.predict(pack['model'],zc);choice=c.threshold(d['y'][cal],prob,d['patient'][cal])
            selections.append(dict(**meta,path=str(path),sha256=c.sha(path),**choice))
            fit_audit.append(dict(**meta,**pack['fit'],seconds=pack['seconds'],warnings=';'.join(pack['fit']['warnings'])))
            if b%20==19:print('NULL_LR_COMPLETE',f,b+1,100,flush=True)
    assert len(selections)==570 and len(fit_audit)==1410 and len(candidate_files)==840 and len(cv_rows)==280
    c.save(freeze_path,dict(binding=binding,cells=selections,candidates=candidate_files,cv_rows=cv_rows))
    table(pub/'inner_cv.csv',cv_rows);table(pub/'fit_audit.csv',fit_audit)
    table(pub/'model_selection.csv',[{**{k:v for k,v in r.items() if k not in ['path','sha256']},
           'model_sha256':r['sha256']} for r in selections if r['kind']=='SELECTED'])
    c.write(pub/'TRAIN_STATUS.json',dict(status='ALL1410_LR_FITS_COMPLETE_PENDING_FREEZE',main_and_supplementary=70,
        null_models=500,inner_models=840,actual_unique_LR_fits=1410))


def score(root):
    d,binding=setup(root);freeze=c.jr(root/'private/FROZEN_BEFORE_OUTER.json')
    assert freeze['input_sha']==c.digest(binding) and len(freeze['cells'])==570
    assert c.jr(root/'public/FREEZE_STATUS.json')['status']=='ALL570_FROZEN_BEFORE_OUTER'
    files={};fh=c.sha(root/'private/FROZEN_BEFORE_OUTER.json')
    for r in freeze['cells']:
        path=Path(r['path']);assert c.sha(path)==r['sha256'];pack=c.read(path)
        ix=d['roles'][r['fold']-1]['test'];cols=c.columns(r['arm'])
        key=f"{r['fold']}_{r['arm']}_{r['permutation']}";dest=root/'private/scores'/(key+'.pkl')
        if dest.exists():q=c.read(dest)
        else:
            z=c.transform(d['x'][ix],d['patient'][ix],pack['pre'])[:,cols]
            prob=c.predict(pack['model'],z)
            q=dict(freeze_sha256=fh,model_sha256=r['sha256'],indices=ix,probability_ez=prob,
                   prediction_ez=prob>=r['threshold'],threshold=r['threshold'],arm=r['arm'],fold=r['fold'],permutation=r['permutation'])
            c.save(dest,q)
        assert q['freeze_sha256']==fh and q['model_sha256']==r['sha256'] and np.array_equal(q['indices'],ix) and 'y' not in q
        files[key]=c.sha(dest)
    seal=root/'private/OUTER_SCORES_SEAL.json'
    assert len(files)==570
    if seal.exists():assert c.jr(seal)==dict(freeze_sha256=fh,files=files)
    else:c.write(seal,dict(freeze_sha256=fh,files=files))
    c.write(root/'public/PREDICTION_STATUS.json',dict(status='ALL570_SEALED_BEFORE_METRICS',cells=570,freeze_sha256=fh,seal_sha256=c.sha(seal)))
    print('ALL570_OUTER_SCORE_CELLS_SEALED',flush=True)


def package(root):
    from report import ALLOW
    import zipfile
    assert c.jr(root/'public/STATUS.json')['status']=='COMPLETE_VERIFIED_SIXFEATURE_EZ_PREDICTION'
    pub=root/'public';manifest=dict(schema='SIXFEATURE_EZ_PREDICTION_V1',files={n:c.sha(pub/n) for n in ALLOW},source=c.sources(),private_exported=False)
    c.write(pub/'PACKAGE_MANIFEST.json',manifest)
    path=root/'public_aggregate_export_v1.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n in ALLOW+['PACKAGE_MANIFEST.json']:z.write(pub/n,n)
    print('ANONYMOUS_PREDICTION_PACKAGE_SHA256',c.sha(path),path.stat().st_size,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['admission','train','freeze','score','report','verify','package']);p.add_argument('--runtime',type=Path,required=True)
    args=p.parse_args();c.cpu();args.runtime.mkdir(parents=True,exist_ok=True)
    if args.stage=='freeze':
        import verify;verify.internal(args.runtime)
    elif args.stage=='verify':
        import verify;verify.outer(args.runtime)
    elif args.stage=='report':
        import report;report.main(args.runtime)
    else:globals()[args.stage](args.runtime)
