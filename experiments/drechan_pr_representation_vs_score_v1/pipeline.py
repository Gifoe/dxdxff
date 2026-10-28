"""Admission and frozen execution stages for the representation/score contrast."""
import inspect
import shutil
from pathlib import Path
import numpy as np
from task_io import *

def probe():
    import torch
    from parents import Parents
    ROOT.mkdir(parents=True,exist_ok=True);PUB.mkdir(exist_ok=True)
    p=Parents()
    rows=[]
    for fold in range(1,6):
        d=p.bank(fold)
        rows.append(dict(fold=fold,fit=len(set(d['patient'][d['train']])),validation=len(set(d['patient'][d['val']])),test=len(d['roles']['test']),feature_shape=list(d['x'].shape)))
    p.verify_protected()
    record=dict(status='PASS',torch=torch.__version__,numpy=np.__version__,cuda=torch.cuda.is_available(),folds=rows,references=len(p.refs),parent_files=len(p.parent_protected),free_disk_gib=shutil.disk_usage(ROOT).free/1024**3)
    write(PUB/'PATH_PROBE.json',record)
    print(json.dumps(record),flush=True)
    print('PARENT_PROBABILITY_SOURCE',inspect.getsource(p.own.probability),flush=True)
    print('PARENT_THRESHOLD_SOURCE',inspect.getsource(p.core.select_threshold),flush=True)

def run(stage,commit):
    if stage=='probe':return probe()
    import torch
    from parents import Parents
    from admission import admit
    ROOT.mkdir(parents=True,exist_ok=True);PUB.mkdir(exist_ok=True)
    p=Parents();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if stage=='admit':return admit(p,commit)
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Formal execution requires frozen Git source commit'
    admission=jr(PUB/'ADMISSION.json');assert admission['status']=='PASS' and admission['source']==sources()
    lock_path=ROOT/'SOURCE_LOCK.json'
    if lock_path.exists():
        lock=jr(lock_path);assert lock['source']==sources() and lock['source_commit']==commit and lock['admission_sha256']==sha(PUB/'ADMISSION.json')
    else:
        lock=dict(source=sources(),source_commit=commit,admission_sha256=sha(PUB/'ADMISSION.json'),parent_hashes=p.parent_protected)
        write(lock_path,lock)
        write(PUB/'SOURCE_LOCK_PUBLIC.json',dict(source=sources(),source_commit=commit,admission_sha256=lock['admission_sha256']))
    binding=digest(lock)
    if stage in ['train','all']:train(p,binding)
    if stage in ['freeze','all']:freeze(p,lock)
    if stage in ['evaluate','all']:
        import evaluation
        evaluation.score(p);evaluation.endpoints(p)
        import reporting
        reporting.report(read(ROOT/'PATIENT_ENDPOINTS_PRIVATE.pkl'))
    if stage in ['verify','all']:
        from verify import verify
        verify(p)
    if stage in ['report','all']:
        from reporting import findings
        assert jr(PUB/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
        findings();p.verify_protected()
        write(PUB/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_PR_REPRESENTATION_VS_SCORE_3_MODELS_7_CONDITIONS_5_FOLDS',logical_cells=105,new_main_fits=40,new_svm_oof_fits=30,patients=80,channels=7635,source_commit=commit))
        from release_check import build_package
        path=build_package(ROOT)
        print('PR_REPR_COMPLETE_VERIFIED',sha(path),path.stat().st_size,flush=True)

def train(p,binding):
    import neural,svm
    from admission import post_selection
    from parents import RAW_ROOT
    selected_path=ROOT/'SELECTED_MODELS_PRIVATE.pkl'
    if selected_path.exists():
        prior=read(selected_path);assert prior['binding']==binding;selections=prior['cells']
    else:selections={}
    references=read(ROOT/'admission_private/REFERENCE_VALIDATION.pkl')
    for model in MODELS:
        for fold in range(1,6):
            for arm in ['NO_PR','FULL_PR']:
                r=p.ref(model,arm,fold);v=references[key(model,arm,fold)]
                vp=Path(r['validation']) if 'validation' in r else Path(r['checkpoint']).parent/'VALIDATION_PRIVATE.pkl'
                out=dict(model=model,condition=arm,fold=fold,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],validation=str(vp),validation_sha256=sha(vp),threshold=float(r['threshold']),orientation=v['orientation'],epoch=int(r.get('epoch',0)),reused=True)
                selections[key(model,arm,fold)]=out
            if model=='LINEAR_SVM':selections[key(model,'GLOBAL',fold)]=dict(selections[key(model,'NO_PR',fold)],condition='GLOBAL',alias='NO_PR')
            for arm in ['SCORE_CENTER','SCORE_Z']:
                out=post_selection(p,model,references[key(model,'NO_PR',fold)],arm,fold)
                r=selections[key(model,'NO_PR',fold)];out.update(checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],validation=r['validation'],validation_sha256=r['validation_sha256'])
                selections[key(model,arm,fold)]=out
    save(selected_path,dict(binding=binding,cells=selections))
    for model in ['LINEAR_SVM','DRECHANNET','TIMECONV']:
        for arm in (['CENTER','SCALE'] if model=='LINEAR_SVM' else ['GLOBAL','CENTER','SCALE']):
            for fold in range(1,6):
                k=key(model,arm,fold)
                if k not in selections:
                    r=svm.fit(p,arm,fold,binding) if model=='LINEAR_SVM' else neural.fit(p,model,arm,fold,binding)
                    selections[k]=r;save(selected_path,dict(binding=binding,cells=selections))
                completed=sum(not r.get('reused') and not r.get('postprocess') and not r.get('alias') for r in selections.values())
                status('MAIN_CELL_COMPLETE',completed_new_main_fits=completed,required_new_main_fits=40,selected_logical_cells=len(selections),logical_cells=105)
                print('PR_REPR_TRAIN_PROGRESS',completed,40,flush=True)
    assert len(selections)==105

def freeze(p,lock):
    q=read(ROOT/'SELECTED_MODELS_PRIVATE.pkl');assert q['binding']==digest(lock) and len(q['cells'])==105
    cells=list(q['cells'].values())
    for r in cells:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        if r.get('preprocessor'):assert sha(r['preprocessor'])==r['preprocessor_sha256']
    assert sum(r.get('reused',False) and not r.get('alias') for r in cells)==30
    assert sum(r.get('postprocess',False) for r in cells)==30
    obj=dict(source=sources(),source_commit=lock['source_commit'],source_lock_sha256=sha(ROOT/'SOURCE_LOCK.json'),cells=cells,all105_selections_frozen_before_new_test=True)
    target=ROOT/'FROZEN_BEFORE_TEST.json'
    if target.exists():assert jr(target)==obj
    else:write(target,obj)
    csv_out('MODEL_SELECTION',[dict(model=r['model'],condition=r['condition'],fold=r['fold'],threshold=r['threshold'],orientation=r['orientation'],epoch=r.get('epoch',0),reused=bool(r.get('reused')),alias=r.get('alias',''),postprocess=bool(r.get('postprocess')),checkpoint_sha256=r['checkpoint_sha256']) for r in cells])
    csv_out('REUSE_AUDIT',[dict(model=r['model'],condition=r['condition'],fold=r['fold'],alias=r.get('alias',''),checkpoint_sha256=r['checkpoint_sha256'],validation_sha256=r['validation_sha256']) for r in cells if r.get('reused')])
    p.verify_protected();write(PUB/'FREEZE_STATUS.json',dict(status='ALL105_SELECTIONS_FROZEN',logical_cells=105,freeze_sha256=sha(target),new_test_results_accessed=False))
    print('PR_REPR_ALL105_FROZEN',flush=True)
