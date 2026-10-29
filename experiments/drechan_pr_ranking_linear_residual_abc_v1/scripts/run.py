"""Reuse sealed scores; replay frozen linear heads; evaluate A-C without fitting."""
import argparse,copy,csv,hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
import trusted as t
import numpy as np
import torch
import analysis_math as m
import core_abc as core

ROOT=Path(r'D:\Codex_Research\experiments')
OLD=ROOT/'drechan_pr_ranking_robustness_v1_runtime'
RES=ROOT/'m8_raw_frozen_pr_linear_residual_seed42_v1_runtime'
MODELS=core.MODELS;ARMS=core.ARMS

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def unchanged(protected):
    for path,h in protected.items():assert t.sha(path)==h,'Frozen parent changed'

def admit(p):
    lock=t.jr(t.SOURCE/'PROTOCOL_LOCK.json')
    for rel,h in lock['source_pins'].items():assert t.sha(ROOT/Path(rel).relative_to('experiments'))==h
    assert t.jr(RES/'public/FINAL_STATUS.json')['status']=='FROZEN_PR_LINEAR_RESIDUAL_ALL_FIVE_OUTER_COMPLETE'
    assert t.jr(OLD/'public/ABC_VERIFICATION.json')['status']=='PASS'
    old=t.jr(OLD/'ABC_FREEZE_PRIVATE.json');seal=t.jr(OLD/'ABC_SCORE_SEAL_PRIVATE.json')['files']
    parent=t.jr(RES/'FROZEN_BEFORE_OUTER_PRIVATE.json');rs=t.jr(RES/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    assert rs['freeze_sha256']==t.sha(RES/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert parent['all20_frozen_before_residual_outer'] and rs['all320_residual_patients_sealed_before_full_labels']
    protected=dict(p.parent_protected)
    for name in ['INPUT_FREEZE_PRIVATE.json','FROZEN_BEFORE_OUTER_PRIVATE.json','OUTER_PREDICTIONS_SEAL_PRIVATE.json','public/FINAL_STATUS.json','public/main_results.csv','public/VALIDATION_SELECTION.csv']:
        path=RES/name;protected[str(path)]=t.sha(path)
    for name in ['ABC_FREEZE_PRIVATE.json','ABC_SCORE_SEAL_PRIVATE.json']:
        path=OLD/name;protected[str(path)]=t.sha(path)
    binding=t.jr(RES/'INPUT_FREEZE_PRIVATE.json')
    for name,h in binding['source'].items():assert t.sha(ROOT/'m8_raw_frozen_pr_linear_residual_seed42_v1'/name)==h
    for path,h in binding['protected'].items():assert t.sha(path)==h;protected[path]=h
    cells=copy.deepcopy(old['cells']);cache_count=0;maxerr=0.
    spec=importlib.util.spec_from_file_location('fixed_pr',ROOT/'m8_raw_backbones_pr_seed42_v1/patient_relative.py')
    pr=importlib.util.module_from_spec(spec);spec.loader.exec_module(pr)
    replacements={}
    for r in cells:
        key=t.frozen_io.key(r['model'],r['condition'],r['fold']);path=OLD/'private/scores'/(key+'.pkl')
        assert t.sha(path)==seal[key];protected[str(path)]=seal[key]
        if r['model']!='TIMECONV' or r['condition']!='FULL_PR':continue
        fold=r['fold'];sel=next(x for x in parent['selections'] if x['model']=='TimeConvCNN' and x['fold']==fold)
        original=next(x for x in cells if x['model']=='TIMECONV' and x['condition']=='NO_PR' and x['fold']==fold)
        assert sel['base_checkpoint_sha256']==original['checkpoint_sha256']
        assert t.sha(sel['checkpoint'])==sel['checkpoint_sha256'];protected[sel['checkpoint']]=sel['checkpoint_sha256']
        path=RES/'TimeConvCNN'/f'fold{fold}'/'OUTER_SCORES_PRIVATE.pkl'
        assert t.sha(path)==rs['files'][f'TimeConvCNN_{fold}'];protected[str(path)]=t.sha(path)
        q=t.read(path);assert q['freeze_sha256']==rs['freeze_sha256'] and q['checkpoint_sha256']==sel['checkpoint_sha256']
        cp=torch.load(sel['checkpoint'],map_location='cpu',weights_only=False)
        net=torch.nn.Linear(256,1);net.load_state_dict(cp['state']);net.eval().requires_grad_(False)
        no=t.read(OLD/'private/scores'/f'TIMECONV_NO_PR_fold{fold}.pkl')
        lookup={int(i):j for j,i in enumerate(q['indices'])};nlookup={int(i):j for j,i in enumerate(no['indices'])}
        all_indices=[];all_logits=[];all_probs=[]
        d=p.bank(fold)
        for pid in sorted(d['roles']['test']):
            token=hashlib.sha256(str(pid).encode()).hexdigest()[:24]
            path=RES/'TimeConvCNN'/f'fold{fold}'/'representations_private/test'/(token+'.pkl')
            protected[str(path)]=t.sha(path);c=t.read(path)
            assert not {'y','label','target'}&set(c) and c['role']=='test' and c['checkpoint_sha256']==sel['base_checkpoint_sha256']
            assert c['source_sha256']==parent['source_sha256']
            ix=c['indices'];expected=np.flatnonzero(p.meta['patient']==pid)
            assert set(ix.tolist())==set(expected.tolist())
            with torch.inference_mode():
                z=pr.patient_relative(torch.from_numpy(c['embedding']).float())
                nez=torch.from_numpy(c['logit_nez']).float()-net(z).flatten()
                probability=(1-torch.sigmoid(nez)).numpy();ezlogit=(-nez).numpy()
            e=float(np.max(np.abs(probability-q['score'][[lookup[int(i)] for i in ix]])))
            maxerr=max(maxerr,e);assert e<=lock['checkpoint_replay_probability_atol']
            assert np.array_equal(c['probability_ez'],no['ranking_anchor'][[nlookup[int(i)] for i in ix]])
            all_indices.extend(ix.tolist());all_logits.extend(ezlogit.tolist());all_probs.extend(probability.tolist());cache_count+=1
        assert set(all_indices)==set(q['indices'].tolist())
        lmap=dict(zip(all_indices,all_logits));replacements[fold]=dict(selection=sel,score_path=str(RES/'TimeConvCNN'/f'fold{fold}'/'OUTER_SCORES_PRIVATE.pkl'),logit_map=lmap)
        r.update(checkpoint=sel['checkpoint'],checkpoint_sha256=sel['checkpoint_sha256'],base_checkpoint_sha256=sel['base_checkpoint_sha256'],threshold=sel['threshold'],orientation='EZ_GE',validation=str(RES/'public/VALIDATION_SELECTION.csv'),validation_sha256=t.sha(RES/'public/VALIDATION_SELECTION.csv'),pr_implementation='FROZEN_BACKBONE_PR_LINEAR_RESIDUAL')
    assert cache_count==80 and len(cells)==45 and len(replacements)==5
    unchanged(protected)
    return cells,replacements,protected,dict(status='PASS',patients=80,channels=7635,logical_cells=45,new_fits=0,cached_fixed_head_replays=80,replay_max_absolute_probability_error=maxerr,base_checkpoint_match_all5=True,unchanged_score_cells=40,main_table_residual_score_cells=5,parent_inputs_unchanged=True)

def freeze_and_seal(p,commit,cells,replacements,protected,admission):
    fr=dict(source=t.source_hashes(),source_commit=commit,cells=cells,protected=protected,exploratory=True,admission=admission)
    fpath=t.RUNTIME/'ABC_FREEZE_PRIVATE.json';assert not fpath.exists(),'Unique runtime required'
    t.write(fpath,fr);t.write(t.PUB/'ADMISSION.json',admission)
    t.write(t.PUB/'manifest.json',dict(source_commit=commit,source=fr['source'],patients=80,channels_per_condition=7635,logical_cells=45,new_fits=0,reused_old_abc_cells=40,reused_main_table_residual_cells=5,exploratory=True,timeconv_version='FROZEN_BACKBONE_PR_LINEAR_RESIDUAL',scope='A/B/C',D='UNCHANGED_IN_ORIGINAL_STUDY',checkpoint_audit=[{k:r[k] for k in ['model','condition','fold','checkpoint_sha256','validation_sha256']} for r in cells],parent_residual_freeze_sha256=t.sha(RES/'FROZEN_BEFORE_OUTER_PRIVATE.json'),parent_residual_score_seal_sha256=t.sha(RES/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    sealed={}
    for r in cells:
        key=t.frozen_io.key(r['model'],r['condition'],r['fold']);q=copy.deepcopy(t.read(OLD/'private/scores'/(key+'.pkl')))
        if r['model']=='TIMECONV' and r['condition']=='FULL_PR':
            rep=replacements[r['fold']];old=t.read(rep['score_path']);ix=q['indices'];mapping={int(i):j for j,i in enumerate(old['indices'])}
            assert set(mapping)==set(ix.tolist())
            q['ranking_anchor']=old['score'][[mapping[int(i)] for i in ix]]
            q['prediction_ez']=q['ranking_anchor']>=r['threshold']
            q['logit_ez']=np.array([rep['logit_map'][int(i)] for i in ix])
            q['provenance']='SEALED_MAIN_TABLE_PR_LINEAR_RESIDUAL';q['checkpoint_sha256']=r['checkpoint_sha256']
        q['freeze_sha256']=t.sha(fpath);path=t.RUNTIME/'private/scores'/(key+'.pkl');t.save(path,q);sealed[key]=t.sha(path)
    assert len(sealed)==45
    t.write(t.RUNTIME/'ABC_SCORE_SEAL_PRIVATE.json',dict(files=sealed,all45_before_metrics=True))
    print('ALL45_FROZEN_AND_SEALED_NO_TRAINING',flush=True);return fr,sealed

def verify_results(patient_rows,budgets,curves):
    old=rows(OLD/'private/patient_endpoints.csv');lookup={(r['model'],r['condition'],r['patient']):r for r in old}
    unchanged_rows=0
    for r in patient_rows:
        if (r['model'],r['condition'])==('TIMECONV','FULL_PR'):continue
        previous=lookup[r['model'],r['condition'],r['patient']]
        assert all(abs(r[k]-float(previous[k]))<1e-12 for k in m.METRICS);unchanged_rows+=1
    anchors=rows(RES/'public/main_results.csv')
    for arm,version in [('NO_PR','Original'),('FULL_PR','PRLinearResidual')]:
        ref=next(r for r in anchors if r['model']=='TimeConvCNN' and r['version']==version and r['unit']=='patient_macro')
        rr=[r for r in patient_rows if r['model']=='TIMECONV' and r['condition']==arm]
        assert len(rr)==80 and sum(r['channels'] for r in rr)==7635
        for k in m.METRICS:assert abs(np.mean([r[k] for r in rr])-float(ref[k]))<1e-12,k
    # Independent source-stratified index resampling checks every reported interval.
    ids=sorted({r['patient'] for r in patient_rows});look={(r['model'],r['condition'],r['patient']):r for r in patient_rows}
    source=np.array([look['TIMECONV','NO_PR',i]['source'] for i in ids]);rng=np.random.default_rng(42)
    ix=np.concatenate([np.flatnonzero(source==s)[rng.integers(0,int(np.sum(source==s)),size=(10000,int(np.sum(source==s))))] for s in m.SOURCES],axis=1)
    def check(v,out):
        bs=np.array([v[ix[k:k+200]].mean(1) for k in range(0,10000,200)]).ravel()
        assert abs(float(out['mean'])-float(v.mean()))<1e-12
        assert np.max(np.abs(np.quantile(bs,[.025,.975])-[float(out['low']),float(out['high'])]))<1e-12
        fam=int(out.get('family_size',0))
        if fam:assert np.max(np.abs(np.quantile(bs,[.05/(2*fam),1-.05/(2*fam)])-[float(out['adjusted_low']),float(out['adjusted_high'])]))<1e-12
    count=0
    for model in MODELS:
        arrays={a:np.array([[look[model,a,i][v] for v in m.METRICS] for i in ids]) for a in ARMS}
        for out in rows(t.PUB/'TABLE_A.csv'):
            if out['model']==model:check(arrays[out['condition']][:,m.METRICS.index(out['metric'])],out);count+=1
        for out in rows(t.PUB/'TABLE_B.csv'):
            if out['model']==model:check((arrays['FULL_PR']-arrays[out['contrast'].replace('FULL_MINUS_','')])[:,m.METRICS.index(out['metric'])],out);count+=1
        bl={(r['condition'],r['patient'],r['budget_kind'],r['budget']):r for r in budgets if r['model']==model}
        for out in rows(t.PUB/'TABLE_D_REVIEW.csv'):
            if out['model']==model:
                v=np.array([bl[out['condition'],i,out['budget_kind'],float(out['budget'])][out['metric']] for i in ids]);check(v,out);count+=1
        for out in rows(t.PUB/'TABLE_D_PAIRED.csv'):
            if out['model']==model:
                v=np.array([bl['FULL_PR',i,out['budget_kind'],float(out['budget'])][out['metric']]-bl['NO_PR',i,out['budget_kind'],float(out['budget'])][out['metric']] for i in ids]);check(v,out);count+=1
        for arm in ['NO_PR','FULL_PR']:
            rr={r['patient']:r['recall'] for r in curves if r['model']==model and r['condition']==arm}
            for out in rows(t.PUB/'COVERAGE_CURVE.csv'):
                if out['model']==model and out['condition']==arm:
                    j=round(float(out['review_fraction'])*100);check(np.array([rr[i][j] for i in ids]),out);count+=1
    assert unchanged_rows==640
    return dict(main_table_all11_metrics_match=True,unchanged_patient_rows_exact=unchanged_rows,independently_recomputed_intervals=count)

def explicit_public_labels():
    for path in t.PUB.glob('*.csv'):
        rr=rows(path)
        for r in rr:
            if r.get('model')=='TIMECONV':
                if r.get('condition')=='FULL_PR':r['condition']='PR_LINEAR_RESIDUAL'
                if r.get('contrast','').startswith('FULL_MINUS_'):r['contrast']=r['contrast'].replace('FULL_MINUS_','PR_LINEAR_RESIDUAL_MINUS_')
        t.csv_out(path.stem,rr)
    t.write(t.PUB/'CONDITION_MAP.json',dict(TimeConv=dict(internal_FULL_PR='PR_LINEAR_RESIDUAL',public_PR_LINEAR_RESIDUAL='Frozen original backbone plus original PR Linear Residual; main-table match'),SVM='Unchanged Full PR',DREChanNet='Unchanged Full PR'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True);args=parser.parse_args();assert len(args.source_commit)==40
    t.RUNTIME.mkdir(parents=True,exist_ok=True);t.PUB.mkdir(exist_ok=True)
    p=t.parents();cells,replacements,protected,admission=admit(p)
    fr,sealed=freeze_and_seal(p,args.source_commit,cells,replacements,protected,admission)
    rr,b,c=core.endpoints(p,fr,sealed);counts=core.summary(rr,b,c);verification=verify_results(rr,b,c)
    unchanged(protected);p.verify_protected()
    t.write(t.PUB/'ABC_VERIFICATION.json',dict(status='PASS',patients=80,channels_per_condition=7635,logical_cells=45,ranking_invariance='EXACT',independent_endpoint_formulas=True,independent_direct_index_bootstrap=True,frozen_parents_unchanged=True,**counts,**verification))
    explicit_public_labels()
    t.write(t.PUB/'STATUS.json',dict(status='ABC_PR_LINEAR_RESIDUAL_COMPLETE_VERIFIED',A='COMPLETE',B='COMPLETE',C='COMPLETE',new_fits=0,TimeConv_PR='PR_LINEAR_RESIDUAL',D='UNCHANGED_IN_ORIGINAL_STUDY'))
    files=sorted(t.PUB.iterdir());manifest={x.name:dict(sha256=t.sha(x),bytes=x.stat().st_size) for x in files}
    target=t.RUNTIME/'public_abc_linear_residual_export_v1.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for x in files:z.write(x,'public/'+x.name)
        z.writestr('PUBLIC_MANIFEST.json',json.dumps(dict(files=manifest,private_data_included=False),sort_keys=True))
    print('ABC_PR_LINEAR_RESIDUAL_COMPLETE_VERIFIED',t.sha(target),flush=True)

if __name__=='__main__':main()
