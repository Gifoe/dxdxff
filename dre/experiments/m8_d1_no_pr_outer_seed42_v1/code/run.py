"""Matched M8-no-PR training and one sealed Outer pass; no raw extraction."""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parents[1]
BASELINE = Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
NOPR = Path(r'D:\d1_m8_progressive_ablation_seed42_v1_runtime')
OLD = Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime')
M8 = Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime')
P1 = Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')
DATA = Path(r'D:\dgm_net_seed42_v1_runtime')
SOURCE = Path(r'E:\DRE-nips\new-pipeline\7-11')
P3CODE = Path(r'D:\d1_small_module_exploration_seed42_v1\phase3\code')
sys.path.insert(0, str(SOURCE/'raw_delta_encoder_seed42_v1/code'))
from io_utils import sha, read, save, write, load
SUP = load('no_pr_upstream_support', Path(r'D:\d1_m8_progressive_ablation_seed42_v1\code\support.py'))
PRE = load('no_pr_preprocessing', Path(r'D:\d1_m8_progressive_ablation_seed42_v1\code\preprocessing.py'))
sys.path.insert(0,str(P3CODE))
P3 = load('original_m8_fit_for_no_pr', P3CODE/'run.py')
M8NoPR = load('matched_m8_no_pr_model',ROOT/'code/model.py').M8NoPR

METRICS = [v for v in SUP.METRICS if v != 'balanced_accuracy']


def jr(p):
    return json.loads(Path(p).read_text())


def setup(runtime):
    a = SimpleNamespace(runtime=NOPR, baseline=BASELINE, m8=M8, old_outer=OLD, source=SOURCE)
    _, c, core, fac, trainer, records, rec, banks, upstream = SUP.setup(a)
    expected = {
        NOPR/'MODEL_FREEZE_PRIVATE.json':'03e481b0d188be2db780cd67352db54576c74beff00be5081ebd3291af74144a',
        NOPR/'FROZEN_BEFORE_OUTER_PRIVATE.json':'53afbec76d6e9837b40a4435fbcc65159298ee1851eac33aa3b3a7bc86950510',
        NOPR/'OUTER_PREDICTIONS_SEAL_PRIVATE.json':'9a8cd7d353092c51e2b628c4f42f9adfd29bc2b054d6b7eb8488a656329ee17d',
        OLD/'FROZEN_BEFORE_OUTER_PRIVATE.json':'4882d62e0492618a277897e56e434578b8481b7a8192213d01cd3a5b5f420d5d',
        OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json':'aa5ae8daeec58fb392e6587a89f24cb26ce34fc83a1125f8f6ae4a575ab81e02',
        DATA/'CONDITIONING_REGISTRY_PRIVATE.pkl':'43cb64729c038e08ce16434d5a50e77cc4f3864f8895059929877cb48b1b8f0e'}
    for p, h in expected.items():
        assert sha(p) == h, p.name
    assert jr(NOPR/'results/INDEPENDENT_VERIFICATION.json')['status'] == 'PASS'
    assert jr(OLD/'public/OUTER_INDEPENDENT_VERIFICATION.json')['status'] == 'PASS'
    features = read(NOPR/'FEATURES_PRIVATE.pkl')
    assert np.array_equal(features['DELTA14'], rec['R14'])
    originals = SUP.references(a, core, banks)
    nofr = jr(NOPR/'MODEL_FREEZE_PRIVATE.json')
    refs = []
    for fold, original in enumerate(banks, 1):
        r = next(q for q in nofr['selections'] if q['arm']=='A1' and q['fold']==fold)
        cp = NOPR/f'A1/fold{fold}/TRAIN/BEST_PRIVATE.pt'
        bp = cp.parent.parent/'BANK_PRIVATE.pt'
        assert sha(cp)==r['checkpoint_sha256'] and sha(bp)==r['bank_sha256']
        d = torch.load(bp, map_location='cpu', weights_only=False)
        assert d['pre']['patient_z'] is False
        assert set(d['pre']['fit_patient_ids']) == original['roles']['fit']
        assert set(d['patient']).isdisjoint(original['roles']['test'])
        assert np.array_equal(d['archive_indices'], original['archive_indices'])
        assert np.array_equal(d['x'], PRE.transform(rec['R14'][d['archive_indices']], d['pre']))
        sq = read(P1/f'scale_fold{fold}_PRIVATE.pkl')
        assert set(sq['fit_patients']) == original['roles']['fit']
        assert sha(P1/f'scale_fold{fold}_PRIVATE.pkl') == jr(M8/'INPUT_FREEZE_PRIVATE.json')['phase1_scales'][str(fold)]
        refs.append(dict(r, arm='D1_NO_PR', checkpoint=str(cp), bank=str(bp)))
    for r in originals:
        refs.append(dict(r, arm='D1_PR' if r['old_arm']=='D1' else 'M8_PR'))
    files = {f'new/{p.relative_to(ROOT).as_posix()}':sha(p) for p in [ROOT/'PROTOCOL_LOCK.json', *sorted((ROOT/'code').glob('*.py'))]}
    for folder, label in [(P3CODE,'original_m8'),(P3CODE.parent.parent/'phase2/code','original_phase2')]:
        files.update({f'{label}/{p.name}':sha(p) for p in sorted(folder.glob('*.py'))})
    files.update({f'upstream/{i}':h for i,h in enumerate(expected.values())})
    binding = dict(files=files, upstream=upstream, torch=torch.__version__, numpy=np.__version__)
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime/'results').mkdir(exist_ok=True)
    ip = runtime/'INPUT_FREEZE_PRIVATE.json'
    if ip.exists(): assert jr(ip)==binding
    else: write(ip,binding)
    return c,core,fac,records,rec,banks,refs,binding


def make(core, fac, records, cp, fold, pr=False):
    return M8NoPR(P3.make_base(core,fac,records,cp,42+1009*fold),pr).cuda()


def public(r):
    return {k:v for k,v in r.items() if k not in ['checkpoint','bank','old_arm']}


def train(runtime):
    c,core,fac,records,rec,banks,refs,binding=setup(runtime)
    pub=runtime/'results'
    store=P3.Inputs(BASELINE,DATA,runtime);store.arm='M1_SCALED_DGM'
    audits=[];selections=[];histories=[];rows=[]
    for r in [v for v in refs if v['arm']=='D1_NO_PR']:
        fold=r['fold'];cp=Path(r['checkpoint']);d=torch.load(r['bank'],map_location='cpu',weights_only=False)
        scale=torch.tensor(read(P1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
        # Exact original constructor/RNG consumption, excluding deliberately different base weights.
        oldcp=BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt'
        old=P3.make(core,fac,records,oldcp,42+1009*fold,'M8_MACRO_AWARE')
        originalck=torch.load(M8/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert core.state_hash(old.state_dict())==originalck['binding']['initial']
        model=make(core,fac,records,cp,fold)
        initial=core.state_hash(model.state_dict())
        assert core.state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('base.')}) == core.state_hash({k:v for k,v in old.state_dict().items() if not k.startswith('base.')})
        del originalck,old
        va=d['val'];pid=sorted(set(d['patient'][va]))[0];ii=va[d['patient'][va]==pid]
        raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
        model.eval()
        with torch.inference_mode():
            expected=P3.MorphCorrection.features(model,raw,mask,times,delta,scale)
            model.patient_relative=True
            actual=model.features(raw,mask,times,delta,scale)
            parity=float((expected[0]-actual[0]).abs().max().cpu());assert parity==0
            model.patient_relative=False
            whole=model.features(raw,mask,times,delta,scale)[0]
            subset=model.features(raw[:,:-1],mask[:,:-1],times,delta[:,:-1],scale)[0]
            independence=float((subset-whole[:-1]).abs().max().cpu());assert independence<1e-5
        del raw,mask,times,delta,perms
        smoke=P3.admission(model,d,store,scale,core,'M8_MACRO_AWARE')
        audits.append(dict(fold=fold,original_PR_feature_parity_max=parity,no_PR_channel_subset_max=independence,original_correction_initialization_exact=True,**smoke))
        del model
        model=make(core,fac,records,cp,fold)
        assert core.state_hash(model.state_dict())==initial
        best,h=P3.fit(model,d,store,scale,runtime/f'M8_NO_PR/fold{fold}/TRAIN',42+1009*fold,binding,core,runtime,'M8_MACRO_AWARE',fold)
        cpp=runtime/f'M8_NO_PR/fold{fold}/TRAIN/BEST_PRIVATE.pt'
        rr=dict(arm='M8_NO_PR',fold=fold,checkpoint=str(cpp),checkpoint_sha256=sha(cpp),bank=r['bank'],bank_sha256=sha(r['bank']),base_checkpoint_sha256=sha(cp),threshold=best['threshold'],epoch=best['epoch'],epochs_run=len(h),initial_state_sha256=initial)
        selections.append(rr);histories.extend(h)
        pred=P3.EV.predict(model,d,va,store,scale)
        save(runtime/f'development_private/M8_NO_PR_fold{fold}.pkl',dict(logit=pred['logit'],indices=d['archive_indices'][va],checkpoint_sha256=sha(cpp),threshold=best['threshold']))
        score=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy()
        rows.extend(SUP.EV.patient_rows(c,d,score,best['threshold'],'M8_NO_PR',fold))
        write(pub/'MODEL_MATH_AND_GRADIENT_AUDIT.json',dict(status='PASS_SO_FAR',folds=audits))
        pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False)
        pd.DataFrame([public(q) for q in selections+refs]).to_csv(pub/'MODEL_SELECTION.csv',index=False)
        print('MATCHED_NO_PR_FOLD_COMPLETE',fold,'selected',best['epoch'],flush=True)
        del model,d;gc.collect()
    assert len(selections)==5
    # Reuse the three previously frozen Development ledgers, not a new fit.
    progressive=pd.DataFrame(read(NOPR/'DEVELOPMENT_METRICS_PRIVATE.pkl'))
    for arm,oldarm in [('D1_NO_PR','A1'),('D1_PR','A2'),('M8_PR','A3')]:
        q=progressive[progressive.arm==oldarm].copy();q['arm']=arm;rows.extend(q.to_dict('records'))
    df=pd.DataFrame(rows);save(runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl',rows)
    summaries(df,pub,'DEVELOPMENT')
    write(pub/'MODEL_MATH_AND_GRADIENT_AUDIT.json',dict(status='PASS',folds=audits,trainable_parameters=9481,frozen_base_parameters=1537,both_cross_channel_PR_sites_disabled=True))
    write(runtime/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selections,references=refs))
    write(pub/'RUN_STATUS.json',dict(status='DEVELOPMENT_COMPLETE',new_cells=5,new_epochs=len(histories)))
    print('ALL_FIVE_M8_NO_PR_SELECTED',flush=True)


def summaries(df,pub,scope):
    df.groupby('arm')[METRICS].mean().reset_index().to_csv(pub/f'{scope}_SUMMARY.csv',index=False)
    for key in ['fold','center']:
        rs=[dict(arm=arm,**{key:v},patient_appearances=len(g),unique_patients=g.patient.nunique(),channels=int(g.channels.sum()),**g[METRICS].mean().to_dict(),**g[['TP','FP','TN','FN']].sum().to_dict()) for (arm,v),g in df.groupby(['arm',key])]
        pd.DataFrame(rs).to_csv(pub/f'{scope}_BY_{key.upper()}.csv',index=False)


def evaluate(runtime):
    c,core,fac,records,rec,banks,refs,binding=setup(runtime)
    pub=runtime/'results';mf=jr(runtime/'MODEL_FREEZE_PRIVATE.json');assert mf['binding']==binding and mf['references']==refs
    selections=mf['selections']+refs
    for r in selections:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        if 'bank' in r: assert sha(r['bank'])==r['bank_sha256']
    fp=runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    obj=dict(binding=binding,selections=selections,new_training_cells=5,new_outer_label_read=False)
    if fp.exists():assert jr(fp)==obj
    else:write(fp,obj)
    fh=sha(fp)
    write(pub/'OUTER_FREEZE_AUDIT.json',dict(status='PASS',freeze_sha256=fh,selections=[public(v) for v in selections],all_five_new_selections_and_thresholds_frozen=True,own_fold_TEST_excluded=True))
    if (pub/'OUTER_STATUS.json').exists():
        assert jr(pub/'OUTER_STATUS.json')['freeze_sha256']==fh
        print('OUTER_ALREADY_COMPLETE_NO_NEW_INFERENCE',flush=True);return
    meta=rec['meta'];store=P3.Inputs(BASELINE,DATA,runtime);store.arm='M1_SCALED_DGM'
    cells={};files={};drifts=[]
    for r in selections:
        arm,fold=r['arm'],r['fold'];ix=np.flatnonzero(np.isin(meta['patient'],list(SUP.roles(meta,fold)['test'])))
        pp=runtime/f'outer_scores_private/{arm}_fold{fold}.pkl'
        if pp.exists():q=read(pp)
        elif arm=='M8_NO_PR':
            bd=torch.load(r['bank'],map_location='cpu',weights_only=False)
            d=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],x=PRE.transform(rec['R14'][ix],bd['pre']))
            basecp=NOPR/f'A1/fold{fold}/TRAIN/BEST_PRIVATE.pt'
            model=make(core,fac,records,basecp,fold).eval()
            ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
            scale=torch.tensor(read(P1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
            pred=P3.EV.predict(model,d,np.arange(len(ix)),store,scale)
            oldbase=read(NOPR/f'outer_scores_private/A1_fold{fold}.pkl')
            assert np.array_equal(oldbase['indices'],ix)
            err=float(np.max(np.abs(oldbase['logit']-pred['base_logit'])));assert err<=1e-6
            assert np.max(np.abs(pred['logit']-pred['base_logit']-pred['residual']))<=1e-6
            assert np.max(np.abs(pred['residual']))<=1+1e-6
            assert core.state_hash(model.state_dict())==core.state_hash(ck['model'])
            drifts.append(dict(fold=fold,embedded_no_PR_D1_logit_max_error=err,residual_cap_verified=True))
            q=dict(logit=pred['logit'],base_logit=pred['base_logit'],residual=pred['residual'],indices=ix,checkpoint_sha256=r['checkpoint_sha256'],freeze_sha256=fh)
            save(pp,q);del model,bd,ck;gc.collect()
        else:
            root=NOPR if arm=='D1_NO_PR' else OLD
            oldarm={'D1_NO_PR':'A1','D1_PR':'D1','M8_PR':'M8_MACRO_AWARE'}[arm]
            op=root/(f'outer_scores_private/{oldarm}_fold{fold}.pkl' if arm=='D1_NO_PR' else f'scores_private/{oldarm}_fold{fold}.pkl')
            seal=jr(root/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
            assert sha(op)==seal['files'][f'{oldarm}_{fold}']
            oldq=read(op);assert np.array_equal(oldq['indices'],ix)
            assert oldq['checkpoint_sha256']==r['checkpoint_sha256'] and oldq['freeze_sha256']==seal['freeze_sha256']
            q=dict(logit=oldq['logit'],indices=ix,checkpoint_sha256=r['checkpoint_sha256'],freeze_sha256=fh);save(pp,q)
        assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(q['indices'],ix)
        assert 'y' not in q and np.isfinite(q['logit']).all()
        files[f'{arm}_{fold}']=sha(pp);cells[arm,fold]=q
        print('OUTER_CELL_SEALED',arm,fold,len(ix),flush=True)
    sp=runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';seal=dict(freeze_sha256=fh,files=files,all_20_cells_sealed_before_full_label_read=True)
    if sp.exists():assert jr(sp)==seal
    else:write(sp,seal)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    assert len(y)==7635 and (y==0).sum()==1743
    rows=[]
    for r in selections:
        arm,fold=r['arm'],r['fold'];q=cells[arm,fold];ix=q['indices'];score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
        d=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],val=np.arange(len(ix)))
        rows.extend(SUP.EV.patient_rows(c,d,score,r['threshold'],arm,fold))
    df=pd.DataFrame(rows)
    for arm,g in df.groupby('arm'): assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
    previous=pd.DataFrame(read(NOPR/'OUTER_METRICS_PRIVATE.pkl'))
    for arm,oldarm in [('D1_NO_PR','A1'),('D1_PR','A2'),('M8_PR','A3')]:
        l=df[df.arm==arm].set_index(['fold','patient']).sort_index();rr=previous[previous.arm==oldarm].set_index(['fold','patient']).sort_index()
        assert l.index.equals(rr.index) and np.array_equal(l[METRICS+['TP','FP','TN','FN']].to_numpy(),rr[METRICS+['TP','FP','TN','FN']].to_numpy())
    save(runtime/'OUTER_METRICS_PRIVATE.pkl',rows);summaries(df,pub,'OUTER')
    boot=[]
    for left,right in [('M8_NO_PR','M8_PR'),('D1_NO_PR','D1_PR'),('M8_NO_PR','D1_NO_PR'),('M8_PR','D1_PR')]:
        boot.extend(dict(comparison=f'{left} minus {right}',**v) for v in SUP.BOOT.compare(c,df[df.arm==left],df[df.arm==right],METRICS))
    pd.DataFrame(boot).to_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False)
    conf=[dict(arm=arm,patients=len(g),channels=int(g.channels.sum()),**g[['TP','FP','TN','FN']].sum().to_dict()) for arm,g in df.groupby('arm')]
    pd.DataFrame(conf).to_csv(pub/'OUTER_CONFUSION_TOTALS.csv',index=False)
    write(pub/'OUTER_BASE_IDENTITY_AUDIT.json',dict(status='PASS',folds=drifts))
    write(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',freeze_sha256=fh,seal_sha256=sha(sp),patients_per_arm=80,channels_per_arm=7635,new_cells=5,reused_cells=15,all_predictions_sealed_before_full_labels=True,reference_metrics_exact=True,scientific_status='EXPLORATORY_REPEATED_TEST'))
    print('FOUR_ARM_OUTER_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','evaluate']);p.add_argument('--runtime',type=Path,default=Path(r'D:\m8_d1_no_pr_outer_seed42_v1_runtime'));a=p.parse_args()
    (train if a.phase=='train' else evaluate)(a.runtime)
