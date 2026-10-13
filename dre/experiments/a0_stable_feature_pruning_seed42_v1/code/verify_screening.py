"""Post-run independent score-only screening and artifact-integrity audit.

No model training, inference, threshold search or new feature selection. This
verifier is added after execution; it checks the original executed source hashes
recorded before screening and never alters that freeze or the original outputs.
"""
import argparse
import json
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args()
    a.original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime');a.previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime');a.features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    dep=load('verify_dependencies',a.source/'a0_cove_seed42_v1/code/dependencies.py');ref,c,core,orig,trainer,helper,banks,base=dep.dependencies(a)
    root=Path(__file__).parents[1];fr=json.loads((a.runtime/'INPUT_FREEZE_PRIVATE.json').read_text());bind=fr['binding']
    assert bind['authoritative']==base
    for name,digest in bind['new'].items():assert c.sha(root/name)==digest
    target=a.runtime/'PREPARED_PRIVATE.pt';assert c.sha(target)==json.loads(target.with_suffix('.sha256.json').read_text())['sha256'];z=torch.load(target,weights_only=False);assert z['binding']==bind
    fm=load('verify_folding',root/'code/folding.py');mp=fm.mapping(json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text())['columns']);assert mp==z['mapping']
    metric=load('verify_independent_metrics',a.source/'a0_cove_seed42_v1/code/metrics.py');names=['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr','top1_is_ez','sensitivity','specificity','nez_f1','balanced_accuracy','accuracy']
    recs=[];cells=0;checked=0;files={};allfolds=[]
    for fold,b in enumerate(z['banks'],1):
        original=banks[fold-1];assert np.array_equal(b['raw_x'],original['raw_x'])
        tr=b['train'];pp=b['patient'][tr];y=b['y'][tr];channel=b['channel'][tr];raw=b['raw_x'][tr];seen=[]
        for h,plan in enumerate(b['plans']):
            train=np.flatnonzero(np.isin(pp,plan['train']));select=np.flatnonzero(np.isin(pp,plan['select']));held=np.flatnonzero(np.isin(pp,plan['screen']));seen.extend(plan['screen'].tolist())
            assert set(plan['train']).isdisjoint(plan['screen']) and set(plan['select']).isdisjoint(plan['screen'])
            assert set(plan['train']).isdisjoint(plan['select']) and set(np.r_[train,select,held])==set(range(len(tr)))
            assert set(pp).isdisjoint(set(b['patient'][b['val']]))
            xx,pre=core.prepare(raw,pp,train);assert set(pre['fit_patient_ids'])==set(plan['train'])
            full=None
            for group in [-1]+list(range(21)):
                cell=a.runtime/f'screen/fold{fold}/inner{h}/group{group}';scorepath=cell/'SCREEN_PRIVATE.pt';q=torch.load(scorepath,weights_only=False);files[f'screen_{fold}_{h}_{group}']=c.sha(scorepath)
                assert q['binding']==bind and q['checkpoint_sha256']==c.sha(cell/'BEST_PRIVATE.pt') and np.array_equal(q['indices'],held)
                for k in ['imputer_statistics','mean','scale','var']:assert np.array_equal(pre[k],q['preprocessor'][k])
                assert pre['fit_patient_ids']==q['preprocessor']['fit_patient_ids']
                cp=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False);last=torch.load(cell/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False);done=json.loads((cell/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==c.sha(cell/'BEST_PRIVATE.pt') and done['last_sha256']==c.sha(cell/'LAST_PRIVATE.pt')
                assert cp['threshold']==q['threshold'] and cp['binding']['seed']==plan['seed'] and cp['binding']['penalty']==0
                cand=max(last['history'],key=lambda r:(r['macro_f1'],r['ez_f1'],-r['epoch']));assert cand['epoch']==cp['epoch']
                vals=[]
                for pid in sorted(set(pp[held])):
                    k=np.flatnonzero(pp[held]==pid);k=k[np.argsort(channel[held][k].astype(str),kind='stable')]
                    v=metric.independent(y[held[k]],q['scores'][k],q['threshold']);stored=next(r for r in q['rows'] if r['patient']==pid)
                    for m in names:assert abs(v[m]-stored[m])<1e-12
                    vals.append(v);checked+=1
                avg=pd.DataFrame(vals)[names].mean()
                if group==-1:full=avg
                else:recs.append(dict(fold=fold,inner=h,group=group,**{'delta_'+m:float(avg[m]-full[m]) for m in names}))
                cells+=1
        assert len(seen)==len(set(seen))==len(set(pp)) and set(seen)==set(pp)
    assert cells==330 and len(recs)==315
    actual=pd.DataFrame(recs).set_index(['fold','inner','group']).sort_index();pub=pd.read_csv(a.runtime/'public/FEATURE_IMPORTANCE_BY_INNER_FOLD.csv').set_index(['fold','inner','group']).sort_index()
    assert actual.index.equals(pub.index)
    delta_names=['delta_'+m for m in names];assert np.allclose(actual[delta_names],pub[delta_names],atol=1e-12,rtol=0)
    sf=json.loads((a.runtime/'FEATURE_SELECTION_PRIVATE.json').read_text());assert sf['binding']==bind;selpub=pd.read_csv(a.runtime/'public/FEATURE_SELECTION_BY_OUTER_FOLD.csv')
    for fold in range(1,6):
        eligible=[]
        for group in range(21):
            q=actual.loc[fold].loc[(slice(None),group),:];avg=q.mean();good=avg.delta_ez_auroc>=.002 and avg.delta_ez_auprc>=0 and avg.delta_macro_f1>=-.002 and int((q.delta_ez_auroc>0).sum())>=2
            if good:eligible.append((group,float(avg.delta_ez_auroc),float(avg.delta_ez_auprc)))
            row=selpub[(selpub.fold==fold)&(selpub.group==group)].iloc[0];assert bool(row.eligible)==good
        selected=[g for g,auc,ap in sorted(eligible,key=lambda q:(-q[1],-q[2],q[0]))[:3]];rnd=sorted(np.random.default_rng(42+1009*fold+20000).choice(21,size=len(selected),replace=False).tolist());stored=next(q for q in sf['selections'] if q['fold']==fold)
        assert selected==stored['selected_groups'] and rnd==stored['random_groups'];allfolds.append(dict(fold=fold,selected_count=len(selected),selected_groups=selected,random_groups=rnd))
    # Seal the private screening score files without exporting any identities.
    c.write_json(a.runtime/'SCREEN_VERIFICATION_SEAL_PRIVATE.json',dict(binding=bind,files=files))
    c.write_json(a.runtime/'public/SCREENING_INDEPENDENT_VERIFICATION.json',dict(status='PASS',cells=cells,patient_metric_cells=checked,group_screen_rows=315,original_executed_source_hashes_unchanged=True,all_nested_roles_disjoint=True,inner_training_only_preprocessor=True,all_eligibility_and_random_controls_exact=True,selection_summary=allfolds,no_training_or_inference=True,no_outer_access=True))
    print('SCREENING_INDEPENDENT_VERIFICATION_PASS',cells,checked,flush=True)


if __name__=='__main__':main()
