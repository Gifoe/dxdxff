"""Fresh exact frozen B0 replay; exports only private, ordered 88D inputs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for n in ('source', 'cache', 'ledger', 'split', 'manifest', 'checkpoints', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'A0_REPRODUCTION.json').exists():
        raise RuntimeError('Do not overwrite an existing reproduction gate')
    sys.path.insert(0, str(a.source))
    from outcome_hifos.cache_schema import load_cache_contract
    from task1_baselines.cache_io import task1_feature_records
    from task1_baselines.feature_aggregation import build_channel_feature_table
    from task1_baselines.fold_protocol import load_task1_sensitivity_protocol, fold_manifest_hash
    from task1_baselines.patient_controls import _ChannelMLP, _Preprocessor, feature_columns
    torch.set_num_threads(2)
    expected_sha = '9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087'
    assert digest(a.cache) == expected_sha
    assert digest(a.ledger) == '3bd2039cc7d01ecdb375256354978066b4a1b4fc5b8adda1a58b84b2abb9f896'
    assert digest(a.split) == 'fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278'
    folds, split = load_task1_sensitivity_protocol(a.ledger, a.ledger, a.split, expected_subjects=80)
    assert fold_manifest_hash(folds) == '3635a5dabf54ee06411f54492f9660eb8807cd1b54796c727bc89ee644787355'
    # NumPy2 pickle names changed; alias to identical NumPy1 array reconstructors.
    # This changes import resolution only, not cached bytes or any array values.
    if np.__version__.startswith('1.'):
        import numpy.core.numeric
        import numpy.core.multiarray
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    cache = load_cache_contract(a.cache)
    table, manifest = build_channel_feature_table(task1_feature_records(cache, set(folds.subject_id)), profile='p2_matched_simple')
    assert manifest == json.loads(a.manifest.read_text(encoding='utf-8'))
    columns = feature_columns(table)
    assert columns == manifest['feature_names'] and len(columns) == 88
    assert len(table) == 7635 and table.subject_id.nunique() == 80
    assert not table[['subject_id','channel_name']].duplicated().any()
    old = pd.read_csv(a.ledger)
    checkhash = ['ada7aef890ac6cfd11033d730ca74d4c9d7fc0b71772c38251174f94e77e2e70',
                 'cce163d40f6850d7ef63355de595bcb125ada178f2124661c7683d9c8fae9120',
                 '286c3a6e0ba98ffe2eb1f74487b780f8015f064e42a7fc0112eaf41813b45ce1',
                 '7cf67486693cb5b6e54a1f7cd76566bff48309d11f84bd435dd46f650ee45431',
                 '411b01f3474b694377a9ec47aa752f7d00506bccca2d1daea9d576f2193e467c']
    audit = {'status':'RUNNING','historical_checkpoint_replay':True,'matched_A0_training_not_yet_run':True,
             'patients':80,'channels':7635,'parameters':8817,'feature_dimension':88,
             'feature_cache_sha256':expected_sha,'feature_manifest_sha256':digest(a.manifest),
             'fold_manifest_hash':fold_manifest_hash(folds),'split_sha256':digest(a.split),
             'source_hashes':{n:digest(a.source/n) for n in ['task1_baselines/patient_controls.py','task1_baselines/thresholds.py','task1_baselines/feature_aggregation.py','neuroez_c/evidence_views.py']},'folds':[]}
    rows=[]
    for fold in range(1,6):
        s=split[split.outer_fold==fold]
        ids={r:set(s.loc[s.partition==r,'subject_id']) for r in ['fit','validation','test']}
        assert all(not ids[r]&ids[t] for r,t in [('fit','validation'),('fit','test'),('validation','test')])
        groups={r:table[table.subject_id.isin(v)].reset_index(drop=True) for r,v in ids.items()}
        pre=_Preprocessor(SimpleImputer(), StandardScaler(), True)
        x=pre.fit_transform(groups['fit'],columns)
        model=_ChannelMLP(88).eval()
        assert sum(p.numel() for p in model.parameters())==8817
        ck=a.checkpoints/f'fold_{fold}'/'best_model.pt'
        assert digest(ck)==checkhash[fold-1]
        saved=torch.load(ck,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict'])
        test=groups['test']
        with torch.no_grad():
            score=torch.sigmoid(model(torch.from_numpy(pre.transform(test,columns)))).numpy()
        replay=test[['subject_id','channel_name','label_nez']].copy()
        replay['score']=score
        ref=old[old.outer_fold==fold]
        merged=replay.merge(ref,on=['subject_id','channel_name'],validate='one_to_one',suffixes=('','_ref'))
        assert len(merged)==len(test) and np.array_equal(merged.label_nez,merged.label_nez_ref)
        drift=float(np.max(np.abs(merged.score-merged.score_nez_probability)))
        assert drift<=2e-6
        tau=float(saved['selected_threshold'])
        assert np.all(np.abs(merged.selected_threshold-tau)<1e-12)
        assert np.array_equal((merged.score>=tau).astype(int),merged.predicted_nez)
        for _,g in merged.groupby('subject_id',sort=True):
            y=g.label_nez.to_numpy(int); pred=(g.score.to_numpy()>=tau).astype(int)
            pe=1-g.score.to_numpy()
            rows.append({'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
                         'ez_f1':float(f1_score(y==0,pred==0,zero_division=0)),
                         'ez_auprc':float(average_precision_score(y==0,pe)),
                         'ez_auroc':float(roc_auc_score(y==0,pe))})
        audit['folds'].append({'fold':fold,'max_score_drift':drift,'checkpoint_sha256':digest(ck),
                               'selected_epoch':int(saved['best_epoch']),'threshold_nez':tau,
                               **{r+'_patients':len(v) for r,v in ids.items()}})
        print('REPLAY_FOLD',fold,drift,flush=True)
    audit['metrics']={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    expected={'macro_f1':.6161672347405565,'ez_f1':.4238657223879644,
              'ez_auprc':.5319959936611567,'ez_auroc':.7268443834107704}
    assert all(abs(audit['metrics'][k]-v)<1e-8 for k,v in expected.items()), audit['metrics']
    destination=a.output/'FEATURES_PRIVATE.npz'
    np.savez_compressed(destination, x=table[columns].to_numpy(float),
        y=table.label_nez.to_numpy(np.int64), patient=table.subject_id.astype(str).to_numpy(),
        channel=table.channel_name.astype(str).to_numpy(), center=table.center.astype(str).to_numpy(),
        features=np.asarray(columns), split_patient=split.subject_id.astype(str).to_numpy(),
        split_fold=split.outer_fold.to_numpy(int),split_role=split.partition.astype(str).to_numpy())
    audit['private_features_sha256']=digest(destination)
    audit['status']='PASS'
    (a.output/'A0_REPRODUCTION.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit['metrics']),flush=True)
    print('B0_REPRODUCTION_PASS',flush=True)


if __name__=='__main__':
    main()
