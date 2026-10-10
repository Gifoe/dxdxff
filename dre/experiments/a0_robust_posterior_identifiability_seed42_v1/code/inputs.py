"""Read-only frozen artifact gate; no scorer training or regenerated OOF scores."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import sha,digest,metric,METRICS,write_json

def load_inputs(previous,source,protocol,pub):
    previous=Path(previous); source=Path(source); protocol=Path(protocol)
    root=protocol.parent; lock=json.loads(protocol.read_text(encoding='utf-8'))
    old_source=source/'a0_source_posterior_decoder_seed42_v1'
    old_lock=json.loads((old_source/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    assert sha(old_source/'PROTOCOL_LOCK.json')==lock['prior_protocol_sha256']
    oldpub=previous/'public'; oldstatus=json.loads((oldpub/'RUN_STATUS.json').read_text())
    for name,h in oldstatus['aggregate_files'].items():assert sha(oldpub/name)==h
    assert oldstatus['terminal']=='POSTERIOR_IDENTIFIABILITY_FAILED'
    old_data=json.loads((oldpub/'DATA_AND_SPLIT_AUDIT.json').read_text())
    for name,h in old_data['binding']['code'].items():assert sha(old_source/'code'/name)==h
    oldtests=json.loads((oldpub/'DECODER_UNIT_TEST_AUDIT.json').read_text())
    assert oldtests['status']=='PASS'
    assert sha(root/'audit/FROZEN_INPUT_MANIFEST.json')==lock['private_input_manifest_sha256']
    manifest=json.loads((root/'audit/FROZEN_INPUT_MANIFEST.json').read_text())
    for rel,h in manifest.items():assert sha(previous/rel)==h, 'Immutable private artifact changed'
    feature=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
    assert sha(feature)==lock['feature_sha256'] and sha(split)==lock['split_sha256']
    core=source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
    assert sha(core)==old_lock['a0_core_sha256']; sys.path.insert(0,str(core.parent))
    from uas_core import prepare,oof_plan,validate_exclusion
    prior=Path(r'C:\pr_uncertainty_aware_supervision_seed42_runtime')
    ta=json.loads((prior/'OOF_TEACHER_AUDIT.json').read_text())
    provenance=json.loads((oldpub/'OOF_PROVENANCE_AUDIT.json').read_text())
    replays=json.loads((oldpub/'INDEPENDENT_OOF_REPLAY.json').read_text())
    assert replays['status']=='PASS' and replays['all_A0_D2_query_predictions_replayed']==40
    scorer_replay=json.loads((oldpub/'INDEPENDENT_SCORER_REPLAY.json').read_text())
    assert scorer_replay['status']=='PASS'
    reproduction=json.loads((oldpub/'A0_REPRODUCTION.json').read_text())
    banks=[]; controls=[]; exposure=[]; teacher_checks=[]
    for fold in range(1,6):
        cell=previous/f'fold{fold}'
        d=torch.load(cell/'BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        full=torch.load(cell/'D2_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
        of=torch.load(cell/'D2_OOF_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert d['binding']==old_data['binding']
        assert sha(cell/'BANK_PRIVATE.pt')==reproduction['folds'][fold-1]['sealed_hash']
        assert full['binding']==of['binding']
        assert full['checkpoint_sha256']==sha(cell/'D2/BEST_PRIVATE.pt')
        ck=torch.load(cell/'D2/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==full['epoch'] and ck['threshold']==full['threshold']
        assert sum(v.numel() for v in ck['model'].values())==9221
        a0ck=Path(d['A0_checkpoint']); assert sha(a0ck)==old_lock['checkpoint_sha256'][fold-1]
        assert d['A0_threshold']==old_lock['thresholds'][fold-1]
        a0=torch.load(a0ck,map_location='cpu',weights_only=False)
        assert a0['epoch']==old_lock['epochs'][fold-1]
        assert sum(v.numel() for v in a0['model'].values())==8817
        tr=d['train']; va=d['val']; fp=d['patient'][tr]; fit=set(fp)
        assert fit.isdisjoint(set(d['patient'][va]))
        assert d['plans']==oof_plan(fit,fold)
        assert np.array_equal(fp,of['patient']) and np.array_equal(d['channel'][tr],of['channel'])
        assert np.isfinite(d['a0_oof']).all() and np.isfinite(of['logits']).all()
        seen=np.zeros(len(tr),int)
        for plan in d['plans']:
            validate_exclusion(plan['train'],plan['validation'],plan['query'],fit)
            qi=np.flatnonzero(np.isin(fp,plan['query'])); seen[qi]+=1
            ti=np.flatnonzero(np.isin(fp,plan['train']))
            _,pre=prepare(d['raw_x'][tr],fp,ti)
            ph=digest({q:pre[q].tolist() for q in ['mean','scale','var','imputer_statistics']})
            k=plan['group']; oldteacher=prior/f'fold{fold}/teacher{k}'
            rec=torch.load(oldteacher/'OOF_PRIVATE.pt',map_location='cpu',weights_only=False)
            aud=next(r for r in ta['folds'] if r['fold']==fold and r['teacher_group']==k)
            assert rec['source']['plan']==plan
            assert sha(oldteacher/'OOF_PRIVATE.pt')==aud['OOF_file_sha256']
            assert sha(oldteacher/'BEST_PRIVATE.pt')==aud['checkpoint_sha256']==rec['source']['checkpoint']
            assert np.array_equal(rec['patient'],fp[qi]) and np.array_equal(rec['channel'],d['channel'][tr][qi])
            for key in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[key],rec['preprocessor'][key])
            for fam,file in [('A0',oldteacher/'BEST_PRIVATE.pt'),('D2',cell/f'D2_teacher{k}/BEST_PRIVATE.pt')]:
                pa=next(r for r in provenance['teachers'] if r['fold']==fold and r['group']==k and r['family']==fam)
                assert sha(file)==pa['checkpoint_sha256'] and pa['preprocessor_hash']==ph
                assert pa['query_identity_hash']==digest(list(zip(fp[qi],d['channel'][tr][qi])))
                teacher_checks.append({'fold':fold,'family':fam,'group':k,'checkpoint_sha256':sha(file),'preprocessor_hash':ph,
                    'query_channels':len(qi),'query_train_selection_overlap':0,'historical_exact_replay_confirmed':True})
        assert (seen==1).all()
        for pat in sorted(fit):
            remaining=fp!=pat; exposed=np.zeros(len(fp),bool); train_exp=0; selection_exp=0
            for plan in d['plans']:
                train_exp+=int(pat in plan['train']); selection_exp+=int(pat in plan['validation'])
                if pat in plan['train'] or pat in plan['validation']:exposed|=np.isin(fp,plan['query'])
            exposure.append({'fold':fold,'patient':pat,'remaining_channels':int(remaining.sum()),
                'exposed_channels':int((exposed&remaining).sum()),'exposed_fraction':float(exposed[remaining].mean()),
                'train_exposed_teachers':train_exp,'selection_exposed_teachers':selection_exp,'any_exposure':bool(exposed[remaining].any())})
        for pat in sorted(set(d['patient'][va])):
            ix=va[d['patient'][va]==pat]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
            for name,scores,tau in [('D0',d['a0_scores'],d['A0_threshold']),('D2',full['scores'],full['threshold'])]:
                controls.append({'fold':fold,'patient':pat,'center':str(d['center'][ix[0]]),'method':name,
                    **metric(d['y'][ix],scores[ix],tau)})
        banks.append((d,full,of))
    c=pd.DataFrame(controls)
    for name,expected in [('D0',.6380797828499001),('D2',.6479260773776448)]:
        sub=c[c.method==name]; assert len(sub)==65 and sub.patient.nunique()==47 and sub.channels.sum()==6273
        assert abs(sub.macro_f1.mean()-expected)<=1e-12
    # Check every legacy published reference metric, not only Macro-F1.
    historic=pd.read_csv(oldpub/'VALIDATION_SUMMARY.csv').set_index('method')
    for name in ['D0','D2']:
        for m in METRICS:assert np.isclose(c[c.method==name][m].mean(),historic.loc[name,m],rtol=0,atol=1e-12,equal_nan=True)
    write_json(pub/'A0_D2_REPRODUCTION.json',{'status':'PASS','prediction_metric_replay_current':True,
        'fresh_backbone_forward_in_this_run':False,'prior_independent_forward_replay_hash':sha(oldpub/'INDEPENDENT_SCORER_REPLAY.json'),
        'metrics':{n:c[c.method==n][METRICS].mean().to_dict() for n in ['D0','D2']},'new_training':False,'outer_TEST':False})
    write_json(pub/'OOF_SOURCE_INTEGRITY.json',{'status':'PASS','private_artifacts_sha_verified':len(manifest),'teachers':teacher_checks,
        'exact_once_per_FIT_channel':True,'no_new_teacher_outputs':True,'prior_independent_OOF_replay_sha256':sha(oldpub/'INDEPENDENT_OOF_REPLAY.json')})
    ef=pd.DataFrame(exposure); rows=[]
    for fold,g in ef.groupby('fold'):
        rows.append({'fold':int(fold),'pseudo_targets':len(g),'targets_with_upstream_exposure':int(g.any_exposure.sum()),
            'mean_density_input_channel_exposed_fraction':g.exposed_fraction.mean(),
            'mean_train_exposed_teachers':g.train_exposed_teachers.mean(),'mean_selection_exposed_teachers':g.selection_exposed_teachers.mean()})
    write_json(pub/'OOF_TEACHER_EXPOSURE_AUDIT.json',{'status':'SCREENING_DIAGNOSTIC_NOT_FULLY_NESTED','same_exposure_for_A0_D2':True,
        'fully_nested_independence':False,'direct_target_labels_excluded_from_density_prior_floor':True,'folds':rows,
        'teacher_selection_exposure_counts_as_label_dependence':True,'no_retraining_to_hide_exposure':True})
    return banks,controls
