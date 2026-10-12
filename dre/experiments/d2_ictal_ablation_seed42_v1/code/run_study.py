"""Seven fixed D2 arms; label-gated audit, original trainer, immutable references."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ablation_core import ARMS, NEW_ARMS, KEYS, KEEP, parse_names, mask_for, masked, source_free_factory


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec); spec.loader.exec_module(out)
    return out


def raw_without_y(a):
    with np.load(a.features, allow_pickle=True) as f:
        raw = {k: f[k] for k in f.files if k != 'y'}
    assert raw['x'].shape == (7635,88) and len(set(raw['patient'])) == 80
    assert len(set(zip(raw['patient'],raw['channel']))) == 7635
    return raw


def roles(raw, fold):
    s = raw['split_fold'] == fold
    out = {r:set(raw['split_patient'][s & (raw['split_role']==r)]) for r in ['fit','validation','test']}
    assert all(out[x].isdisjoint(out[y]) for x,y in [('fit','validation'),('fit','test'),('validation','test')])
    return out


def setup(a):
    lock = json.loads(a.protocol.read_text()); assert lock['arms'] == ARMS and lock['new_arms'] == NEW_ARMS
    oc = a.source/'a0_source_posterior_decoder_seed42_v1/code'
    common = module('study_common', oc/'common.py')
    amendment = a.source/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json'
    assert common.sha(amendment) == lock['baseline_amendment_sha256']
    ref = json.loads(amendment.read_text())
    cp = a.source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
    assert common.sha(cp) == ref['core_sha256']
    sys.path.insert(0,str(a.source)); sys.path.insert(0,str(cp.parent)); sys.path.insert(0,str(oc))
    import uas_core, model, train
    tests = json.loads((a.original/'public/DECODER_UNIT_TEST_AUDIT.json').read_text())
    ch = {p.name:common.sha(p) for p in oc.glob('*.py')}
    assert tests['status']=='PASS' and ch == tests['code_sha256']
    assert common.sha(a.features)==ref['feature_sha256']
    split = Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
    assert common.sha(split)==ref['split_sha256']
    paths={'features':a.features,'split':split,'protocol':a.protocol,'core':cp,'amendment':amendment}
    for key,root,seal in [('full',a.full,'OUTER_PREDICTIONS_SEAL.json'),('noZ',a.noz,'OUTER_PREDICTION_SEAL.json')]:
        for tag,name in [('freeze','FROZEN_BEFORE_OUTER.json'),('seal',seal)]:
            path=root/name; assert common.sha(path)==lock[f'{key}_{tag}_sha256'];paths[f'{key}_{tag}']=path
    assert json.loads((a.full/'public/OUTER_STATUS.json').read_text())['status']=='ONE_FROZEN_D2_OUTER_PASS_COMPLETE'
    assert json.loads((a.noz/'public/OUTER_STATUS.json').read_text())['status']=='ONE_FROZEN_NO_PATIENT_Z_OUTER_PASS_COMPLETE'
    for name in ['feature_aggregation','evidence_views']:
        path=a.source/('task1_baselines' if name=='feature_aggregation' else 'neuroez_c')/(name+'.py')
        assert common.sha(path)==lock[name+'_sha256'];paths[name]=path
    helper_path=a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py'
    helper=module('study_outer',helper_path);paths['outer_helper']=helper_path
    for fold in range(1,6):
        bp=a.original/f'fold{fold}/BANK_PRIVATE.pt';ck=a.original/f'fold{fold}/D2/BEST_PRIVATE.pt'
        assert common.sha(bp)==ref['bank_sha256'][fold-1] and common.sha(ck)==ref['checkpoint_sha256'][fold-1]
        paths[f'original_bank_{fold}']=bp;paths[f'original_checkpoint_{fold}']=ck
        nofr=json.loads((a.noz/'FROZEN_BEFORE_OUTER.json').read_text())['folds'][fold-1]
        no=a.noz/f'fold{fold}/NO_PR/BEST_PRIVATE.pt'; nb=a.noz/f'fold{fold}/BANK_PRIVATE.pt'
        assert common.sha(no)==nofr['checkpoint_sha256'] and common.sha(nb)==nofr['bank_sha256']
        paths[f'noZ_checkpoint_{fold}']=no;paths[f'noZ_bank_{fold}']=nb
    for path in sorted(Path(__file__).parent.glob('*.py')):paths['newcode_'+path.name]=path
    for path in sorted(Path(__file__).parents[1].joinpath('tests').glob('*.py')):paths['newtest_'+path.name]=path
    bind={'dependencies':{k:common.sha(v) for k,v in paths.items()},'original_training_code':ch}
    a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'public').mkdir(exist_ok=True)
    return lock,ref,common,uas_core,model,train,helper,bind


def feature_audit(a,o):
    lock,ref,c,core,model,trainer,helper,bind=o
    pub=a.runtime/'public'
    if (pub/'FEATURE_GROUP_MANIFEST.json').exists():
        old=json.loads((pub/'FEATURE_GROUP_MANIFEST.json').read_text());assert old['binding']==bind;return
    raw=raw_without_y(a);names=raw['features'].tolist(); columns=parse_names(names)
    assert hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()==ref['feature_order_sha256']
    from task1_baselines.feature_aggregation import build_channel_feature_table
    from neuroez_c.evidence_views import BASE_SPECTRAL_FEATURE_NAMES
    rng=np.random.default_rng(42)
    synthetic={'features':rng.normal(size=(6,2,len(BASE_SPECTRAL_FEATURE_NAMES))).astype('float32'),
        'window_centers':np.array([-3.,-2.,-1.,1.,2.,3.]),'feature_names':BASE_SPECTRAL_FEATURE_NAMES,
        'channel_names':['synthetic_a','synthetic_b'],'labels_nez':np.array([0,1]),'subject_id':'synthetic'}
    table,m=build_channel_feature_table([synthetic],profile='p2_matched_simple')
    assert m['feature_names']==names,'Export and exact original generation names differ'
    duplicates=[]
    for col in columns:
        if col['block']=='physics':
            i=col['index']; j=names.index(col['name'].replace('physics_','b0_',1))
            assert np.array_equal(raw['x'][:,i],raw['x'][:,j],equal_nan=True)
            assert np.array_equal(table[names[i]],table[names[j]])
            col['duplicate_of']=j
            duplicates.append({'auxiliary_index':i,'main_index':j,'view':col['view'],
                'exact_export_parity':True,'exact_synthetic_generation_parity':True,'max_absolute_difference':0.})
    assert len(duplicates)==16
    zero=[i for i in range(88) if np.all(raw['x'][:,i]==0.)]
    for col in columns:col['constant_zero_in_frozen_export']=col['index'] in zero
    counts={v:sum(col['view']==v for col in columns) for v in ['abs','delta','zdelta','ratio']}
    assert counts=={'abs':18,'delta':26,'zdelta':26,'ratio':18}
    masks=[]
    for arm in ARMS:
        keep=mask_for(columns,arm)
        assert all(keep[p['auxiliary_index']]==keep[p['main_index']] for p in duplicates)
        masks.append({'arm':arm,'retained_feature_count':int(keep.sum()),'excluded_feature_count':int((~keep).sum()),
            'retained_views':','.join(KEEP[arm]),'constant_zero_retained':sum(keep[i] for i in zero),
            'varying_dimensions_retained':sum(keep[i] for i in range(88) if i not in zero),
            'retained_auxiliary_duplicate_count':sum(keep[p['auxiliary_index']] for p in duplicates),
            'all_same_view_duplicate_copies_masked_together':True,'mask_after_full_preprocessing':True})
    c.write_json(pub/'FEATURE_GROUP_MANIFEST.json',{'status':'PASS','binding':bind,'columns':columns,'group_counts':counts,
        'feature_order_sha256':ref['feature_order_sha256'],'source_name_parity':True,'indices_zero_based':True})
    c.write_json(pub/'FEATURE_DUPLICATION_AUDIT.json',{'status':'PASS','auxiliary_duplicates':duplicates,
        'exact_duplicate_auxiliary_dimensions':16,'constant_zero_indices':zero,
        'constant_zero_dimension_count':len(zero),'no_deduplication_or_reextraction':True,
        'constant_cross_view_equalities_not_treated_as_information_leak':True})
    pd.DataFrame(masks).to_csv(pub/'FEATURE_MASK_AUDIT.csv',index=False)
    print('FEATURE_MAPPING_AND_DUPLICATION_GATE_PASS',counts,len(zero),flush=True)


def prepare(a,o):
    lock,ref,c,core,model,trainer,helper,bind=o;pub=a.runtime/'public'
    assert json.loads((pub/'FEATURE_GROUP_MANIFEST.json').read_text())['binding']==bind
    if (pub/'PRETRAIN_AUDIT.json').exists():
        assert json.loads((pub/'PRETRAIN_AUDIT.json').read_text())['binding']==bind;return
    raw=raw_without_y(a);columns=parse_names(raw['features'].tolist());rows=[]; all_test=[]
    for fold in range(1,6):
        d=torch.load(a.original/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        ids=roles(raw,fold);di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])))
        for k in ['patient','channel','center']:assert np.array_equal(raw[k][di],d[k])
        assert np.array_equal(raw['x'][di],d['raw_x'],equal_nan=True) and np.array_equal(d['features'],raw['features'])
        assert set(d['patient'][d['train']])==ids['fit'] and set(d['patient'][d['val']])==ids['validation']
        assert set(d['pre']['fit_patient_ids'])==ids['fit'] and not set(d['patient'])&ids['test']
        x,pre=core.prepare(d['raw_x'],d['patient'],d['train'])
        assert np.array_equal(x,d['x'])
        for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[k],d['pre'][k])
        assert np.array_equal(helper.frozen_transform(d['raw_x'],d['patient'],d['pre'],core),x)
        seed=42+1009*fold;core.seed_all(seed); full=model.SourceA0(seed)
        ck=torch.load(a.original/f'fold{fold}/D2/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert ck['binding']['initial_hash']==core.state_hash(full.state_dict())
        replay=model.SourceA0().eval();replay.load_state_dict(ck['model'])
        with torch.no_grad():logits=replay(torch.from_numpy(x),torch.from_numpy(d['g'])).numpy()
        development=torch.load(a.original/f'fold{fold}/D2_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
        drift=float(np.max(np.abs(logits-development['logits'])));assert drift<=ref['replay_logit_absolute_tolerance']
        assert len(ids['test'])==ref['expected_test_patients_by_fold'][fold-1];all_test.extend(ids['test'])
        for arm in NEW_ARMS:
            mask=mask_for(columns,arm);xx=masked(x,mask)
            core.seed_all(seed);tm=(source_free_factory(model.SourceA0) if arm=='w/o Source' else model.SourceA0)(seed)
            assert core.state_hash(tm.network.state_dict())==core.state_hash(full.network.state_dict())
            n=sum(p.numel() for p in tm.parameters());assert n==(8817 if arm=='w/o Source' else 9221)
            bank=a.runtime/KEYS[arm]/f'fold{fold}/BANK_PRIVATE.pt'
            assert not bank.exists()
            cell={k:d[k] for k in ['patient','channel','center','features','train','val','y','g','pre']}
            cell.update({'x':xx,'mask':mask,'binding':bind,'arm':arm,'initial_hash':core.state_hash(tm.state_dict())})
            core.torch_write(bank,cell)
            rows.append({'arm':arm,'fold':fold,'bank_sha256':c.sha(bank),'initial_hash':cell['initial_hash'],
                'shared_initial_state_matches_Full':True,'parameters':n,'retained_columns':int(mask.sum()),
                'excluded_input_columns_exact_zero':True,'retained_transformation_bitwise_unchanged':True,
                'FIT_only_preprocessor':True,'outer_overlap':0,'full_replay_max_logit_drift':drift})
    assert len(all_test)==len(set(all_test))==80 and len(rows)==25
    c.write_json(pub/'PRETRAIN_AUDIT.json',{'status':'PASS','binding':bind,'cells':rows,'new_full_cohort_y_loaded':False})
    print('ALL_25_FIT_VAL_BANKS_AND_FULL_REPLAY_PASS',flush=True)


def train_all(a,o):
    lock,ref,c,core,model,trainer,helper,bind=o;pub=a.runtime/'public'
    pre=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());assert pre['binding']==bind
    tests=json.loads((pub/'ENGINEERING_TESTS.json').read_text());assert tests['status']=='PASS' and tests['binding']==bind
    complete=0
    for arm in NEW_ARMS:
        factory=source_free_factory(model.SourceA0) if arm=='w/o Source' else model.SourceA0
        for fold in range(1,6):
            bp=a.runtime/KEYS[arm]/f'fold{fold}/BANK_PRIVATE.pt'
            rec=next(r for r in pre['cells'] if r['arm']==arm and r['fold']==fold)
            assert c.sha(bp)==rec['bank_sha256']
            d=torch.load(bp,map_location='cpu',weights_only=False)
            assert d['binding']==bind and np.all(d['x'][:,~d['mask']]==0)
            old_factory=trainer.SourceA0;trainer.SourceA0=factory
            try:
                trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],bp.parent/'TRAIN',42+1009*fold,
                    c.digest({'bank':c.sha(bp),'binding':bind}),device='cuda',penalty=0. if arm=='w/o Source' else .001)
            finally:trainer.SourceA0=old_factory
            ck=torch.load(bp.parent/'TRAIN/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert ck['binding']['initial_hash']==d['initial_hash']
            complete+=1
            c.write_json(pub/'TRAIN_PROGRESS.json',{'status':'RUNNING','complete':complete,'total':25,'last_arm':arm,'last_fold':fold})
            print('ABLATION_FOLD_SELECTED',arm,fold,ck['epoch'],ck['threshold'],flush=True)
    c.write_json(pub/'TRAIN_PROGRESS.json',{'status':'ALL_25_NEW_TRAIN_VALIDATION_COMPLETE','complete':25,'total':25,'torch_version':torch.__version__})


def bank_ck(a,arm,fold):
    if arm=='Full D2':return a.original/f'fold{fold}/BANK_PRIVATE.pt',a.original/f'fold{fold}/D2/BEST_PRIVATE.pt'
    if arm=='w/o Patient Z':return a.noz/f'fold{fold}/BANK_PRIVATE.pt',a.noz/f'fold{fold}/NO_PR/BEST_PRIVATE.pt'
    root=a.runtime/KEYS[arm]/f'fold{fold}'
    return root/'BANK_PRIVATE.pt',root/'TRAIN/BEST_PRIVATE.pt'


def freeze(a,o):
    lock,ref,c,core,model,trainer,helper,bind=o;pub=a.runtime/'public'
    assert json.loads((pub/'TRAIN_PROGRESS.json').read_text())['complete']==25
    out=a.runtime/'FROZEN_BEFORE_OUTER.json'
    if out.exists():assert json.loads(out.read_text())['binding']==bind;return
    columns=json.loads((pub/'FEATURE_GROUP_MANIFEST.json').read_text())['columns'];rows=[]
    for arm in ARMS:
        for fold in range(1,6):
            bp,cp=bank_ck(a,arm,fold);ck=torch.load(cp,map_location='cpu',weights_only=False)
            assert ck['binding']['seed']==42+1009*fold and ck['binding']['max_epochs']==30
            assert ck['binding']['penalty']==(0. if arm=='w/o Source' else .001)
            n=sum(v.numel() for v in ck['model'].values());assert n==(8817 if arm=='w/o Source' else 9221)
            if arm in NEW_ARMS:
                d=torch.load(bp,map_location='cpu',weights_only=False)
                assert np.array_equal(d['mask'],mask_for(columns,arm)) and np.all(d['x'][:,~d['mask']]==0)
                done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text())
                assert done['best_sha256']==c.sha(cp) and done['threshold']==ck['threshold'] and done['selected_epoch']==ck['epoch']
            rows.append({'arm':arm,'fold':fold,'epoch':ck['epoch'],'nez_threshold':ck['threshold'],
                'bank_sha256':c.sha(bp),'checkpoint_sha256':c.sha(cp),'mask':mask_for(columns,arm).tolist(),
                'parameters':n,'reused':arm not in NEW_ARMS})
    c.write_json(out,{'status':'PASS','binding':bind,'selections':rows,'feature_manifest_sha256':c.sha(pub/'FEATURE_GROUP_MANIFEST.json'),
        'all_35_models_masks_preprocessors_thresholds_locked':True,'new_outer_inference_started':False,'new_full_y_loaded':False})
    pd.DataFrame([{k:v for k,v in r.items() if k!='mask'} for r in rows]).to_csv(pub/'MODEL_SELECTION.csv',index=False)
    c.write_json(pub/'FREEZE_AUDIT.json',{'status':'PASS','freeze_sha256':c.sha(out),'new_models':25,'reused_models':10,
        'all_masks_frozen':True,'all_35_selections_before_new_outer':True})
    print('ALL_35_SELECTIONS_FROZEN',flush=True)


def prediction_path(a,arm,fold):
    if arm=='Full D2':return a.full/f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt'
    if arm=='w/o Patient Z':return a.noz/f'fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt'
    return a.runtime/KEYS[arm]/f'fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt'


def evaluate(a,o):
    lock,ref,c,core,model,trainer,helper,bind=o;pub=a.runtime/'public'
    assert not (pub/'RUN_STATUS.json').exists(),'Completed scientific outer evaluation cannot rerun'
    frozen=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text());assert frozen['binding']==bind
    assert c.sha(pub/'FEATURE_GROUP_MANIFEST.json')==frozen['feature_manifest_sha256']
    freeze_sha=c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json');raw=raw_without_y(a);cells={};sealed={}
    full_seal=json.loads((a.full/'OUTER_PREDICTIONS_SEAL.json').read_text())['files']
    noz_seal=json.loads((a.noz/'OUTER_PREDICTION_SEAL.json').read_text())['hashes']
    test_masks=[]
    for arm in ARMS:
        visited=[]
        for fold in range(1,6):
            rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold)
            bp,cp=bank_ck(a,arm,fold);assert c.sha(bp)==rec['bank_sha256'] and c.sha(cp)==rec['checkpoint_sha256']
            ix=np.flatnonzero(np.isin(raw['patient'],list(roles(raw,fold)['test'])))
            path=prediction_path(a,arm,fold)
            if arm not in NEW_ARMS:
                expected=(full_seal if arm=='Full D2' else noz_seal)[str(fold)]
                assert c.sha(path)==expected
                cell=torch.load(path,map_location='cpu',weights_only=False)
                assert np.array_equal(cell['indices'],ix)
            elif path.exists():
                assert json.loads(path.with_suffix('.sha256.json').read_text())=={'binding':freeze_sha,'sha256':c.sha(path)}
                cell=torch.load(path,map_location='cpu',weights_only=False)
                assert cell['binding']==freeze_sha and np.array_equal(cell['indices'],ix)
            else:
                d=torch.load(bp,map_location='cpu',weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False)
                x=helper.frozen_transform(raw['x'][ix],raw['patient'][ix],d['pre'],core)
                x=masked(x,np.asarray(rec['mask'],dtype=bool))
                factory=source_free_factory(model.SourceA0) if arm=='w/o Source' else model.SourceA0
                tm=factory().eval();tm.load_state_dict(ck['model'],strict=True)
                with torch.inference_mode():
                    logits=tm(torch.from_numpy(x),torch.from_numpy(model.source_index(raw['center'][ix]))).numpy()
                    score=torch.sigmoid(torch.from_numpy(logits)).numpy()
                cell={'binding':freeze_sha,'indices':ix,'logits':logits,'scores':score,'threshold':ck['threshold']}
                core.torch_write(path,cell);c.write_json(path.with_suffix('.sha256.json'),{'binding':freeze_sha,'sha256':c.sha(path)})
                test_masks.append({'arm':arm,'fold':fold,'outer_channels':len(ix),'excluded_input_columns_exact_zero':True,
                    'retained_preprocessing_unchanged':True})
            assert cell['threshold']==rec['nez_threshold'] and np.isfinite(cell['scores']).all()
            cells[(arm,fold)]=cell;sealed[KEYS[arm]+'_'+str(fold)]=c.sha(path);visited.extend(ix.tolist())
        assert sorted(visited)==list(range(7635))
    c.write_json(a.runtime/'OUTER_PREDICTIONS_SEAL.json',{'binding':freeze_sha,'files':sealed,'all_35_predictions_before_full_y':True})
    # First new full-cohort label materialization occurs only after all prediction files seal.
    with np.load(a.features,allow_pickle=True) as f:y=f['y']
    assert len(y)==7635 and int((y==0).sum())==1743
    rows=[]
    for (arm,fold),cell in cells.items():
        ix=cell['indices']
        for pat in sorted(set(raw['patient'][ix])):
            k=np.flatnonzero(raw['patient'][ix]==pat)
            k=k[np.argsort(raw['channel'][ix][k].astype(str),kind='stable')];si=ix[k]
            rows.append({'arm':arm,'fold':fold,'patient':pat,'center':str(raw['center'][si[0]]),
                **c.metric(y[si],cell['scores'][k],cell['threshold'])})
    df=pd.DataFrame(rows);assert len(df)==560 and df.patient.nunique()==80
    core.torch_write(a.runtime/'PATIENT_METRICS_PRIVATE.pt',rows)
    masks=pd.read_csv(pub/'FEATURE_MASK_AUDIT.csv').set_index('arm');summary=[]
    for arm in ARMS:
        part=df[df.arm==arm];assert len(part)==part.patient.nunique()==80 and part.channels.sum()==7635
        if arm in ['Full D2','w/o Patient Z']:
            old=json.loads(((a.full if arm=='Full D2' else a.noz)/'public/OUTER_STATUS.json').read_text())
            original=old['primary_metrics'] if arm=='Full D2' else old['primary_no_PR_metrics']
            for key in c.METRICS:assert abs(part[key].mean()-original[key])<1e-12
        summary.append({'arm':arm,**part[c.METRICS].mean().to_dict(),
            'parameters':8817 if arm=='w/o Source' else 9221,'effective_feature_count':int(masks.loc[arm,'retained_feature_count']),
            'varying_feature_count':int(masks.loc[arm,'varying_dimensions_retained']),
            'training_status':'trained_from_scratch' if arm in NEW_ARMS else 'verified_frozen_reuse','patients':80,'channels':7635})
    pd.DataFrame(summary).to_csv(pub/'ABLATION_SUMMARY.csv',index=False)
    for field,file in [('fold','OUTER_BY_FOLD.csv'),('center','OUTER_BY_CENTER.csv')]:
        output=[]
        for (arm,value),part in df.groupby(['arm',field]):
            output.append({'arm':arm,field:value,'patients':len(part),'channels':int(part.channels.sum()),**part[c.METRICS].mean().to_dict()})
        pd.DataFrame(output).to_csv(pub/file,index=False)
    full=df[df.arm=='Full D2'].set_index('patient').sort_index();contrasts=[]
    for arm in ARMS[1:]:
        part=df[df.arm==arm].set_index('patient').sort_index();assert np.array_equal(part.index,full.index)
        delta=full[c.METRICS].to_numpy()-part[c.METRICS].to_numpy()
        boot=c.cluster_bootstrap(full.index,delta,10000)
        for j,key in enumerate(c.METRICS):
            contrasts.append({'comparison':'Full D2 minus '+arm,'ablated_arm':arm,'metric':key,'delta':np.nanmean(delta[:,j]),
                'ci_low':np.nanpercentile(boot[:,j],2.5),'ci_high':np.nanpercentile(boot[:,j],97.5),
                'patients_full_better':int((delta[:,j]>0).sum()),'patients_ablation_better':int((delta[:,j]<0).sum()),
                'patients_tied':int((delta[:,j]==0).sum()),'draws':10000,'seed':42,'multiplicity_adjusted':False})
    pd.DataFrame(contrasts).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
    history=[]
    for arm in NEW_ARMS:
        for fold in range(1,6):
            _,cp=bank_ck(a,arm,fold);last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            history.extend({'arm':arm,'fold':fold,**h} for h in last['history'])
    pd.DataFrame(history).to_csv(pub/'TRAINING_HISTORY.csv',index=False)
    assert setup(a)[-1]==bind
    c.write_json(pub/'IMMUTABILITY_AUDIT.json',{'status':'PASS','historical_inputs_models_and_predictions_unchanged':True,
        'all_35_selections_before_new_outer':True,'all_predictions_before_full_y':True,'patients_per_arm':80,'channels_per_arm':7635,
        'freeze_sha256':freeze_sha,'prediction_seal_sha256':c.sha(a.runtime/'OUTER_PREDICTIONS_SEAL.json'),'actual_TEST_mask_checks':test_masks})
    c.write_json(pub/'RUN_STATUS.json',{'status':'SEVEN_ARM_FROZEN_OUTER_COMPLETE','arms':{arm:'COMPLETE' for arm in ARMS},
        'scientific_status':lock['scientific_status'],'new_training_cells':25,'reused_cells':10,'private_predictions_not_published':True})
    print('SEVEN_ARM_FROZEN_OUTER_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','prepare','train','freeze','evaluate','run'])
    p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
    p.add_argument('--runtime',type=Path,default=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'))
    p.add_argument('--original',type=Path,default=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'))
    p.add_argument('--full',type=Path,default=Path(r'C:\d2_ictal_outer_seed42_runtime'))
    p.add_argument('--noz',type=Path,default=Path(r'C:\d2_no_patient_z_outer_seed42_runtime'))
    p.add_argument('--features',type=Path,default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
    p.add_argument('--protocol',type=Path,default=Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
    a=p.parse_args();torch.set_num_threads(2);o=setup(a);start=time.monotonic()
    phases={'audit':feature_audit,'prepare':prepare,'train':train_all,'freeze':freeze,'evaluate':evaluate}
    for phase in (['train','freeze','evaluate'] if a.phase=='run' else [a.phase]):phases[phase](a,o)
    print('ELAPSED_SECONDS',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
