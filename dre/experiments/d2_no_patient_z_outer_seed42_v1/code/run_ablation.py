"""One original D2 training arm, removing only patient_z. No raw signal I/O.

Phases: prepare FIT/VAL-only banks; train exact historical trainer; freeze all
five selections; evaluate once; aggregate against already frozen PR scores.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    out=importlib.util.module_from_spec(spec); spec.loader.exec_module(out)
    return out


def prepare_absolute(x, train):
    values=np.asarray(x,dtype=np.float64)
    imputer=SimpleImputer(); scaler=StandardScaler()
    imputer.fit(values[train]); scaler.fit(imputer.transform(values[train]))
    transformed=scaler.transform(imputer.transform(values)).astype(np.float32)
    assert transformed.shape==values.shape and np.isfinite(transformed).all()
    return transformed, {'imputer_statistics':imputer.statistics_, 'mean':scaler.mean_,
                         'scale':scaler.scale_, 'var':scaler.var_}


def frozen_absolute(x, pre):
    z=np.asarray(x,dtype=np.float64)
    result=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)
    assert result.shape==z.shape and np.isfinite(result).all()
    return result


def roles(raw, fold):
    s=raw['split_fold']==fold
    out={r:set(raw['split_patient'][s&(raw['split_role']==r)]) for r in ['fit','validation','test']}
    assert all(out[a].isdisjoint(out[b]) for a,b in [('fit','test'),('validation','test'),('fit','validation')])
    return out


def setup(a):
    lock=json.loads(a.protocol.read_text())
    original_amendment=a.source/'d2_ictal_outer_seed42_v1/PROTOCOL_AMENDMENT.json'
    common=module('historical_no_pr_common',a.source/'a0_source_posterior_decoder_seed42_v1/code/common.py')
    assert common.sha(original_amendment)==lock['baseline_amendment_sha256']
    reference=json.loads(original_amendment.read_text())
    original_code=a.source/'a0_source_posterior_decoder_seed42_v1/code'
    core_path=a.source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
    assert common.sha(core_path)==reference['core_sha256']
    sys.path.insert(0,str(core_path.parent)); sys.path.insert(0,str(original_code))
    import uas_core, model, train
    source_hashes={q.name:common.sha(q) for q in original_code.glob('*.py')}
    tests=json.loads((a.original/'public/DECODER_UNIT_TEST_AUDIT.json').read_text())
    assert source_hashes==tests['code_sha256'] and tests['status']=='PASS'
    assert common.sha(a.features)==reference['feature_sha256']
    split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
    assert common.sha(split)==reference['split_sha256']
    assert common.sha(a.baseline/'FROZEN_BEFORE_OUTER.json')==lock['baseline_freeze_sha256']
    assert common.sha(a.baseline/'OUTER_PREDICTIONS_SEAL.json')==lock['baseline_prediction_seal_sha256']
    oldfreeze=json.loads((a.baseline/'FROZEN_BEFORE_OUTER.json').read_text())
    oldstatus=json.loads((a.baseline/'public/OUTER_STATUS.json').read_text())
    assert oldstatus['status']=='ONE_FROZEN_D2_OUTER_PASS_COMPLETE'
    paths={'features':a.features, 'split':split, 'protocol':a.protocol, 'core':core_path,
           'code':Path(__file__), 'tests':Path(__file__).parents[1]/'tests/test_contract.py'}
    for fold in range(1,6):
        bankpath=a.original/f'fold{fold}/BANK_PRIVATE.pt'
        ckpath=a.original/f'fold{fold}/D2/BEST_PRIVATE.pt'
        assert common.sha(bankpath)==reference['bank_sha256'][fold-1]
        assert common.sha(ckpath)==reference['checkpoint_sha256'][fold-1]
        assert common.sha(bankpath)==oldfreeze['hashes'][f'bank_{fold}']
        paths[f'original_bank_{fold}']=bankpath; paths[f'original_checkpoint_{fold}']=ckpath
    bind={'locked_inputs':{k:common.sha(v) for k,v in paths.items()},'original_training_code':source_hashes}
    a.runtime.mkdir(parents=True,exist_ok=True)
    return lock,reference,common,uas_core,model,train,bind


def raw_without_labels(a):
    with np.load(a.features,allow_pickle=True) as f:
        raw={k:f[k] for k in f.files if k!='y'}
    assert raw['x'].shape==(7635,88) and len(set(raw['patient']))==80
    assert len(set(zip(raw['patient'],raw['channel'])))==7635
    return raw


def prepare(a, objects):
    lock,ref,common,core,model,trainer,binding=objects
    pub=a.runtime/'public'; pub.mkdir(exist_ok=True)
    raw=raw_without_labels(a)
    assert hashlib.sha256(json.dumps(raw['features'].tolist(),separators=(',',':')).encode()).hexdigest()==ref['feature_order_sha256']
    rows=[]
    for fold in range(1,6):
        out=a.runtime/f'fold{fold}/BANK_PRIVATE.pt'
        if out.exists():
            old=torch.load(out,map_location='cpu',weights_only=False)
            assert old['binding']==binding
            continue
        d=torch.load(a.original/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        ids=roles(raw,fold)
        di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])))
        for k in ['patient','channel','center']:
            assert np.array_equal(raw[k][di],d[k])
        assert np.array_equal(raw['x'][di],d['raw_x'],equal_nan=True)
        assert set(d['patient'][d['train']])==ids['fit']
        assert set(d['patient'][d['val']])==ids['validation']
        assert not set(d['patient'])&ids['test']
        x,pre=prepare_absolute(d['raw_x'],d['train'])
        assert np.array_equal(x,frozen_absolute(d['raw_x'],pre))
        assert not np.array_equal(x,d['x'])
        pre['fit_patient_ids']=sorted(ids['fit'])
        initial_seed=42+1009*fold; core.seed_all(initial_seed)
        initial=model.SourceA0(initial_seed)
        initial_hash=core.state_hash(initial.state_dict())
        original_ck=torch.load(a.original/f'fold{fold}/D2/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert initial_hash==original_ck['binding']['initial_hash']
        cell={k:d[k] for k in ['patient','channel','center','features','train','val','y','g']}
        cell.update({'x':x,'pre':pre,'binding':binding,'initial_hash':initial_hash})
        core.torch_write(out,cell)
        rows.append({'fold':fold,'fit_patients':len(ids['fit']),'validation_patients':len(ids['validation']),
            'fit_channels':len(d['train']),'validation_channels':len(d['val']),
            'outer_overlap':0,'initial_state_matches_original_D2':True,'initial_hash':initial_hash,
            'patient_z_removed':True,'features_retained':88,'FIT_only_scaler':True,
            'bank_sha256':common.sha(out),'preprocessor_sha256':common.digest({k:pre[k].tolist() for k in ['mean','scale','var','imputer_statistics']})})
    # Preparing an already complete set never replaces the original audit.
    if not (pub/'PRETRAIN_AUDIT.json').exists():
        assert len(rows)==5
        common.write_json(pub/'PRETRAIN_AUDIT.json',{'status':'PASS','binding':binding,'folds':rows,
            'all_cohort_y_member_loaded':False,'raw_signal_extraction':False,'only_one_new_training_arm':True})
    print('FIVE_NO_PATIENT_Z_BANKS_READY',flush=True)


def train_all(a, objects):
    lock,ref,common,core,model,trainer,binding=objects
    pretest=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text())
    assert pretest['binding']==binding
    for fold in range(1,6):
        path=a.runtime/f'fold{fold}/BANK_PRIVATE.pt'
        assert common.sha(path)==pretest['folds'][fold-1]['bank_sha256']
        d=torch.load(path,map_location='cpu',weights_only=False)
        cell=a.runtime/f'fold{fold}/NO_PR'
        # The only executable training function is the pinned original D2 function.
        trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],cell,
                      42+1009*fold,common.digest({'bank':common.sha(path),'binding':binding}),device='cuda')
        ck=torch.load(cell/'BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert ck['binding']['initial_hash']==d['initial_hash']
        print('NO_PATIENT_Z_FOLD_SELECTED',fold,ck['epoch'],ck['threshold'],flush=True)
    common.write_json(a.runtime/'public/TRAIN_STATUS.json',{'status':'ALL_FIVE_TRAIN_VALIDATION_COMPLETE',
        'binding':binding,'outer_evaluated':False,'torch_version':torch.__version__})


def freeze(a,objects):
    lock,ref,common,core,model,trainer,binding=objects
    assert json.loads((a.runtime/'public/TRAIN_STATUS.json').read_text())['status']=='ALL_FIVE_TRAIN_VALIDATION_COMPLETE'
    out=a.runtime/'FROZEN_BEFORE_OUTER.json'
    assert not out.exists(), 'Never replace a freeze'
    rows=[]
    for fold in range(1,6):
        root=a.runtime/f'fold{fold}'; ckpath=root/'NO_PR/BEST_PRIVATE.pt'
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        completed=json.loads((root/'NO_PR/COMPLETE_PRIVATE.json').read_text())
        assert completed['best_sha256']==common.sha(ckpath)
        assert completed['selected_epoch']==ck['epoch'] and completed['threshold']==ck['threshold']
        rows.append({'fold':fold,'epoch':ck['epoch'],'nez_threshold':ck['threshold'],
            'checkpoint_sha256':common.sha(ckpath),'bank_sha256':common.sha(root/'BANK_PRIVATE.pt')})
    freeze={'status':'PASS','binding':binding,'folds':rows,'test_labels_loaded':False,
            'all_selections_precede_outer_inference':True}
    common.write_json(out,freeze);common.write_json(a.runtime/'public/FREEZE_AUDIT.json',freeze)
    print('ALL_FIVE_SELECTIONS_FROZEN',flush=True)


def evaluate(a,objects):
    lock,ref,common,core,model,trainer,binding=objects
    pub=a.runtime/'public'
    assert not (pub/'OUTER_STATUS.json').exists(),'Do not rerun a completed scientific test'
    frozen=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text())
    assert frozen['binding']==binding
    freeze_sha=common.sha(a.runtime/'FROZEN_BEFORE_OUTER.json')
    raw=raw_without_labels(a); cells=[]; visited=[]; sealed={}
    for fold in range(1,6):
        root=a.runtime/f'fold{fold}'; rec=frozen['folds'][fold-1]
        assert common.sha(root/'NO_PR/BEST_PRIVATE.pt')==rec['checkpoint_sha256']
        assert common.sha(root/'BANK_PRIVATE.pt')==rec['bank_sha256']
        d=torch.load(root/'BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        ck=torch.load(root/'NO_PR/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        ix=np.flatnonzero(np.isin(raw['patient'],list(roles(raw,fold)['test'])))
        path=root/'OUTER_PREDICTIONS_PRIVATE.pt'
        if path.exists():
            h=json.loads(path.with_suffix('.sha256.json').read_text())
            assert h=={'binding':freeze_sha,'sha256':common.sha(path)}
            cell=torch.load(path,map_location='cpu',weights_only=False)
            assert cell['binding']==freeze_sha and np.array_equal(cell['indices'],ix)
        else:
            x=frozen_absolute(raw['x'][ix],d['pre']); tm=model.SourceA0().eval();tm.load_state_dict(ck['model'])
            # Match the original D2 outer's CPU inference backend.
            with torch.inference_mode():
                logits=tm(torch.from_numpy(x),torch.from_numpy(model.source_index(raw['center'][ix]))).numpy()
                score=torch.sigmoid(torch.from_numpy(logits)).numpy()
            assert np.isfinite(logits).all() and np.isfinite(score).all()
            cell={'binding':freeze_sha,'indices':ix,'scores':score,'logits':logits,'threshold':ck['threshold']}
            core.torch_write(path,cell)
            common.write_json(path.with_suffix('.sha256.json'),{'binding':freeze_sha,'sha256':common.sha(path)})
        visited.extend(ix.tolist());cells.append(cell);sealed[str(fold)]=common.sha(path)
    assert sorted(visited)==list(range(7635))
    common.write_json(a.runtime/'OUTER_PREDICTION_SEAL.json',{'binding':freeze_sha,'hashes':sealed,'all_scores_complete_before_y':True})
    # Labels become legal for new outer metrics only after the five scores are sealed.
    with np.load(a.features,allow_pickle=True) as f:y=f['y']
    assert len(y)==7635 and int((y==0).sum())==1743
    baseline_seal=json.loads((a.baseline/'OUTER_PREDICTIONS_SEAL.json').read_text())
    rows=[]; histories=[]
    for fold,cell in enumerate(cells,1):
        ix=cell['indices']; oldpath=a.baseline/f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt'
        assert common.sha(oldpath)==baseline_seal['files'][str(fold)]
        old=torch.load(oldpath,map_location='cpu',weights_only=False)
        assert np.array_equal(ix,old['indices'])
        for pat in sorted(set(raw['patient'][ix])):
            k=np.flatnonzero(raw['patient'][ix]==pat)
            k=k[np.argsort(raw['channel'][ix][k].astype(str),kind='stable')]
            si=ix[k]
            for arm,item in [('D2_PR',old),('D2_NO_PATIENT_Z',cell)]:
                rows.append({'patient':pat,'fold':fold,'center':str(raw['center'][si[0]]),'arm':arm,
                    **common.metric(y[si],item['scores'][k],item['threshold'])})
        d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        di=np.flatnonzero(np.isin(raw['patient'],d['patient']))
        assert np.array_equal(y[di],d['y'])
        last=torch.load(a.runtime/f'fold{fold}/NO_PR/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        histories.extend({'fold':fold,**h} for h in last['history'])
    df=pd.DataFrame(rows);core.torch_write(a.runtime/'PATIENT_METRICS_PRIVATE.pt',rows)
    primary=[]
    for arm,part in df.groupby('arm'):
        assert len(part)==part.patient.nunique()==80 and part.channels.sum()==7635
        if arm=='D2_PR':
            orig=json.loads((a.baseline/'public/OUTER_STATUS.json').read_text())['primary_metrics']
            for k in common.METRICS:assert abs(part[k].mean()-orig[k])<1e-12
        draws=common.cluster_bootstrap(part.patient,part[common.METRICS].to_numpy(),10000)
        for j,k in enumerate(common.METRICS):
            primary.append({'arm':arm,'metric':k,'estimate':part[k].mean(),
                'ci_low':np.nanpercentile(draws[:,j],2.5),'ci_high':np.nanpercentile(draws[:,j],97.5),
                'estimable_patients':int(part[k].notna().sum()),'patients':80,'channels':7635})
    pd.DataFrame(primary).to_csv(pub/'OUTER_METRICS.csv',index=False)
    for field,file in [('fold','OUTER_BY_FOLD.csv'),('center','OUTER_BY_CENTER.csv')]:
        grouped=[]
        for (arm,value),part in df.groupby(['arm',field]):
            grouped.append({'arm':arm,field:value,'patients':len(part),'channels':int(part.channels.sum()),**part[common.METRICS].mean().to_dict()})
        pd.DataFrame(grouped).to_csv(pub/file,index=False)
    original=df[df.arm=='D2_PR'].set_index('patient').sort_index()
    ablated=df[df.arm=='D2_NO_PATIENT_Z'].set_index('patient').sort_index()
    assert np.array_equal(original.index,ablated.index)
    delta=ablated[common.METRICS].to_numpy()-original[common.METRICS].to_numpy()
    boot=common.cluster_bootstrap(ablated.index,delta,10000)
    contrast=[]
    for j,k in enumerate(common.METRICS):
        contrast.append({'comparison':'NO_PATIENT_Z_MINUS_PR','metric':k,'delta':np.nanmean(delta[:,j]),
            'ci_low':np.nanpercentile(boot[:,j],2.5),'ci_high':np.nanpercentile(boot[:,j],97.5),
            'patients_improved':int((delta[:,j]>0).sum()),'patients_worsened':int((delta[:,j]<0).sum()),
            'patients_tied':int((delta[:,j]==0).sum()),'draws':10000,'seed':42})
    pd.DataFrame(contrast).to_csv(pub/'PAIRED_BOOTSTRAP.csv',index=False)
    pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False)
    pd.DataFrame(frozen['folds']).to_csv(pub/'MODEL_SELECTION.csv',index=False)
    # Hashes of immutable historical inputs and experiment sources must still match.
    final_setup=setup(a)
    assert final_setup[-1]==binding
    common.write_json(pub/'IMMUTABILITY_AUDIT.json',{'status':'PASS','original_model_inputs_unchanged':True,
        'baseline_predictions_unchanged':True,'five_selections_locked_before_outer':True,
        'all_test_predictions_before_y':True,'test_patients_unique_once':80,'channels':7635,
        'freeze_sha256':freeze_sha,'outer_prediction_seal_sha256':common.sha(a.runtime/'OUTER_PREDICTION_SEAL.json')})
    common.write_json(pub/'OUTER_STATUS.json',{'status':'ONE_FROZEN_NO_PATIENT_Z_OUTER_PASS_COMPLETE',
        'scientific_status':lock['scientific_status'],'primary_no_PR_metrics':ablated[common.METRICS].mean().to_dict(),
        'baseline_PR_metrics':original[common.METRICS].mean().to_dict(),'patients':80,'channels':7635,
        'source_feature_reference_views_retained':True})
    print('NO_PATIENT_Z_OUTER_COMPLETE',json.dumps(ablated[common.METRICS].mean().to_dict()),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','freeze','evaluate'])
    p.add_argument('--runtime',type=Path,default=Path(r'C:\d2_no_patient_z_outer_seed42_runtime'))
    p.add_argument('--original',type=Path,default=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'))
    p.add_argument('--baseline',type=Path,default=Path(r'C:\d2_ictal_outer_seed42_runtime'))
    p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
    p.add_argument('--features',type=Path,default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
    p.add_argument('--protocol',type=Path,default=Path(__file__).parents[1]/'PROTOCOL_LOCK.json')
    a=p.parse_args();torch.set_num_threads(2);objects=setup(a)
    {'prepare':prepare,'train':train_all,'freeze':freeze,'evaluate':evaluate}[a.phase](a,objects)


if __name__=='__main__':main()
