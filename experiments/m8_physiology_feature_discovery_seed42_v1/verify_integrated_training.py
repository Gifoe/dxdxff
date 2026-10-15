"""Independent frozen-input/epoch/threshold and validation-forward replay; no fit/Test."""
import train_integrated_v2 as T
from integrated_data_v2 import *


def main():
    lock,c,core,fac,trainer,records,rec,banks,prior=T.setup_new()
    run=T.RUN;pub=T.PUB;fr=jr(run/'MODEL_FREEZE_PRIVATE.json');assert fr['binding']['prior']==prior
    assert fr['all45_logical_models_frozen'] and len(fr['selections'])==45
    for n,h in fr['binding']['source'].items():assert sha(TASK/n)==h
    admission=jr(pub/'REAL_FIT_ADMISSION.json');assert admission['status']=='PASS' and admission['all_actual_admissions_before_training']
    plan=read(run/'PREPARED_PLAN_PRIVATE.pkl');assert plan['binding']==fr['binding']
    pool=FeaturePool(rec);pool.populate();assert plan['pool_sources']==pool.files
    selection=jr(OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json');seen={};maximum=0.;basekeys=set();checks=[]
    for row in fr['selections']:
        fold=row['fold'];variant=row['variant'];sig=row['signature'];original=dict(banks[fold-1],fold=fold)
        for key in ['D1_checkpoint','M8_checkpoint','bank','validation_path']:assert sha(row[key])==row[key+'_sha256']
        dmap,gmap,wrong=recipes(selection['folds'][str(fold)])[variant]
        assert dmap==row['dmap'] and gmap==row['gmap'] and wrong==row['wrong_pair']
        d=torch.load(row['bank'],map_location='cpu',weights_only=False);assert d['binding']==fr['binding']
        rr=pool.mapped_summary(dmap,wrong)[original['archive_indices']]
        expected=pool.bank(original,dmap,wrong,core)
        assert np.array_equal(d['x'],expected['x'])
        for key in ['imputer_statistics','mean','scale','var']:assert np.array_equal(d['pre'][key],expected['pre'][key])
        assert set(d['pre']['fit_patient_ids'])==original['roles']['fit']
        assert set(d['patient']).isdisjoint(original['roles']['test'])
        # Independently reproduce patient local statistics and FIT-only sufficient moments.
        z=np.empty(rr.shape,np.float64)
        for pid in sorted(set(d['patient'])):
            ii=d['patient']==pid;a=rr[ii].astype(np.float64);sd=a.std(0)
            z[ii]=(a-a.mean(0))/np.where(sd>1e-8,sd,1.)
        fit=z[d['train']]
        assert np.allclose(fit.mean(0),d['pre']['mean'],atol=1e-12,rtol=0)
        assert np.allclose(fit.var(0),d['pre']['var'],atol=1e-12,rtol=0)
        assert np.array_equal(transform_frozen(rr,d['patient'],d['pre'],core),d['x'])
        assert np.array_equal(pool.scale(original,gmap),np.asarray(row['scale'],np.float32))
        for head in ['D1','M8']:
            cp=Path(row[head+'_checkpoint']);q=torch.load(cp,map_location='cpu',weights_only=False)
            done=jr(cp.parent/'COMPLETE_PRIVATE.json');assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
            last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert 6<=len(last['history'])<=30 and [r['epoch'] for r in last['history']]==list(range(1,len(last['history'])+1))
            winner=max(last['history'],key=lambda h:(h['macro_f1'],h['ez_f1'],-h['epoch']))
            assert winner['epoch']==q['epoch']==row[head+'_epoch'] and winner['threshold']==q['threshold']==row[head+'_threshold']
        if sig in seen:
            assert row['M8_checkpoint_sha256']==seen[sig]['M8_checkpoint_sha256'] and row['M8_threshold']==seen[sig]['M8_threshold']
            assert row['alias_of']==seen[sig]['variant'];continue
        assert row['alias_of'] is None
        model=T.make(core,fac,records,fold,Path(row['D1_checkpoint']))
        cp=torch.load(row['M8_checkpoint'],map_location='cuda',weights_only=False);model.load_state_dict(cp['model']);model.eval()
        base=torch.load(row['D1_checkpoint'],map_location='cpu',weights_only=False)
        assert core.state_hash(model.base.state_dict())==core.state_hash(base['model'])
        store=MappedInputs(pool,gmap,wrong);scale=torch.tensor(row['scale'],device='cuda')
        pred=T.P3.EV.predict(model,d,d['val'],store,scale);cached=read(row['validation_path'])
        for key,field in [('logit','M8_logit'),('base_logit','D1_logit')]:
            error=float(np.max(np.abs(pred[key]-cached[field])));maximum=max(maximum,error);assert error<=1e-6
        assert np.array_equal(cached['validation_archive_indices'],d['archive_indices'][d['val']])
        for head in ['D1','M8']:
            pn=torch.sigmoid(torch.from_numpy(cached[head+'_logit'])).numpy()
            chosen=core.select_threshold(d['y'][d['val']],pn,d['patient'][d['val']]);assert chosen['threshold']==row[head+'_threshold']
        seen[sig]=row;basekeys.add(row['base_key']);checks.append(dict(variant=variant,fold=fold,status='PASS'))
        print('INTEGRATED_FROZEN_VALIDATION_REPLAY_PASS',variant,fold,flush=True)
        del model,d,cp,base,store;pool.raw.cache.clear();gc.collect();torch.cuda.empty_cache()
    assert len(seen)==admission['unique_M8_inputs'] and len(basekeys)==admission['unique_D1_inputs']
    for path,h in prior['protected'].items():assert sha(path)==h
    write(pub/'INDEPENDENT_TRAINING_VERIFICATION.json',dict(status='PASS',logical_cells=45,actual_M8_cells=len(seen),actual_D1_cells=len(basekeys),
        aliases=45-len(seen),actual_validation_inference_replayed=True,maximum_logit_replay_error=maximum,registered_tolerance=1e-6,
        all_per_FIT_feature_maps_and_routes_exact=True,all_preprocessing_and_gate_scales_reproduced=True,independent_FIT_moments_verified=True,
        epoch_thresholds_reconstructed=True,all_bases_frozen=True,original_splits_preserved=True,no_training_Outer_read=True,
        verifier_sha256=sha(__file__),cells=checks,full_goal_complete=False))
    print('ALL45_INTEGRATED_FEATURE_PROCEDURES_INDEPENDENT_TRAINING_VERIFICATION_PASS',flush=True)


if __name__=='__main__':main()
