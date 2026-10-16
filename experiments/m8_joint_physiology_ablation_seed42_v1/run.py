"""Four new matched mask pairs/fold; hash reuse; freeze then sealed exploratory Outer."""
import argparse
import faulthandler
import sys
import time
from pathlib import Path
faulthandler.enable()
HERE=Path(__file__).resolve().parent
OLD_CODE=Path(r'D:\m8_physiology_feature_discovery_seed42_v1')
sys.path.insert(0,str(OLD_CODE))
import train_old7_m8 as OLD
from audit_old7 import *
from analysis import CONFIGS,keep

ROOT=Path(r'D:\m8_joint_physiology_ablation_seed42_v1_runtime')
PUB=ROOT/'public'
OLD_OUTER=OLD.OUT/'final_Outer'

def source_hashes():
    return {p.name:sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.json']}

def setup():
    old_lock,c,core,fac,trainer,records,rec,banks,old_binding=OLD.setup_new()
    assert jr(OLD.RUN/'public/INDEPENDENT_RETRAIN_VERIFICATION.json')['status']=='PASS'
    prot=jr(HERE/'PROTOCOL_LOCK.json')
    prior=read(OLD.RUN/'MODEL_SELECTION_PRIVATE.pkl'); assert prior['binding']==old_binding
    rows=[dict(r) for r in prior['selections'] if r['variant'] in prot['reuse'].values()]
    assert len(rows)==20
    reuse={}
    for config,name in prot['reuse'].items():
        for r in rows:
            if r['variant']!=name: continue
            assert r['keep']==keep(config)
            for key in ['bank','D1_checkpoint','M8_checkpoint']:
                assert sha(r[key])==r[key+'_sha256']
            q=read(r['validation_path'])
            assert q['D1_checkpoint_sha256']==r['D1_checkpoint_sha256'] and q['M8_checkpoint_sha256']==r['M8_checkpoint_sha256']
            reuse[config,r['fold']]=dict(r,config=config,reused=True)
    ROOT.mkdir(exist_ok=True); PUB.mkdir(exist_ok=True)
    binding=dict(source=source_hashes(),prior_binding=old_binding,metric_engine_sha256=sha(OLD.EV.__file__),
        old_model_selection_sha256=sha(OLD.RUN/'MODEL_SELECTION_PRIVATE.pkl'),old_verifier_sha256=sha(OLD.RUN/'public/INDEPENDENT_RETRAIN_VERIFICATION.json'),
        reused={config+'_'+str(f):{key:r[key+'_sha256'] for key in ['bank','D1_checkpoint','M8_checkpoint']} for (config,f),r in reuse.items()})
    fp=ROOT/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists(): assert jr(fp)==binding
    else: write(fp,binding)
    return prot,c,core,fac,trainer,records,rec,banks,binding,reuse

def train():
    prot,c,core,fac,trainer,records,rec,banks,binding,reuse=setup()
    if (ROOT/'MODEL_FREEZE_PRIVATE.pkl').exists():
        fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl'); assert fr['binding']==binding
        print('ALL40_MODEL_PAIRS_ALREADY_FROZEN_NO_REFIT',flush=True); return
    selections=list(reuse.values());history=[]
    store=OLD.P3.Inputs(OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT); store.arm='M1_SCALED_DGM'
    for config in prot['new']:
        mask=keep(config)
        for fold in range(1,6):
            original=banks[fold-1]
            d=torch.load(OLD.BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
            for k in ['patient','channel','y','train','val']: assert np.array_equal(d[k],original[k])
            masked=d['x']*np.repeat(np.asarray(mask,np.float32),2)[None]
            d=dict(d,x=masked,archive_indices=original['archive_indices'],roles=original['roles'])
            assert set(d['patient']).isdisjoint(d['roles']['test'])
            cell=ROOT/config/f'fold{fold}'; bp=cell/'BANK_PRIVATE.pt'
            if bp.exists():
                old=torch.load(bp,map_location='cpu',weights_only=False)
                assert old['binding']==binding and np.array_equal(old['x'],masked)
            else: core.torch_write(bp,dict(d,binding=binding))
            write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING',config=config,fold=fold,new_pairs_complete=len(selections)-20,new_pairs_total=20))
            factory=trainer.SourceA0
            try:
                trainer.SourceA0=lambda seed=42:OLD.CompactDelta(fac(seed).network,records)
                base,best,h=trainer.train(masked,d['y'],d['patient'],d['g'],d['train'],d['val'],cell/'D1_TRAIN',42+1009*fold,
                    c.digest(dict(binding=binding,config=config,bank_sha256=sha(bp))),device='cuda',penalty=0.)
            finally: trainer.SourceA0=factory
            base.eval()
            with torch.inference_mode(): base_logit=base(torch.as_tensor(masked[d['val']],device='cuda')).cpu().numpy()
            assert core.select_threshold(d['y'][d['val']],torch.sigmoid(torch.from_numpy(base_logit)).numpy(),d['patient'][d['val']])['threshold']==best['threshold']
            cp=cell/'D1_TRAIN/BEST_PRIVATE.pt'
            scale=torch.tensor(read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
            if fold==1 and not (PUB/f'ADMISSION_{config}.json').exists():
                model,original_model=OLD.build(core,fac,records,cp,fold,mask); del original_model
                tr=d['train']; ii=tr[d['patient'][tr]==sorted(set(d['patient'][tr]))[0]]
                raw,rawmask,times,delta,perms=store.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
                model.eval(); x=torch.as_tensor(d['x'][ii],device='cuda')
                with torch.inference_mode():
                    y1=model(x,raw,rawmask,times,delta,scale)
                    changed=delta.clone(); changed[...,np.flatnonzero(np.asarray(mask)==0)]=123
                    xx=x.clone(); xx[:,np.repeat(np.asarray(mask)==0,2)]=321
                    y2=model(xx,raw,rawmask,times,changed,scale)
                    assert torch.equal(y1[0],y2[0]) and torch.equal(y1[1],y2[1])
                    assert torch.equal(model.features(raw,rawmask,times,delta,scale)[0],model.features(raw,rawmask,times,changed,scale)[0])
                del raw,rawmask,times,delta,perms,x,xx,y1,y2,changed
                smoke=OLD.P3.admission(model,d,store,scale,core,'M8_MACRO_AWARE')
                write(PUB/f'ADMISSION_{config}.json',dict(status='PASS',simultaneously_removed_D1_and_Gate_inputs_independent=True,
                    scratch_initialization_matches_original=True,smoke=smoke))
                del model
            model,original_model=OLD.build(core,fac,records,cp,fold,mask); del original_model
            runtime=ROOT/config;(runtime/'results').mkdir(parents=True,exist_ok=True)
            mb,mh=OLD.P3.fit(model,d,store,scale,cell/'M8_TRAIN',42+1009*fold,binding,core,runtime,'M8_MACRO_AWARE',fold)
            mcp=cell/'M8_TRAIN/BEST_PRIVATE.pt'; pred=OLD.P3.EV.predict(model,d,d['val'],store,scale)
            assert np.max(np.abs(pred['base_logit']-base_logit))<=1e-6
            pp=cell/'VALIDATION_PRIVATE.pkl'; q=dict(D1_logit=base_logit,M8_logit=pred['logit'],D1_checkpoint_sha256=sha(cp),
                M8_checkpoint_sha256=sha(mcp),validation_archive_indices=d['archive_indices'][d['val']])
            if pp.exists():
                old=read(pp); assert all(old[k]==q[k] for k in ['D1_checkpoint_sha256','M8_checkpoint_sha256'])
                assert np.array_equal(old['D1_logit'],q['D1_logit']) and np.array_equal(old['M8_logit'],q['M8_logit'])
            else: save(pp,q)
            selections.append(dict(config=config,variant='JOINT_'+config,fold=fold,keep=mask,reused=False,
                D1_checkpoint=str(cp),D1_checkpoint_sha256=sha(cp),M8_checkpoint=str(mcp),M8_checkpoint_sha256=sha(mcp),
                bank=str(bp),bank_sha256=sha(bp),validation_path=str(pp),D1_threshold=best['threshold'],D1_epoch=best['epoch'],D1_epochs_run=len(h),
                M8_threshold=mb['threshold'],M8_epoch=mb['epoch'],M8_epochs_run=len(mh)))
            history += [dict(config=config,fold=fold,head='D1',**row) for row in h]
            history += [dict(config=config,head='M8',**{k:v for k,v in row.items() if k!='arm'}) for row in mh]
            save(ROOT/'PROGRESS_PRIVATE.pkl',dict(binding=binding,selections=selections))
            pd.DataFrame(history).to_csv(PUB/'NEW_TRAINING_HISTORY.csv',index=False)
            print('JOINT_PAIR_COMPLETE',config,fold,len(selections)-20,'of20NEW',flush=True)
            del model,base,d; gc.collect();torch.cuda.empty_cache()
    assert len(selections)==40
    freeze=dict(binding=binding,selections=sorted(selections,key=lambda r:(r['config'],r['fold'])),new_pairs=20,reused_pairs=20)
    save(ROOT/'MODEL_FREEZE_PRIVATE.pkl',freeze)
    pd.DataFrame([{k:r[k] for k in ['config','fold','reused','D1_epoch','M8_epoch','D1_threshold','M8_threshold','D1_epochs_run','M8_epochs_run','D1_checkpoint_sha256','M8_checkpoint_sha256']} for r in selections]).to_csv(PUB/'MODEL_SELECTION.csv',index=False)
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL40_MODEL_PAIRS_FROZEN',new_models=40,reused_models=40,all_epochs_preserved=True,no_new_Outer=True))

def verify_training():
    prot,c,core,fac,trainer,records,rec,banks,binding,reuse=setup()
    fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');assert fr['binding']==binding
    store=OLD.P3.Inputs(OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);store.arm='M1_SCALED_DGM'
    frames=[];maximum=0.;epoch_count=0
    for r in fr['selections']:
        f=r['fold'];d=torch.load(r['bank'],map_location='cpu',weights_only=False)
        original=torch.load(OLD.BASELINE/f'D1/fold{f}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(d['x'],original['x']*np.repeat(np.asarray(r['keep'],np.float32),2)[None])
        assert set(d['patient'][d['train']])==banks[f-1]['roles']['fit'] and set(d['patient'][d['val']])==banks[f-1]['roles']['validation']
        assert set(d['patient']).isdisjoint(banks[f-1]['roles']['test'])
        ck={}
        for head in ['D1','M8']:
            cp=Path(r[head+'_checkpoint']);assert sha(cp)==r[head+'_checkpoint_sha256']
            ck[head]=torch.load(cp,map_location='cpu',weights_only=False)
            done=jr(cp.parent/'COMPLETE_PRIVATE.json');assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
            last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            hh=last['history']; assert 6<=len(hh)<=30 and [h['epoch'] for h in hh]==list(range(1,len(hh)+1))
            win=max(hh,key=lambda h:(h['macro_f1'],h['ez_f1'],-h['epoch']))
            assert win['epoch']==ck[head]['epoch']==r[head+'_epoch'] and win['threshold']==r[head+'_threshold']==ck[head]['threshold']
            if not r['reused']: epoch_count+=len(hh)
        model,unused=OLD.build(core,fac,records,Path(r['D1_checkpoint']),f,r['keep']);del unused
        model.load_state_dict(ck['M8']['model']);model.eval()
        assert core.state_hash(model.base.state_dict())==core.state_hash(ck['D1']['model'])
        scale=torch.tensor(read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{f}_PRIVATE.pkl')['scale'],device='cuda')
        pred=OLD.P3.EV.predict(model,d,d['val'],store,scale);q=read(r['validation_path'])
        with torch.inference_mode(): base=model.base(torch.as_tensor(d['x'][d['val']],device='cuda')).cpu().numpy()
        for head,logit in [('D1',base),('M8',pred['logit'])]:
            error=float(np.max(np.abs(logit-q[head+'_logit'])));maximum=max(maximum,error);assert error<=1e-6
            pn=torch.sigmoid(torch.from_numpy(q[head+'_logit'])).numpy()
            assert core.select_threshold(d['y'][d['val']],pn,d['patient'][d['val']])['threshold']==r[head+'_threshold']
            frames.append(pd.DataFrame(dict(scope='development',config=r['config'],head=head,model=r['config'],version=head,fold=f,
                patient=d['patient'][d['val']],channel=d['channel'][d['val']],center=d['center'][d['val']],y=(d['y'][d['val']]==0).astype(int),
                probability=1-pn.astype(float),prediction=pn<r[head+'_threshold'])))
        del model,d,ck;gc.collect();torch.cuda.empty_cache()
        print('JOINT_VALIDATION_REPLAY_PASS',r['config'],f,flush=True)
    frame=pd.concat(frames,ignore_index=True);save(ROOT/'DEVELOPMENT_PREDICTIONS_PRIVATE.pkl',frame)
    for p,h in binding['prior_binding']['protected'].items():assert sha(p)==h
    write(PUB/'TRAINING_VERIFICATION.json',dict(status='PASS',model_pairs=40,actual_new_models=40,reused_models=40,new_complete_epochs=epoch_count,
        all80_validation_models_replayed=True,maximum_logit_error=maximum,thresholds_epochs_original_roles_verified=True,
        model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),source=binding['source']))

def outer():
    prot,c,core,fac,trainer,records,rec,banks,binding,reuse=setup()
    assert jr(PUB/'TRAINING_VERIFICATION.json')['status']=='PASS'
    choice=jr(PUB/'DEVELOPMENT_SELECTION_FROZEN.json');assert choice['status']=='FROZEN_BEFORE_NEW_OUTER'
    assert choice['model_freeze_sha256']==sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl')
    fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl'); assert fr['binding']==binding
    oldfr=jr(OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json');oldseal=jr(OLD_OUTER/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    assert oldseal['freeze_sha256']==sha(OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    obj=dict(binding=binding,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),candidate_decision_sha256=sha(PUB/'DEVELOPMENT_SELECTION_FROZEN.json'),
        old_freeze_sha256=sha(OLD_OUTER/'FROZEN_BEFORE_OUTER_PRIVATE.json'),old_seal_sha256=sha(OLD_OUTER/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),
        all80_head_fold_cells_frozen=True,scientific_status='EXPLORATORY_REPEATED_TEST')
    target=ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if target.exists():assert jr(target)==obj
    else:write(target,obj)
    fh=sha(target);files={};timing=[];outdir=ROOT/'outer_scores_private';outdir.mkdir(exist_ok=True)
    store=OLD.P3.Inputs(OLD.BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),ROOT);store.arm='M1_SCALED_DGM'
    for r in fr['selections']:
        config=r['config'];f=r['fold'];t=time.perf_counter();meta=rec['meta'];roles=banks[f-1]['roles']
        ix=np.flatnonzero(np.isin(meta['patient'],list(roles['test'])))
        assert set(meta['patient'][ix])==roles['test'] and roles['test'].isdisjoint(roles['fit']|roles['validation'])
        for key in ['bank','D1_checkpoint','M8_checkpoint']:assert sha(r[key])==r[key+'_sha256']
        reused_patient=0
        if r['reused']:
            name='LEGACY_'+r['variant'];oldr=next(q for q in oldfr['selections'] if q['model']==name and q['fold']==f)
            assert oldr['checkpoint_sha256']==r['M8_checkpoint_sha256'] and oldr['base_sha256']==r['D1_checkpoint_sha256']
            pp=OLD_OUTER/'armfold_scores_private'/f'{name}_fold{f}.pkl';assert sha(pp)==oldseal['files'][name+'_'+str(f)]
            q=read(pp);assert q['freeze_sha256']==obj['old_freeze_sha256'] and np.array_equal(q['indices'],ix)
            logits=dict(D1=q['base_logit'],M8=q['logit']);reused_patient=len(roles['test'])
        else:
            bp=torch.load(r['bank'],map_location='cpu',weights_only=False);assert set(bp['patient']).isdisjoint(roles['test'])
            z=core.patient_z(rec['R14'][ix],meta['patient'][ix]);pre=bp['pre']
            x=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)
            x*=np.repeat(np.asarray(r['keep'],np.float32),2)[None]
            model,unused=OLD.build(core,fac,records,Path(r['D1_checkpoint']),f,r['keep']);del unused
            model.load_state_dict(torch.load(r['M8_checkpoint'],map_location='cuda',weights_only=False)['model']);model.eval()
            before=core.state_hash(model.state_dict())
            scale=torch.tensor(read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{f}_PRIVATE.pkl')['scale'],device='cuda')
            logits={h:np.empty(len(ix),np.float32) for h in ['D1','M8']}
            for pid in sorted(roles['test']):
                jj=np.flatnonzero(meta['patient'][ix]==pid);ii=ix[jj]
                cb=dict(freeze=fh,config=config,fold=f,checkpoint=r['M8_checkpoint_sha256'],base=r['D1_checkpoint_sha256'],indices=hashlib.sha256(ii.tobytes()).hexdigest(),
                    x=hashlib.sha256(x[jj].tobytes()).hexdigest())
                token=hashlib.sha256(str(pid).encode()).hexdigest()[:24];path=ROOT/'patient_scores_private'/config/f'fold{f}'/(token+'.pkl')
                if path.exists():q=read(path);assert q['binding']==cb;reused_patient+=1
                else:
                    raw,mask,times,delta,perms=store.get(str(pid),meta['channel'][ii].astype(str).tolist(),'cuda')
                    with torch.inference_mode(): out=model(torch.as_tensor(x[jj],device='cuda'),raw,mask,times,delta,scale)
                    q=dict(binding=cb,indices=ii,M8=out[0].cpu().numpy(),D1=out[2].cpu().numpy());save(path,q)
                    del raw,mask,times,delta,perms,out
                assert np.array_equal(q['indices'],ii)
                for head in ['D1','M8']:assert np.isfinite(q[head]).all();logits[head][jj]=q[head]
            assert core.state_hash(model.state_dict())==before
            del model,bp;gc.collect();torch.cuda.empty_cache()
        for head in ['D1','M8']:
            result=dict(indices=ix,logit=logits[head],score_nez=torch.sigmoid(torch.from_numpy(logits[head])).numpy(),
                threshold=r[head+'_threshold'],checkpoint_sha256=r[head+'_checkpoint_sha256'],freeze_sha256=fh,reused=r['reused'])
            key=config+'_'+head+'_'+str(f);pp=outdir/(key+'.pkl')
            if pp.exists():
                q=read(pp);assert q['freeze_sha256']==fh and np.array_equal(q['logit'],result['logit']) and q['threshold']==result['threshold']
            else:save(pp,result)
            files[key]=sha(pp)
        timing.append(dict(config=config,fold=f,reused_model=r['reused'],reused_patient_pairs=reused_patient,seconds=time.perf_counter()-t))
        print('JOINT_OUTER_PAIR_SEALED',config,f,len(files),'of80CELLS',flush=True)
    assert len(files)==80
    seal=dict(freeze_sha256=fh,files=files,all80_cells_sealed_before_full_labels=True,new_inference_pairs=20,reused_pairs=20)
    sp=ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'
    if sp.exists():assert jr(sp)==seal
    else:write(sp,seal)
    # Metadata only; no full cohort labels are loaded before the seal above.
    save(ROOT/'OUTER_METADATA_PRIVATE.pkl',dict(meta=rec['meta'],archive_sha256=rec['binding']['archive']))
    pd.DataFrame(timing).to_csv(PUB/'OUTER_INFERENCE_TIMING.csv',index=False)
    write(PUB/'OUTER_INFERENCE_STATUS.json',dict(status='ALL80_CELLS_SEALED',freeze_sha256=fh,seal_sha256=sha(sp),all80_cells_sealed_before_full_labels=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','verify','outer']);a=p.parse_args()
    {'train':train,'verify':verify_training,'outer':outer}[a.stage]()
