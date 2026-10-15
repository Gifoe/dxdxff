"""Nine prospectively fixed M8 procedures; validation training only, no Outer reader."""
import argparse
import time
from train_old7_m8 import setup_new,normalize_rng,P3,P3CODE
from integrated_data_v2 import *
from integrated_model import SlotBase,IntegratedM8

RUN=OUT/'integrated_M8_v2';PUB=RUN/'public'


def new_base(core,fac,records,fold):
    seed=42+1009*fold;core.seed_all(seed)
    return SlotBase(fac(seed).network,records).cuda()


def make(core,fac,records,fold,checkpoint=None):
    seed=42+1009*fold
    fresh=P3.make(core,fac,records,BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',seed,'M8_MACRO_AWARE')
    base=new_base(core,fac,records,fold)
    if checkpoint is not None:base.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
    model=IntegratedM8(base,fresh).cuda()
    for name in ['cnn','head']:
        assert core.state_hash(getattr(model,name).state_dict())==core.state_hash(getattr(fresh,name).state_dict())
    del fresh;return model


def admission(model,d,store,scale,core,variant,fold):
    tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
    x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda')
    frozen=core.state_hash(model.base.state_dict());model.eval()
    with torch.inference_mode():q=model(x,raw,mask,times,delta,scale);assert torch.equal(q[0],q[2])
    weight=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1)
    fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device='cuda'))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001);norms=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for step in range(3):
        model.train();opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale)
        loss=P3.objective(q[0],y,fn,'M8_MACRO_AWARE');assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        ns={n:float(p.grad.norm().cpu()) for n,p in model.named_parameters() if p.grad is not None}
        assert ns['head.2.weight']>0
        if step>=1:assert ns['cnn.0.weight']>0 and ns['head.0.weight']>0 and ns['gate.network.2.weight']>0
        if step>=2:
            assert ns['gate.network.0.weight']>0
            if torch.count_nonzero(delta[...,7:]):assert float(model.gate.network[0].weight.grad[:,7:].norm())>0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();norms.append(ns)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    assert core.state_hash(model.base.state_dict())==frozen
    return dict(variant=variant,fold=fold,status='PASS',real_whole_FIT_patient=True,three_discarded_steps=True,
        total=11690,trainable=9577,initial_residual_identity=True,base_frozen=True,gradient_norms=norms,
        three_step_seconds=seconds,valid_channel_windows=int(mask.sum()))


def prepare():
    PUB.mkdir(parents=True,exist_ok=True);(RUN/'results').mkdir(exist_ok=True)
    oldlock,c,core,fac,trainer,records,rec,banks,prior=setup_new()
    lock=jr(TASK/'M8_INTEGRATION_LOCK.json');selection_path=OUT/'SELECTED_FEATURES_FIT_ONLY_PRIVATE.json'
    assert sha(selection_path)==lock['FIT_feature_freeze_sha256']
    assert jr(OUT/'public/NONLINEAR_INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert jr(OUT/'public/INTEGRATED_MODEL_SYNTHETIC_AUDIT.json')['status']=='PASS'
    source=['M8_INTEGRATION_LOCK.json','integrated_model.py','integrated_data.py','train_integrated.py','integrated_data_v2.py','train_integrated_v2.py','test_pairing_index.py','train_old7_m8.py','old7_m8_model.py','audit_old7.py']
    binding=dict(source={n:sha(TASK/n) for n in source},prior=prior,feature_freeze=sha(selection_path),
        candidate_registry=sha(OUT/'CANDIDATE_REGISTRY_PRIVATE.pkl'),numeric_registry=sha(OUT/'NUMERICAL_CONTROL_REGISTRY_PRIVATE.pkl'),
        synthetic=sha(OUT/'public/INTEGRATED_MODEL_SYNTHETIC_AUDIT.json'),screen_verification=sha(OUT/'public/NONLINEAR_INDEPENDENT_VERIFICATION.json'))
    fp=RUN/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert jr(fp)==binding
    else:write(fp,binding)
    pool=FeaturePool(rec);pool.populate();selection=jr(selection_path);prepared=[];audits=[];aliases={}
    for fold,original in enumerate(banks,1):
        original=dict(original,fold=fold);legacy=torch.load(BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        for variant,(dmap,gmap,wrong) in recipes(selection['folds'][str(fold)]).items():
            assert variant in lock['variants'];d=pool.bank(original,dmap,wrong,core)
            assert np.array_equal(transform_frozen(pool.mapped_summary(dmap,wrong)[d['archive_indices']],d['patient'],d['pre'],core),d['x'])
            if dmap[:7]==list(range(7)):assert np.array_equal(d['x'][:,:14],legacy['x'])
            scale=pool.scale(original,gmap);store=MappedInputs(pool,gmap,wrong)
            bp=RUN/variant/f'fold{fold}/BANK_PRIVATE.pt'
            if bp.exists():q=torch.load(bp,map_location='cpu',weights_only=False);assert q['binding']==binding and np.array_equal(q['x'],d['x'])
            else:core.torch_write(bp,dict(d,binding=binding))
            sig=c.digest(dict(fold=fold,x=digest_array(d['x']),gate_map=gmap,wrong=wrong,scale=digest_array(scale)))
            basekey=c.digest(dict(fold=fold,x=digest_array(d['x']),pre={k:digest_array(d['pre'][k]) for k in ['imputer_statistics','mean','scale','var']}))
            alias=aliases.setdefault(sig,dict(variant=variant,fold=fold))
            row=dict(variant=variant,fold=fold,dmap=dmap,gmap=gmap,wrong_pair=wrong,bank=str(bp),bank_sha256=sha(bp),
                signature=sig,base_key=basekey,alias=alias,scale=scale.tolist())
            prepared.append(row)
            ap=RUN/'admissions_private'/f'{sig}.json'
            if not ap.exists():
                model=make(core,fac,records,fold)
                a=admission(model,d,store,torch.tensor(scale,device='cuda'),core,variant,fold)
                if variant=='SLOT10_OLD7':
                    tr=d['train'];ii=tr[d['patient'][tr]==sorted(set(d['patient'][tr]))[0]]
                    raw,mask,times,delta,_=store.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
                    fresh=make(core,fac,records,fold);orig=P3.make(core,fac,records,BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold,'M8_MACRO_AWARE')
                    with torch.inference_mode():error=float((fresh.features(raw,mask,times,delta,torch.tensor(scale,device='cuda'))[0]-orig.features(raw,mask,times,delta[...,:7],torch.tensor(scale[:7],device='cuda'))[0]).abs().max())
                    assert error==0;a['original_Raw_prefix_max_error']=error;del fresh,orig,raw,mask,times,delta
                write(ap,dict(binding_sha256=c.digest(binding),signature=sig,audit=a));del model
            aa=jr(ap);assert aa['binding_sha256']==c.digest(binding) and aa['signature']==sig;audits.append(aa['audit'])
            del d,store;pool.raw.cache.clear();gc.collect();torch.cuda.empty_cache()
            print('INTEGRATED_REAL_FIT_ADMISSION_PASS',variant,fold,'alias',alias,flush=True)
    assert len(prepared)==45
    save(RUN/'PREPARED_PLAN_PRIVATE.pkl',dict(binding=binding,cells=prepared,pool_sources=pool.files))
    write(PUB/'REAL_FIT_ADMISSION.json',dict(status='PASS',logical_cells=45,unique_M8_inputs=len(aliases),
        unique_D1_inputs=len(set(r['base_key'] for r in prepared)),all_feature_lists_FIT_only=True,all_actual_admissions_before_training=True,
        base_parameters=2113,correction_parameters=9577,total_parameters=11690,relative_parameters=11690/11018,audits=audits))
    cover=pd.DataFrame(pool.coverage);rows=[]
    for name,g in [('ALL',cover)]+list(cover.groupby('center')):
        rows.append(dict(center=name,patients=len(g),channels=int(g.channels.sum()),moved_channels=int(g.moved_channels.sum()),
            unchanged_singleton_channels=int(g.unchanged_singleton_channels.sum()),movement_fraction=float(g.moved_channels.sum()/g.channels.sum())))
    pd.DataFrame(rows).to_csv(PUB/'WRONG_PAIR_COVERAGE.csv',index=False)
    return c,core,fac,trainer,records,rec,banks,binding,pool,prepared


def train():
    c,core,fac,trainer,records,rec,banks,binding,pool,plan=prepare();selections=[];history=[];completed={};bases={}
    if (RUN/'MODEL_FREEZE_PRIVATE.json').exists():
        fr=jr(RUN/'MODEL_FREEZE_PRIVATE.json');assert fr['binding']==binding
        for r in fr['selections']:
            for key in ['D1_checkpoint','M8_checkpoint','bank','validation_path']:assert sha(r[key])==r[key+'_sha256']
        print('INTEGRATED45_MODELS_ALREADY_FROZEN_NO_REFIT',flush=True);return
    original_factory=trainer.SourceA0
    for row in plan:
        variant=row['variant'];fold=row['fold'];seed=42+1009*fold;sig=row['signature'];basekey=row['base_key']
        if sig in completed:
            source=completed[sig];r=dict(source,variant=variant,alias_of=source['variant'],bank=row['bank'],bank_sha256=row['bank_sha256'])
            selections.append(r);print('INTEGRATED_LOGICAL_ALIAS',variant,fold,'of',source['variant'],flush=True);continue
        d=torch.load(row['bank'],map_location='cpu',weights_only=False);assert d['binding']==binding
        basecell=RUN/'shared_D1_private'/f'fold{fold}_{basekey}'
        try:
            trainer.SourceA0=lambda seed=42:SlotBase(fac(seed).network,records)
            base,best,bh=trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],basecell,seed,
                c.digest(dict(binding=binding,basekey=basekey)),device='cuda',penalty=0.)
        finally:trainer.SourceA0=original_factory
        cp=basecell/'BEST_PRIVATE.pt';model=make(core,fac,records,fold,cp)
        with torch.inference_mode():bl=base(torch.as_tensor(d['x'][d['val']],device='cuda')).cpu().numpy()
        assert core.select_threshold(d['y'][d['val']],torch.sigmoid(torch.from_numpy(bl)).numpy(),d['patient'][d['val']])['threshold']==best['threshold']
        store=MappedInputs(pool,row['gmap'],row['wrong_pair']);scale=torch.tensor(row['scale'],device='cuda');cell=RUN/variant/f'fold{fold}'
        mb,mh=P3.fit(model,d,store,scale,cell/'M8_TRAIN',seed,dict(binding=binding,cell_signature=sig,D1_sha256=sha(cp)),core,RUN,'M8_MACRO_AWARE',fold)
        pred=P3.EV.predict(model,d,d['val'],store,scale);assert np.max(np.abs(pred['base_logit']-bl))<=1e-6
        mcp=cell/'M8_TRAIN/BEST_PRIVATE.pt';vp=cell/'VALIDATION_PRIVATE.pkl'
        q=dict(signature=sig,D1_checkpoint_sha256=sha(cp),M8_checkpoint_sha256=sha(mcp),D1_logit=bl,M8_logit=pred['logit'],validation_archive_indices=d['archive_indices'][d['val']])
        if vp.exists():old=read(vp);assert old['signature']==sig and old['M8_checkpoint_sha256']==sha(mcp)
        else:save(vp,q)
        r=dict(row,alias_of=None,D1_checkpoint=str(cp),D1_checkpoint_sha256=sha(cp),M8_checkpoint=str(mcp),M8_checkpoint_sha256=sha(mcp),
            validation_path=str(vp),validation_path_sha256=sha(vp),D1_epoch=best['epoch'],D1_threshold=best['threshold'],M8_epoch=mb['epoch'],M8_threshold=mb['threshold'],
            D1_epochs_run=len(bh),M8_epochs_run=len(mh),D1_reused=basekey in bases)
        if basekey not in bases:history.extend(dict(variant=variant,fold=fold,head='D1',**r) for r in bh)
        history.extend(dict(variant=variant,head='M8',**{k:v for k,v in r.items() if k!='arm'}) for r in mh)
        bases[basekey]=str(cp);completed[sig]=r;selections.append(r)
        save(RUN/'MODEL_SELECTION_PRIVATE.pkl',dict(binding=binding,selections=selections))
        public=[{k:v for k,v in r.items() if k not in ['D1_checkpoint','M8_checkpoint','bank','validation_path','alias','dmap','gmap','scale']} for r in selections]
        pd.DataFrame(public).to_csv(PUB/'MODEL_SELECTION.csv',index=False);pd.DataFrame(history).to_csv(PUB/'TRAINING_HISTORY.csv',index=False)
        write(PUB/'STATUS.json',dict(status='TRAINING',logical_cells_complete=len(selections),logical_cells=45,actual_M8_complete=len(completed),actual_D1_complete=len(bases),no_Outer=True,full_goal_complete=False))
        print('INTEGRATED_M8_CELL_COMPLETE',variant,fold,len(selections),'of45',flush=True)
        del base,model,d,store;pool.raw.cache.clear();gc.collect();torch.cuda.empty_cache()
    assert len(selections)==45
    for r in selections:
        for key in ['D1_checkpoint','M8_checkpoint','bank','validation_path']:assert sha(r[key])==r[key+'_sha256']
    fr=dict(binding=binding,selections=selections,all45_logical_models_frozen=True,selection_FIT_only=True,
        validation_only_epoch_and_threshold=True,no_new_Outer=True)
    write(RUN/'MODEL_FREEZE_PRIVATE.json',fr)
    write(PUB/'MODEL_FREEZE_AUDIT.json',dict(status='ALL45_LOGICAL_INTEGRATED_MODELS_FROZEN_BEFORE_OUTER',logical_cells=45,
        actual_M8_cells=len(completed),actual_D1_cells=len(bases),aliases=45-len(completed),freeze_sha256=sha(RUN/'MODEL_FREEZE_PRIVATE.json'),
        validation_only_epoch_threshold=True,all_feature_maps_preprocessing_scales_checkpoints_locked=True,full_goal_complete=False))
    print('ALL45_LOGICAL_INTEGRATED_M8_TRAIN_VALIDATION_COMPLETE_FROZEN_NO_OUTER',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--admission-only',action='store_true');args=parser.parse_args()
    if args.admission_only:prepare()
    else:train()
