"""Freeze ten scratch cells; seal new Outer scores before full label access."""
from support import *
PRE=load('progressive_eval_preprocessing',ROOT/'code/preprocessing.py')

def freeze(a):
    lock,c,core,fac,trainer,records,rec,banks,binding=setup(a)
    mf=jread(a.runtime/'MODEL_FREEZE_PRIVATE.json');assert mf['binding']==binding
    assert mf['feature_sha256']==sha(a.runtime/'FEATURES_PRIVATE.pkl')
    assert references(a,core,banks)==mf['references']
    for r in mf['selections']:
        cp=Path(r['checkpoint']);bp=cp.parent.parent/'BANK_PRIVATE.pt'
        assert sha(cp)==r['checkpoint_sha256'] and sha(bp)==r['bank_sha256']
        ck=torch.load(cp,map_location='cpu',weights_only=False);b=torch.load(bp,map_location='cpu',weights_only=False)
        assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
        done=jread(cp.parent/'COMPLETE_PRIVATE.json');assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
        d=banks[r['fold']-1];assert set(b['pre']['fit_patient_ids'])==d['roles']['fit']
        f=read(a.runtime/'FEATURES_PRIVATE.pkl')['ABS14' if r['arm']=='A0' else 'DELTA14']
        assert np.array_equal(PRE.transform(f[d['archive_indices']],b['pre']),b['x'])
    oldfr=jread(a.old_outer/'FROZEN_BEFORE_OUTER_PRIVATE.json');oldseal=jread(a.old_outer/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    oldstatus=jread(a.old_outer/'public/OUTER_STATUS.json')
    assert oldseal['freeze_sha256']==sha(a.old_outer/'FROZEN_BEFORE_OUTER_PRIVATE.json')==oldstatus['freeze_sha256']
    assert oldstatus['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
    assert oldstatus['seal_sha256']==sha(a.old_outer/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    assert jread(a.old_outer/'public/OUTER_INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    for r in mf['references']:
        s=next(v for v in oldfr['selections'] if v['arm']==r['old_arm'] and v['fold']==r['fold'])
        assert all(s[k]==r[k] for k in ['checkpoint_sha256','bank_sha256','threshold','epoch'])
    obj=dict(binding=binding,selections=mf['selections'],references=mf['references'],feature_sha256=mf['feature_sha256'],old_outer_freeze_sha256=sha(a.old_outer/'FROZEN_BEFORE_OUTER_PRIVATE.json'),old_outer_seal_sha256=sha(a.old_outer/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),old_outer_metric_sha256=sha(a.old_outer/'OUTER_METRICS_PRIVATE.pkl'),full_new_test_labels_unread=True)
    fp=a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if fp.exists():assert jread(fp)==obj
    else:write(fp,obj)
    write(a.runtime/'results/OUTER_FREEZE_AUDIT.json',dict(status='PASS',freeze_sha256=sha(fp),selections=[public_selection(v) for v in obj['selections']+obj['references']],all10_new_cells_frozen_before_inference=True,original_10_reference_cells_reused=True,own_fold_TEST_exclusion=True))
    return obj,c,core,fac,records,rec,banks

def infer(a):
    fr,c,core,fac,records,rec,banks=freeze(a);fp=a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json';fh=sha(fp);pub=a.runtime/'results'
    if (pub/'OUTER_STATUS.json').exists():
        assert jread(pub/'OUTER_STATUS.json')['freeze_sha256']==fh;print('OUTER_COMPLETE_NO_NEW_INFERENCE',flush=True);return
    features=read(a.runtime/'FEATURES_PRIVATE.pkl');meta=rec['meta'];cells={};files={}
    for r in fr['selections']+fr['references']:
        arm,fold=r['arm'],r['fold'];ix=np.flatnonzero(np.isin(meta['patient'],list(roles(meta,fold)['test'])))
        pp=a.runtime/f'outer_scores_private/{arm}_fold{fold}.pkl'
        if pp.exists():q=read(pp)
        else:
            if arm in ['A0','A1']:
                ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
                bp=Path(r['checkpoint']).parent.parent/'BANK_PRIVATE.pt';bd=torch.load(bp,map_location='cpu',weights_only=False)
                x=PRE.transform(features['ABS14' if arm=='A0' else 'DELTA14'][ix],bd['pre'])
                model=CompactDelta(fac(42+1009*fold).network,records).cuda().eval();model.load_state_dict(ck['model'],strict=True)
                with torch.inference_mode():ll=model(torch.as_tensor(x,device='cuda')).cpu().numpy()
                assert core.state_hash(model.state_dict())==core.state_hash(ck['model']);del model;torch.cuda.empty_cache()
            else:
                old=r['old_arm'];op=a.old_outer/f'scores_private/{old}_fold{fold}.pkl'
                seal=jread(a.old_outer/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
                assert sha(op)==seal['files'][f'{old}_{fold}']
                oldq=read(op);assert oldq['checkpoint_sha256']==r['checkpoint_sha256']
                assert oldq['freeze_sha256']==fr['old_outer_freeze_sha256'] and np.array_equal(oldq['indices'],ix)
                ll=oldq['logit'].copy()
            q=dict(logit=ll,indices=ix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(pp,q)
        assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(q['indices'],ix)
        assert np.isfinite(q['logit']).all() and 'y' not in q
        cells[arm,fold]=q;files[f'{arm}_{fold}']=sha(pp);print('OUTER_FROZEN_CELL_SEALED',arm,fold,len(ix),flush=True)
    seal=a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';obj=dict(freeze_sha256=fh,files=files,all10_new_and10_reused_predictions_sealed_before_new_full_y=True)
    if seal.exists():assert jread(seal)==obj
    else:write(seal,obj)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    assert len(y)==7635 and (y==0).sum()==1743 and (y==1).sum()==5892
    rs=[];conf=[]
    for r in fr['selections']+fr['references']:
        arm,fold=r['arm'],r['fold'];q=cells[arm,fold];ix=q['indices'];score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
        d=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],val=np.arange(len(ix)))
        rs.extend(EV.patient_rows(c,d,score,r['threshold'],arm,fold))
    df=pd.DataFrame(rs)
    for arm,g in df.groupby('arm'):assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
    original=pd.DataFrame(read(a.old_outer/'OUTER_METRICS_PRIVATE.pkl'))
    for arm,old in [('A2','D1'),('A3','M8_MACRO_AWARE')]:
        left=df[df.arm==arm].set_index(['fold','patient']).sort_index();right=original[original.arm==old].set_index(['fold','patient']).sort_index()
        assert left.index.equals(right.index)
        assert np.array_equal(left[METRICS+['TP','FP','TN','FN','channels']].to_numpy(),right[METRICS+['TP','FP','TN','FN','channels']].to_numpy())
    save(a.runtime/'OUTER_METRICS_PRIVATE.pkl',rs);summary(df,pub,'OUTER')
    for scope,group in [('overall',['arm']),('fold',['arm','fold']),('center',['arm','center'])]:
        for kk,g in df.groupby(group):
            kk=kk if isinstance(kk,tuple) else (kk,)
            conf.append(dict(scope=scope,**dict(zip(group,kk)),patients=len(g),channels=int(g.channels.sum()),**g[['TP','FP','TN','FN']].sum().to_dict(),**g[METRICS].mean().to_dict()))
    pd.DataFrame(conf).to_csv(pub/'CONFUSION_AND_SENSITIVITY.csv',index=False)
    write(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',freeze_sha256=fh,seal_sha256=sha(seal),patients_per_model=80,channels_per_model=7635,new_cells=10,reused_cells=10,all_predictions_sealed_before_new_full_y=True,A2_A3_patient_metrics_exact_original=True,scientific_status='EXPLORATORY_REPEATED_TEST'))
    print('FOUR_MODEL_OUTER_COMPLETE',flush=True)

if __name__=='__main__':infer(args())
