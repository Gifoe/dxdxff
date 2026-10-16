"""All fixed new heads and reference/diagnostic scores frozen then sealed once."""
from common import *
from complexity import MACMeter
import time

def main():
    prot,core,fac,records,rec,banks,binding=setup();fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');assert fr['binding']==binding
    assert jr(PUB/'TRAINING_VERIFICATION.json')['status']=='PASS'
    decision=jr(PUB/'CANDIDATE_FROZEN.json');assert decision['status']=='SELECTED_BEFORE_NEW_OUTER' and decision['model_freeze_sha256']==sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl')
    obj=dict(binding=binding,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),candidate_sha256=sha(PUB/'CANDIDATE_FROZEN.json'),
        selected_model=decision['selected_model'],all25_new_checkpoints_frozen=True,main_cells=50,diagnostic_cells=30,scientific_status='EXPLORATORY_REPEATED_TEST')
    fp=ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if fp.exists():assert jr(fp)==obj
    else:write(fp,obj)
    fh=sha(fp);outdir=ROOT/'outer_scores_private';outdir.mkdir(exist_ok=True);inputs=store();files={};timing=[]
    for r in fr['selections']:
        arm=r['model'];fold=r['fold'];assert sha(r['checkpoint'])==r['checkpoint_sha256']
        roles=banks[fold-1]['roles'];meta=rec['meta'];ix=np.flatnonzero(np.isin(meta['patient'],list(roles['test'])))
        assert set(meta['patient'][ix])==roles['test'] and roles['test'].isdisjoint(roles['fit']|roles['validation'])
        d=bank(fold,banks);z=core.patient_z(rec['R14'][ix],meta['patient'][ix]);pre=d['pre']
        x=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)
        m=make(core,fac,records,fold,arm);m.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model']);m.eval()
        before=core.state_hash(m.state_dict());ss=scale(fold);modes=[arm]
        if arm!='S0':modes.append(arm+'__PHYS0')
        if arm in ['S3','S4']:modes.append(arm+'__IDENTITY_GATE')
        logits={name:np.empty(len(ix),np.float32) for name in modes};macro=[];reused=0;t=time.perf_counter()
        for pid in sorted(roles['test']):
            jj=np.flatnonzero(meta['patient'][ix]==pid);ii=ix[jj]
            cb=dict(freeze=fh,checkpoint=r['checkpoint_sha256'],input=array_sha(x[jj]),indices=array_sha(ii),scale=array_sha(ss.cpu().numpy()))
            token=hashlib.sha256(str(pid).encode()).hexdigest()[:24];pp=ROOT/'patient_scores_private'/arm/f'fold{fold}'/(token+'.pkl')
            if pp.exists():q=read(pp);assert q['binding']==cb;reused+=1
            else:
                raw,mask,times,delta,perms=inputs.get(str(pid),meta['channel'][ii].astype(str).tolist(),'cuda');xx=torch.as_tensor(x[jj],device='cuda')
                meter=MACMeter(m);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    start.record();out=m(xx,raw,mask,times,delta,ss);end.record();end.synchronize();mac=meter.close()
                    values={arm:out['logit'].cpu().numpy()}
                    if arm!='S0':values[arm+'__PHYS0']=m.classify(out['raw_summary'],xx,zero_phys=True).cpu().numpy()
                    if arm in ['S3','S4']:values[arm+'__IDENTITY_GATE']=m(xx,raw,mask,times,delta,ss,identity_gate=True)['logit'].cpu().numpy()
                assert mac['valid_windows']==int(mask.sum())
                q=dict(binding=cb,indices=ii,values=values,mac=mac,forward_cuda_ms=float(start.elapsed_time(end)),channels=len(ii));save(pp,q)
                del raw,mask,times,delta,perms,xx,out,values
            assert np.array_equal(q['indices'],ii)
            for name in modes:assert np.isfinite(q['values'][name]).all();logits[name][jj]=q['values'][name]
            macro.append(q)
        assert core.state_hash(m.state_dict())==before
        for name in modes:
            result=dict(indices=ix,logit=logits[name],score_nez=torch.sigmoid(torch.from_numpy(logits[name])).numpy(),threshold=r['threshold'],freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256'])
            key=name+'_'+str(fold);pp=outdir/(key+'.pkl')
            if pp.exists():old=read(pp);assert old['freeze_sha256']==fh and np.array_equal(old['logit'],result['logit'])
            else:save(pp,result)
            files[key]=sha(pp)
        timing.append(dict(model=arm,fold=fold,patients=len(roles['test']),channels=len(ix),seconds=time.perf_counter()-t,reused_patient_caches=reused,
            actual_affine_MAC=sum(q['mac']['total'] for q in macro),valid_windows=sum(q['mac']['valid_windows'] for q in macro),
            instrumented_forward_cuda_ms=sum(q['forward_cuda_ms'] for q in macro)))
        print('SINGLE_HEAD_OUTER_FOLD_SEALED',arm,fold,len(files),'newscorecells',flush=True)
        del m,d;gc.collect();torch.cuda.empty_cache()
    for r in binding['references']:
        assert sha(r['path'])==r['sha256'];q=read(r['path']);assert q['freeze_sha256']==r['freeze_sha256']
        fold=r['fold'];roles=banks[fold-1]['roles'];ix=np.flatnonzero(np.isin(rec['meta']['patient'],list(roles['test'])))
        assert np.array_equal(q['indices'],ix)
        result=dict(indices=ix,score_nez=q['score_nez'],threshold=q['threshold'],freeze_sha256=fh,reused_reference_sha256=r['sha256'])
        key=r['model']+'_'+str(fold);pp=outdir/(key+'.pkl')
        if pp.exists():old=read(pp);assert old['freeze_sha256']==fh and np.array_equal(old['score_nez'],result['score_nez'])
        else:save(pp,result)
        files[key]=sha(pp)
    assert len(files)==80
    seal=dict(freeze_sha256=fh,files=files,all80_scores_sealed_before_full_labels=True)
    sp=ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'
    if sp.exists():assert jr(sp)==seal
    else:write(sp,seal)
    save(ROOT/'OUTER_METADATA_PRIVATE.pkl',dict(meta=rec['meta'],archive_sha256=rec['binding']['archive']))
    pd.DataFrame(timing).to_csv(PUB/'OUTER_FORWARD_COST.csv',index=False)
    protected_unchanged(binding);write(PUB/'OUTER_STATUS.json',dict(status='ALL80_MAIN_AND_DIAGNOSTIC_SCORES_SEALED',freeze_sha256=fh,seal_sha256=sha(sp),selected_model=decision['selected_model']))
    print('SINGLE_HEAD_ALL_SCORES_SEALED_NO_OUTER_SELECTION',flush=True)

if __name__=='__main__':main()
