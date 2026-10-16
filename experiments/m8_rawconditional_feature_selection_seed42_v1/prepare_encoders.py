"""20 genuinely inner-cross-fitted Raw encoders; no old trained weights/test I/O."""
import gc
from conditional_io import *


def main():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    common,data,training,support,primitives=old_modules()
    c,core,fac,original,records,rec,banks,deps=common.legal()
    binding=bindings();freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    restore=core.restore_rng;core.restore_rng=lambda s:restore(data.restore_cpu_rng(s))
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    registry=read(OLDROOT/'POOL_REGISTRY_PRIVATE.pkl');inputs=data.Inputs(registry);entries=[];counts=[]
    for fold,d in enumerate(banks,1):
        for screen in range(4):
            q=roles(fold,screen,d);fit=d['train'];pid=d['patient'];cell=ROOT/'encoders_private'/f'fold{fold}_screen{screen}'
            role_binding=dict(input=binding,fold=fold,screen=screen,role_sha256=binding['original_role_files'][f'{fold}_{screen}'])
            target=cell/'REPRESENTATIONS_PRIVATE.pkl'
            if target.exists():
                stored=read(target);assert stored['binding']==role_binding
                assert sha(cell/'BEST_PRIVATE.pt')==stored['checkpoint_sha256']
            else:
                train_ids=set(pid[q['train']]);cal_ids=set(pid[q['cal']]);use=np.flatnonzero(np.isin(pid,sorted(train_ids|cal_ids)))
                # Held/choice labels are physically absent from this trainer bank.
                bank=dict(patient=pid[use],channel=d['channel'][use],y=d['y'][use].copy(),x=np.zeros((len(use),64),np.float32),
                    train=np.flatnonzero(np.isin(pid[use],sorted(train_ids))),val=np.flatnonzero(np.isin(pid[use],sorted(cal_ids))))
                assert set(bank['patient'])==train_ids|cal_ids
                assert set(bank['patient']).isdisjoint(set(pid[q['choice']])|(set(pid[q['held']]) if screen else set()))
                model=data.make(core,fac,records,fold,'Concat',[])
                scale=torch.ones(32,device='cuda')
                best,history=training.fit_raw(model,bank,inputs,scale,cell,42+1009*fold,role_binding,core,f'CrossfitRaw_s{screen}',fold)
                checkpoint=sha(cell/'BEST_PRIVATE.pt');model.eval()
                for p in model.parameters():p.requires_grad_(False)
                before=core.state_hash(model.state_dict());embedding=np.empty((len(fit),192),np.float32);logit=np.empty(len(fit),np.float32)
                with torch.inference_mode():
                    for patient in sorted(set(pid[fit])):
                        loc=np.flatnonzero(pid[fit]==patient);ix=fit[loc]
                        raw,mask,times,delta,_=inputs.get(str(patient),d['channel'][ix].astype(str).tolist(),'cuda');delta=torch.zeros_like(delta)
                        z20,_=model.features(raw,mask,times,delta,scale,mode='identity')
                        kind=model.__class__;support.promote(model);z30,_=model.features(raw,mask,times,delta,scale,mode='identity');model.__class__=kind
                        embedding[loc]=torch.cat([z20,z30],1).cpu().numpy()
                        logit[loc]=model.head(torch.cat([z20,torch.zeros((len(loc),64),device='cuda')],1)).squeeze(-1).cpu().numpy()
                        del raw,mask,times,delta,z20,z30
                assert core.state_hash(model.state_dict())==before and np.isfinite(embedding).all()
                # Complete calibration replay, not a metric/readout on assessment labels.
                reference=training.predict(model,bank,bank['val'],inputs,scale)
                lookup={int(i):j for j,i in enumerate(fit)};local=np.array([lookup[int(i)] for i in use[bank['val']]])
                error=float(np.max(np.abs(reference-logit[local])));assert error<=1e-6
                chosen=core.select_threshold(bank['y'][bank['val']],torch.sigmoid(torch.from_numpy(reference)).numpy(),bank['patient'][bank['val']])
                assert chosen['threshold']==best['threshold']
                mapper=np.full(len(pid),-1,int);mapper[fit]=np.arange(len(fit))
                stored=dict(binding=role_binding,checkpoint_sha256=checkpoint,raw=embedding,raw_logit_nez=logit,
                    archive_indices=d['archive_indices'][fit],roles={k:mapper[q[k]] for k in ['train','cal','choice']+(['held'] if screen else [])},
                    calibration_replay_max_error=error,encoder_train_patients=len(train_ids),encoder_calibration_patients=len(cal_ids),
                    epochs=len(history),selected_epoch=best['epoch'],original_validation_and_Outer_excluded=True)
                save(target,stored);del model;gc.collect();torch.cuda.empty_cache()
            entries.append(dict(fold=fold,screen=screen,path=str(target),sha256=sha(target)))
            counts.append(dict(fold=fold,screen=screen,encoder_train_patients=stored['encoder_train_patients'],
                encoder_calibration_patients=stored['encoder_calibration_patients'],representation_channels=len(stored['archive_indices']),
                representation_dimensions=192,epochs=stored['epochs'],selected_epoch=stored['selected_epoch'],
                calibration_replay_max_error=stored['calibration_replay_max_error'],assessment_and_choice_labels_excluded_from_encoder=True,
                original_validation_and_Outer_excluded=True))
            pd.DataFrame(counts).to_csv(PUB/'ENCODER_ADMISSION.csv',index=False)
            write(PUB/'ENCODER_STATUS.json',dict(status='PREPARING_ENCODERS',complete_cells=len(entries),total_cells=20))
            print('INNER_CROSSFIT_RAW_ENCODER_COMPLETE',fold,screen,len(entries),'of20',flush=True)
    assert len(entries)==20 and sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==binding['parent_Outer_seal_protection']
    save(ROOT/'ENCODER_REGISTRY_PRIVATE.pkl',dict(binding=binding,entries=entries))
    write(PUB/'ENCODER_STATUS.json',dict(status='ALL20_INNER_CROSSFIT_ENCODERS_COMPLETE',complete_cells=20,
        epochs=sum(r['epochs'] for r in counts),no_original_validation_or_Outer_labels_used=True,
        parent_Outer_seal_unchanged=True,registry_sha256=sha(ROOT/'ENCODER_REGISTRY_PRIVATE.pkl')))


if __name__=='__main__':main()
