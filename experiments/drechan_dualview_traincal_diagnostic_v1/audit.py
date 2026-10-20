"""Frozen selected bodies; only complete TRAIN/CAL support, no neural fitting."""
import os
import re
from audit_support import *
from audit_math import legal_scope,stability,raw_stats,score_stats


def phase_forward(m,d,ix,phase,scale,store):
    legal_scope(d['patient'],ix,d['roles'],phase)
    vectors=np.empty((len(ix),96),np.float32);logit=np.empty(len(ix),np.float32);windows=0
    with torch.no_grad():
        for pid in sorted(set(d['patient'][ix])):
            loc=np.flatnonzero(d['patient'][ix]==pid);ii=ix[loc]
            raw,mask,times,delta,_=store.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
            v,_=rdv.MODEL.DualView.pooled_vector(m,raw,mask,times,delta,scale)
            both=rdv.MODEL.views(v,m.anchor_mean,m.anchor_scale)
            z=both if m.head[0].in_features==206 else both[:,96:]
            x=torch.as_tensor(d['x'][ii],device='cuda');out=m.classify(z,x)
            vectors[loc]=v.cpu().numpy();logit[loc]=out.cpu().numpy();windows+=int(mask.sum())
            del raw,delta,v,z,both,x,out
    assert np.isfinite(vectors).all() and np.isfinite(logit).all()
    return dict(indices=d['archive_indices'][ix],patient=d['patient'][ix],v=vectors,logit=logit,valid_windows=windows)


def main():
    binding,core,fac,records,rec,banks,fr=setup();store=inputs()
    commit=os.environ.get('DIAG_SOURCE_COMMIT','');assert re.fullmatch('[a-f0-9]{40}',commit)
    freeze(PUB/'SOURCE_PUSH_BEFORE_AUDIT.json',dict(commit=commit,source=sources(),source_pushed_before_audit=True))
    cells=[r for r in fr['cells'] if r['arm'] in ARMS];assert {(r['arm'],r['seed'],r['fold']) for r in cells}==expected()
    files={};roles=[]
    for fold in range(1,6):
        d,scale=rdv.data(fold,core,rec,banks,store,jr(rdv.ROOT/'INPUT_FREEZE_PRIVATE.json'))
        legal_scope(d['patient'],d['train'],d['roles'],'TRAIN');legal_scope(d['patient'],d['val'],d['roles'],'CAL')
        roles.append(dict(fold=fold,TRAIN_patients=len(d['roles']['train']),CAL_patients=len(d['roles']['cal']),
                          TRAIN_channels=len(d['train']),CAL_channels=len(d['val']),ASSESS_labels_indexed=False,Outer_scored=False))
        for r in [q for q in cells if q['fold']==fold]:
            arm,seed=r['arm'],r['seed'];path=cell_path(arm,seed,fold);bind=cell_binding(binding,r,d)
            assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['calibration'])==r['calibration_sha256']
            old=read(r['calibration']);assert np.array_equal(old['archive_indices'],d['archive_indices'][d['val']])
            if path.exists():
                q=read(path);assert q['binding']==bind
            else:
                q=dict(binding=bind,threshold=r['threshold'],cal_logit=old['logit'].copy(),cal_y=d['y'][d['val']].copy(),
                       cal_patient=d['patient'][d['val']].copy(),cal_indices=d['archive_indices'][d['val']].copy(),phases={},
                       reused_stored_CAL_only=arm=='B0',new_CAL_forward_bit_exact=arm!='B0')
                if arm in LATENT:
                    m=rdv.make(core,fac,records,fold,arm,seed)
                    state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);m.load_state_dict(state['model']);m.eval()
                    assert not m.early and not m.late and m.raw_pr and m.feature_pr and m.pre_reference and not m.interaction
                    before=rdv.state_hash(m.state_dict());anchor=read(rdv.anchor_path(fold,seed,'pre'))
                    assert torch.equal(m.anchor_mean.cpu(),anchor['mean']) and torch.equal(m.anchor_scale.cpu(),anchor['scale'])
                    q['anchor_mean']=m.anchor_mean.cpu().numpy().copy();q['anchor_scale']=m.anchor_scale.cpu().numpy().copy()
                    for phase,ix in [('TRAIN',d['train']),('CAL',d['val'])]:q['phases'][phase]=phase_forward(m,d,ix,phase,scale,store)
                    assert np.array_equal(q['phases']['CAL']['logit'],old['logit'])
                    assert rdv.state_hash(m.state_dict())==before
                    q['model_state_unchanged']=True;del m,state;gc.collect();torch.cuda.empty_cache()
                save(path,q)
            assert q['threshold']==r['threshold'] and np.array_equal(q['cal_logit'],old['logit'])
            key=f'{arm}_seed{seed}_fold{fold}';files[key]=sha(path)
            write(PUB/'PROGRESS.json',dict(status='COLLECTING_TRAIN_CAL_ONLY',arm=arm,seed=seed,fold=fold,complete_cells=len(files)))
            print('TRAINCAL_PRIVATE_CELL_COMPLETE',arm,seed,fold,len(files),45,flush=True)
    assert len(files)==45
    freeze(ROOT/'AUDIT_CACHE_SEAL_PRIVATE.json',dict(input=digest(binding),files=files,all45_before_aggregate_reporting=True))
    csv_out('ROLE_SCOPE',roles);calrows=[];rawrows=[];scorerows=[];replay=[]
    for r in cells:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];q=read(cell_path(arm,seed,fold));tag=dict(arm=arm,seed=seed,fold=fold)
        assert sha(cell_path(arm,seed,fold))==files[f'{arm}_seed{seed}_fold{fold}']
        probability=rdv.probability(q['cal_logit'])
        calrows.append(dict(**tag,**stability(q['cal_y'],probability,q['cal_patient'],q['threshold'])))
        replay.append(dict(**tag,stored_CAL_hash_verified=True,new_CAL_forward_bit_exact=q['new_CAL_forward_bit_exact'],
                           reused_stored_CAL_only=q['reused_stored_CAL_only'],selected_threshold=q['threshold'],no_model_or_threshold_change=True))
        if arm=='B0':
            scorerows.append(dict(**tag,phase='CAL',**score_stats(q['cal_logit'],probability,q['cal_patient'])))
        else:
            for phase,values in q['phases'].items():
                rawrows.append(dict(**tag,phase=phase,patients=len(set(values['patient'])),valid_windows=values['valid_windows'],
                                    **raw_stats(values['v'],q['anchor_mean'],q['anchor_scale'])))
                scorerows.append(dict(**tag,phase=phase,**score_stats(values['logit'],rdv.probability(values['logit']),values['patient'])))
    assert len(calrows)==len(replay)==45 and len(rawrows)==60 and len(scorerows)==75
    for name,rows in [('CAL_STABILITY',calrows),('RAW_ANCHOR_DRIFT',rawrows),('LABEL_FREE_SCORE_SUMMARY',scorerows),('CAL_SELECTED_REPLAY',replay)]:csv_out(name,rows)
    write(PUB/'RUN_STATUS.json',dict(status='ALL45_DIAGNOSTIC_CELLS_REPORTED',calibration_cells=45,latent_phase_cells=60,score_phase_cells=75,
                                    neural_fits=0,new_deployment_thresholds=0,assessment_metrics=0,new_Outer=False,active_goal_achieved=False))
    print('TRAINCAL_ALL45_60_75_REPORTED_NO_TRAINING_NO_ASSESS',flush=True)


if __name__=='__main__':main()
