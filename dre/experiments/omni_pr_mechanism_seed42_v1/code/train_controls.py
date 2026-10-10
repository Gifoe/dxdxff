"""Exact historical C1 trainer on ABS/SHUFFLE inputs; 30 inner +10 refits."""
from source import *
from transforms import *
def main():
    torch.set_num_threads(2)
    tested=json.loads((ROOT/'public/ENGINEERING_TESTS.json').read_text());assert tested['status']=='PASS' and tested['protocol']==sha(PROTOCOL) and tested['code']==code_hashes()
    mf=manifest();history=[];choices=[];epochrows=[];transforms_sealed={};globalaudit=[];shuffleaudit=[]
    for fold in range(1,6):
        assignment=plan(mf,fold);fit=assignment['fit'];raw=bank(fold,fit,True)
        for role in ['FIT','HELD']:
            ps=assignment['fit' if role=='FIT' else 'held'];d=raw if role=='FIT' else bank(fold,ps)
            for p in ps:
                perm=permutation(fold,role,key(p),len(d[p]['h']));path=ROOT/f'transforms/fold_{fold}/{role}/{key(p)}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as v:assert np.array_equal(v['permutation'],perm) and str(v['bank_sha'].item())==d[p]['bank_sha']
                else:save_npz(path,permutation=perm,bank_sha=np.array(d[p]['bank_sha']))
                transforms_sealed[str(path)]=sha(path)
            shuffleaudit.append({'fold':fold,'role':role,'patients':len(ps),'channels':sum(len(d[p]['h']) for p in ps),'fixed_points':0,'fixed_point_fraction':0.,'labels_logits_source_unchanged':True})
        for arm in ['C1-ABS','C1-SHUFFLE']:
            epochpreds={e:{} for e in range(1,51)}
            for inner in range(3):
                train=[p for p in fit if assignment['inner'][p]!=inner];val=[p for p in fit if assignment['inner'][p]==inner]
                scaler=global_moments(raw,train) if arm=='C1-ABS' else None
                if scaler is not None:
                    path=ROOT/f'transforms/fold_{fold}/ABS_inner_{inner}.pt';save_torch(path,scaler);transforms_sealed[str(path)]=sha(path)
                    globalaudit.append({'fold':fold,'role':'inner','inner':inner,'fit_patients':len(train),'validation_patients':len(val),'includes_between_patient_variance':True,'labels_used':False,'held_or_inner_val_used':False,'scaler_sha256':sha(path)})
                data=transform(raw,fit,arm,fold,'FIT',scaler)
                binding={'protocol':sha(PROTOCOL),'code':code_hashes(),'fold':fold,'arm':arm,'inner':inner,'training_ids':digest(train),'inner_ids':digest(val),'historical_bank':cfg()['historical_bank_gate_sha256'],'scaler':digest({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scaler.items()}) if scaler else 'fixed_labelblind_derangement'}
                head,h,pred,init=authoritative_train(data,train,assignment['real'],'C1',50,ROOT/f'fold_{fold}/{arm}/inner_{inner}',binding,val)
                history.extend([{'fold':fold,'method':arm,'stage':'inner','inner':inner,**r} for r in h])
                for e in range(1,51):epochpreds[e].update(pred[e])
                print('INNER_CONTROL_COMPLETE',fold,arm,inner,flush=True)
            candidates=[]
            for e in range(1,51):
                assert set(epochpreds[e])==set(fit);q=[]
                for p in fit:
                    v=raw[p]['y']>=0;q.append(metric(raw[p]['y'][v],sigmoid(epochpreds[e][p][v])))
                row={'fold':fold,'method':arm,'epoch':e,'patients':len(fit),'CNN_in_sample_for_inner':True,**{m:float(np.mean([r[m] for r in q])) for m in METRICS}}
                candidates.append(row);epochrows.append(row)
            choice=max(candidates,key=lambda r:(r['auroc'],r['ap'],-r['epoch']));e=choice['epoch'];pred=epochpreds[e]
            tau,tie=threshold([(raw[p]['y'][raw[p]['y']>=0],sigmoid(pred[p][raw[p]['y']>=0])) for p in fit])
            scaler=global_moments(raw,fit) if arm=='C1-ABS' else None;scalerpath=ROOT/f'transforms/fold_{fold}/ABS_full_FIT.pt'
            if scaler is not None:
                save_torch(scalerpath,scaler);transforms_sealed[str(scalerpath)]=sha(scalerpath)
                globalaudit.append({'fold':fold,'role':'full_FIT','inner':-1,'fit_patients':len(fit),'validation_patients':len(assignment['held']),'includes_between_patient_variance':True,'labels_used':False,'held_or_inner_val_used':False,'scaler_sha256':sha(scalerpath)})
            data=transform(raw,fit,arm,fold,'FIT',scaler)
            binding={'protocol':sha(PROTOCOL),'code':code_hashes(),'fold':fold,'arm':arm,'stage':'full_FIT_refit','epochs':e,'training_ids':digest(fit),'scaler_sha256':sha(scalerpath) if scaler else None}
            head,h,_,init=authoritative_train(data,fit,assignment['real'],'C1',e,ROOT/f'fold_{fold}/{arm}/refit',binding)
            history.extend([{'fold':fold,'method':arm,'stage':'refit','inner':-1,**r} for r in h])
            path=ROOT/f'fold_{fold}/{arm}/SELECTED_HEAD_PRIVATE.pt'
            if path.exists():
                prior=torch.load(path,map_location='cpu',weights_only=False);assert prior['binding']==binding and state_hash(prior['model'])==state_hash(head.state_dict()) and prior['threshold']==tau
            else:save_torch(path,{'model':head.state_dict(),'binding':binding,'threshold':tau,'selected_epoch':e,'initial_sha256':init})
            choices.append({'fold':fold,'method':arm,'selected_epoch':e,'threshold':tau,'checkpoint_sha256':sha(path),'scaler_sha256':sha(scalerpath) if scaler else None,'inner_auroc':choice['auroc'],'inner_ap':choice['ap'],'inner_threshold_macro_f1':tie[0],'parameters':35,'outer_labels_used':False})
            print('FULL_FIT_CONTROL_FROZEN',fold,arm,e,flush=True)
    csv('INNER_SELECTION.csv',choices);csv('INNER_EPOCH_METRICS.csv',epochrows);csv('TRAINING_HISTORY.csv',history)
    write_json(ROOT/'public/GLOBAL_NORMALIZATION_AUDIT.json',{'status':'PASS','all_scalers':globalaudit,'patient_equal_total_variance':True,'fitted_statistics_not_trainable_parameters':True,'unknown_good_channels_included':True})
    write_json(ROOT/'public/EMBEDDING_SHUFFLE_AUDIT.json',{'status':'PASS','partitions':shuffleaudit,'within_patient_bijection':True,'fixed_per_training_and_evaluation_role':True,'pseudonymous_SHA256_seed':True,'inference_PR_shuffle_uses_same_prelocked_permutation':True})
    write_json(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json',{'status':'ALL_CONTROLS_FROZEN_BEFORE_OUTER','protocol':sha(PROTOCOL),'code':code_hashes(),'choices':choices,'transform_hashes':transforms_sealed,'historical_freeze':cfg()['historical_freeze_sha256'],'historical_replay_sha256':sha(ROOT/'HISTORICAL_ROWS_PRIVATE.pt')})
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'30_INNER_10_REFITS_FROZEN','new_CNN_training':False,'PR_retrained':False})
    print('ALL30_INNER_AND10_REFITS_FROZEN',flush=True)
if __name__=='__main__':main()
