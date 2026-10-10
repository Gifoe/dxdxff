"""45 inner heads,15 fixed-epoch refits. No outer labels loaded or scored."""
import time
import torch
from common import *
from model import Head
from head_data import plan,bank,state_hash,save_torch,predict

def train(data,patients,source,arm,epochs,path,binding,held=None):
    model=Head(arm);initial=state_hash(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    counts=torch.tensor([sum(source[p]==g for p in patients) for g in range(3)],dtype=torch.float32)
    tensor={p:(torch.from_numpy(data[p]['z']),torch.from_numpy(data[p]['base'].astype(np.float32)),torch.from_numpy(data[p]['y'].astype(np.float32)),torch.full((len(data[p]['z']),),source[p],dtype=torch.long)) for p in patients}
    start=1;history=[];oof={};last=path/'LAST_PRIVATE.pt'
    if last.exists():
        st=torch.load(last,map_location='cpu',weights_only=False);assert st['binding']==binding and st['initial']==initial
        model.load_state_dict(st['model']);optimizer.load_state_dict(st['optimizer']);start=st['epoch']+1;history=st['history'];oof=st['oof']
    for epoch in range(start,epochs+1):
        began=time.perf_counter();losses=[];gradU=gradV=0.
        order=np.random.default_rng(seeded('patient-order',epoch)).permutation(sorted(patients)).tolist()
        for p in order:
            z,base,y,g=tensor[p];v=y>=0;optimizer.zero_grad(set_to_none=True)
            logits=model(z,g,base);weight=torch.where(y[v]>0,2.,1.)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits[v],y[v],reduction='none')*weight).sum()/weight.sum()+model.regularizer(counts)
            assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(t.grad).all() for t in model.parameters())
            if arm!='C1':gradU+=float(model.U.grad.abs().sum());gradV+=float(model.V.grad.abs().sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        if held is not None:oof[epoch]=predict(model,data,held,source)
        history.append({'epoch':epoch,'patient_steps':len(patients),'loss':float(np.mean(losses)),'source_U_gradient_L1':gradU,'source_V_gradient_L1':gradV,'seconds':time.perf_counter()-began})
        save_torch(last,{'binding':binding,'initial':initial,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'history':history,'oof':oof})
    assert len(history)==epochs
    return model,history,oof,initial

def main():
    torch.set_num_threads(2)
    tested=json.loads((ROOT/'public/ENGINEERING_TESTS.json').read_text());assert tested['status']=='PASS' and tested['protocol']==sha(PROTOCOL) and tested['code']==code_hashes()
    mf=manifest();selection=[];history=[];inner_summary=[];shuffle=[];all_frozen=[]
    for fold in range(1,6):
        assignment=plan(mf,fold);save_torch(ROOT/f'fold_{fold}/ASSIGNMENT_PRIVATE.pt',assignment)
        fit=assignment['fit'];data=bank(fold,fit,include_fit_labels=True)
        for role in ['fit','held']:
            ps=assignment[role];counts={s:sum(assignment['real'][p]==i for p in ps) for i,s in enumerate(SOURCES)}
            pseudo_counts={s:sum(assignment['pseudo'][p]==i for p in ps) for i,s in enumerate(SOURCES)}
            shuffle.append({'fold':fold,'partition':role,'patients':len(ps),'changed_fraction':float(np.mean([assignment['real'][p]!=assignment['pseudo'][p] for p in ps])),'counts_real':counts,'counts_randomized':pseudo_counts,'labelblind':True})
        c0tau,_=threshold([(data[p]['y'][data[p]['y']>=0],sigmoid(data[p]['base'][data[p]['y']>=0])) for p in fit])
        c0={'fold':fold,'method':'C0','selected_epoch':0,'threshold':c0tau,'threshold_source':'outer FIT in-sample frozen CNN','new_parameters':0}
        selection.append(c0);all_frozen.append(c0)
        for arm in ['C1','C2','C3']:
            source=assignment['pseudo'] if arm=='C3' else assignment['real'];epoch_predictions={e:{} for e in range(1,51)}
            for inner in range(3):
                train_ids=[p for p in fit if assignment['inner'][p]!=inner];val_ids=[p for p in fit if assignment['inner'][p]==inner]
                assert not set(train_ids)&set(val_ids) and set(train_ids)|set(val_ids)==set(fit)
                binding={'protocol':sha(PROTOCOL),'code':code_hashes(),'fold':fold,'arm':arm,'inner':inner,'bank_gate':sha(ROOT/'BANK_GATE_PRIVATE.json'),'fit_ids':digest(train_ids),'inner_ids':digest(val_ids),'source':digest({p:source[p] for p in fit})}
                model,h,oof,initial=train(data,train_ids,source,arm,50,ROOT/f'fold_{fold}/{arm}/inner_{inner}',binding,val_ids)
                history.extend([{'fold':fold,'method':arm,'stage':'inner','inner':inner,**r} for r in h])
                for epoch in range(1,51):epoch_predictions[epoch].update(oof[epoch])
                print('INNER_HEAD_COMPLETE',fold,arm,inner,flush=True)
            candidates=[]
            for epoch in range(1,51):
                preds=epoch_predictions[epoch];assert set(preds)==set(fit)
                metrics=[]
                for p in fit:
                    v=data[p]['y']>=0;metrics.append(metric(data[p]['y'][v],sigmoid(preds[p][v])))
                r={'fold':fold,'method':arm,'epoch':epoch,'inner_head_OOF_patients':len(fit),'CNN_is_in_sample':True,**{m:float(np.mean([q[m] for q in metrics])) for m in METRICS}}
                candidates.append(r);inner_summary.append(r)
            chosen=max(candidates,key=lambda r:(r['auroc'],r['ap'],-r['epoch']));epoch=chosen['epoch'];preds=epoch_predictions[epoch]
            tau,tie=threshold([(data[p]['y'][data[p]['y']>=0],sigmoid(preds[p][data[p]['y']>=0])) for p in fit])
            binding={'protocol':sha(PROTOCOL),'code':code_hashes(),'fold':fold,'arm':arm,'stage':'full_FIT_refit','epochs':epoch,'fit_ids':digest(fit),'source':digest({p:source[p] for p in fit}),'bank_gate':sha(ROOT/'BANK_GATE_PRIVATE.json')}
            model,h,_,initial=train(data,fit,source,arm,epoch,ROOT/f'fold_{fold}/{arm}/refit',binding)
            history.extend([{'fold':fold,'method':arm,'stage':'refit','inner':-1,**r} for r in h])
            ck=ROOT/f'fold_{fold}/{arm}/SELECTED_HEAD_PRIVATE.pt';seal=ck.with_suffix('.json')
            frozen={'model':model.state_dict(),'binding':binding,'initial':initial,'threshold':tau,'selected_epoch':epoch}
            if ck.exists():
                old=torch.load(ck,map_location='cpu',weights_only=False);assert old['binding']==binding and old['threshold']==tau and state_hash(old['model'])==state_hash(model.state_dict())
            else:save_torch(ck,frozen)
            bound={'fold':fold,'method':arm,'selected_epoch':epoch,'threshold':tau,'checkpoint_sha256':sha(ck),'protocol':sha(PROTOCOL),'code':code_hashes(),'inner_auroc':chosen['auroc'],'inner_ap':chosen['ap'],'inner_threshold_MacroF1':tie[0],'new_parameters':35 if arm=='C1' else 105,'outer_labels_used':False,'initial_sha256':initial}
            if seal.exists():assert json.loads(seal.read_text())==bound
            else:write_json(seal,bound)
            selection.append(bound);all_frozen.append(bound);print('FULL_FIT_HEAD_FROZEN',fold,arm,epoch,flush=True)
    csv('MODEL_SELECTION.csv',selection);csv('INNER_HEAD_CV_SUMMARY.csv',inner_summary);csv('TRAINING_HISTORY.csv',history)
    write_json(ROOT/'public/SOURCE_ASSIGNMENT_AUDIT.json',{'status':'PASS','partitions':shuffle,'all_source_permutations_labelblind':True,'limitation':'Open-iEEG103/124; marginal-preserving permutation cannot change every source ID'})
    write_json(ROOT/'FROZEN_BEFORE_HEAD_OUTER_PRIVATE.json',{'status':'FROZEN_BEFORE_HEAD_OUTER_PREDICT','protocol':sha(PROTOCOL),'code':code_hashes(),'selections':all_frozen})
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'ALL_HEADS_SELECTED_REFIT_AND_FROZEN','inner_training_runs':45,'refits':15,'new_CNN_training':False,'outer_labels_used_for_selection':False})
    print('ALL45_INNER_AND15_REFITS_FROZEN',flush=True)

if __name__=='__main__':main()
