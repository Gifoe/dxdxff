"""Exactly fifteen registered runs; channel BCE with one update per FIT patient."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_raw import ROOT
from common import digest, metric, sha, write_json
from data import RawBank
from model import SSSMIL
from prepare import PRIOR
from runtime import seed_all, state_hash, restore_rng, rng_state, select_threshold, torch_write


def initialize(arm, fold, device):
    seed = 42 + 1009 * fold
    seed_all(seed)
    model = SSSMIL(arm, seed)
    if arm != 'S0':
        frozen = torch.load(PRIOR/f'fold{fold}/D2/BEST_PRIVATE.pt', map_location='cpu', weights_only=False)
        model.d2.load_state_dict(frozen['model'])
    return model.to(device)


def predict(model, d, raw, fold, device, diagnostics=False):
    model.eval()
    va = d['val']
    score = np.full(len(d['y']), np.nan)
    disabled = np.full(len(d['y']), np.nan)
    rows, details = [], []
    with torch.no_grad():
        for patient in sorted(set(d['patient'][va])):
            ix = va[d['patient'][va] == patient]
            ix = ix[np.argsort(d['channel'][ix].astype(str), kind='stable')]
            batch = raw.sample(patient, d['channel'][ix], 42+1009*fold, None,
                               model.arm=='S2', device)
            raw_embedding, raw_rms, engineered_rms, weights, variances = [], [], [], [], []
            available = []
            for lo in range(0, len(ix), 8):
                jj=ix[lo:lo+8]
                bb=tuple(v[lo:lo+8] for v in batch)
                xx=torch.as_tensor(d['x'][jj], device=device)
                gg=torch.as_tensor(d['g'][jj], device=device)
                logit, info=model(*bb,xx,gg,diagnostics=True)
                avail=info['available'].cpu().numpy()
                out=torch.sigmoid(logit).cpu().numpy()
                if model.arm=='S0':
                    out[~avail]=np.nan
                score[jj]=out
                available.extend(avail)
                if diagnostics:
                    raw_embedding.append(info['raw'].cpu().numpy())
                    mask=bb[2]
                    weight=info['attention']
                    valid_bags=mask.any(-1)
                    if valid_bags.any():
                        ent=-(weight*weight.clamp(min=1e-12).log()).sum(-1)
                        effective=torch.exp(ent)
                        counts=mask.sum(-1)
                        max_weight=weight.max(-1).values
                        weights.extend(zip(ent[valid_bags].cpu().tolist(),effective[valid_bags].cpu().tolist(),
                                           max_weight[valid_bags].cpu().tolist(),counts[valid_bags].cpu().tolist()))
                    z=info['window_embedding']
                    n=mask.sum(-1).clamp(min=1).unsqueeze(-1)
                    mean=(z*mask.unsqueeze(-1)).sum(-2)/n
                    vv=((z-mean.unsqueeze(-2)).square()*mask.unsqueeze(-1)).sum(-2)/n
                    variances.extend(vv[valid_bags].mean(-1).cpu().tolist())
                    if model.arm!='S0':
                        raw_rms.extend(info['contribution'].square().mean(-1).sqrt().cpu().tolist())
                        engineered_rms.extend(info['engineered'].square().mean(-1).sqrt().cpu().tolist())
                        off=model.d2.head(info['engineered'],gg)
                        disabled[jj]=torch.sigmoid(off).cpu().numpy()
            if diagnostics:
                embedding=np.concatenate(raw_embedding)
                detail={'fold':fold,'patient':patient,'method':model.arm,
                        'raw_embedding_variance':float(np.var(embedding,axis=0).mean()),
                        'window_embedding_variance':float(np.mean(variances)),
                        'valid_seizures_per_channel':float(batch[2].any(-1).sum(-1).float().mean()),
                        'raw_available_fraction':float(np.mean(available))}
                if weights:
                    w=np.array(weights)
                    detail.update(attention_entropy=float(w[:,0].mean()),effective_windows=float(w[:,1].mean()),
                                  max_attention=float(w[:,2].mean()),attention_gt_99_fraction=float((w[:,2]>.99).mean()),
                                  valid_windows_per_seizure=float(w[:,3].mean()))
                if model.arm!='S0':
                    r=np.asarray(raw_rms); e=np.asarray(engineered_rms)
                    detail.update(gamma=float(model.gamma),raw_rms=float(r.mean()),engineered_rms=float(e.mean()),
                                  raw_engineered_ratio=float(np.mean(r/np.maximum(e,1e-12))),
                                  nonzero_raw_fraction=float((r>1e-10).mean()))
                details.append(detail)
            keep=np.isfinite(score[ix])
            if keep.any():
                rows.append((patient,ix[keep]))
    return score, disabled, rows, details


def train_cell(arm, fold, protocol, raw, device):
    seed=42+1009*fold
    d=torch.load(PRIOR/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
    cell=ROOT/f'fold{fold}'/arm
    cell.mkdir(parents=True,exist_ok=True)
    model=initialize(arm,fold,device)
    initial=state_hash(model.state_dict())
    binding={'protocol':sha(protocol),'bank':sha(PRIOR/f'fold{fold}/BANK_PRIVATE.pt'),
             'D2_checkpoint':sha(PRIOR/f'fold{fold}/D2/BEST_PRIVATE.pt'),
             'raw_index':sha(ROOT/'RAW_INDEX_PRIVATE.json'), 'initial':initial,'arm':arm,
             'fold':fold,'seed':seed,'device':device,
             'code':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    tests=json.loads((ROOT/'audit/ENGINEERING_TESTS.json').read_text())
    assert tests['status']=='PASS' and tests['protocol_sha256']==binding['protocol']
    assert tests['code_sha256']==binding['code']
    if (cell/'COMPLETE.json').exists():
        complete=json.loads((cell/'COMPLETE.json').read_text())
        assert complete['binding']==binding
        assert complete['best_sha256']==sha(cell/'BEST_PRIVATE.pt')
        assert complete['validation_sha256']==sha(cell/'VALIDATION_PRIVATE.pt')
        print('REUSE_COMPLETED',fold,arm,flush=True)
        return
    if arm=='S0':
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    else:
        old=list(model.d2.parameters())
        oldids=set(map(id,old))
        new=[p for p in model.parameters() if id(p) not in oldids]
        opt=torch.optim.AdamW([{'params':old,'lr':1e-4},{'params':new,'lr':3e-4}],weight_decay=1e-4)
    seed_all(seed)
    start=1; hist=[]; best=None; key=None; stale=0
    last=cell/'LAST_PRIVATE.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        assert saved['binding']==binding
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng']); start=saved['epoch']+1
        hist=saved['history']; best=saved['best']; key=saved['key']; stale=saved['stale']
        if saved['epoch']>=6 and stale>=6:
            start=31
        print('RESUME_EPOCH',fold,arm,start,flush=True)
    fit=d['train']
    patient_groups=[fit[d['patient'][fit]==p] for p in sorted(set(d['patient'][fit]))]
    counts=torch.tensor([len(set(d['patient'][fit][d['g'][fit]==k])) for k in range(4)],dtype=torch.float32,device=device)
    pos=float((d['y'][fit]==0).sum())/float((d['y'][fit]==1).sum())
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device=device),reduction='sum')
    for epoch in range(start,31):
        tick=time.perf_counter()
        model.train(); losses=[]; grads=[]; update_count=0
        for position in np.random.default_rng(seed+epoch).permutation(len(patient_groups)):
            ix=patient_groups[int(position)]
            p=d['patient'][ix[0]]
            inp=raw.sample(p,d['channel'][ix],seed,epoch,arm=='S2',device)
            available=inp[2].any(-1).any(-1)
            chosen=torch.arange(len(ix),device=device)
            if arm=='S0':
                chosen=chosen[available]
            assert len(chosen)>0, 'Patient without valid raw was blocked in phase A'
            opt.zero_grad(set_to_none=True)
            patient_loss=0.
            for lo in range(0,len(chosen),8):
                local=chosen[lo:lo+8]
                source_ix=ix[local.cpu().numpy()]
                rawinputs=tuple(v[local] for v in inp)
                x=torch.as_tensor(d['x'][source_ix],device=device)
                g=torch.as_tensor(d['g'][source_ix],device=device)
                y=torch.as_tensor(d['y'][source_ix],dtype=torch.float32,device=device)
                value=model(*rawinputs,x,g)
                loss=lossfn(value,y)/len(chosen)
                assert torch.isfinite(value).all() and torch.isfinite(loss)
                loss.backward()
                patient_loss+=float(loss.detach())
            if arm!='S0':
                reg=.001*model.d2.regularizer(counts)
                reg.backward(); patient_loss+=float(reg.detach())
                grad=sum(float(q.grad.abs().sum()) for q in model.raw.encoder.parameters() if q.grad is not None)
                grads.append(grad)
                if abs(float(model.gamma.detach()))>1e-7:
                    assert grad>0, 'raw encoder blocked despite gamma movement'
            assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); update_count+=1; losses.append(patient_loss)
            if update_count%10==0:
                print('PATIENT_UPDATES',fold,arm,epoch,update_count,flush=True)
        assert update_count==len(patient_groups)
        scores,_,records,_=predict(model,d,raw,fold,device)
        va=np.concatenate([ix for p,ix in records])
        selected=select_threshold(d['y'][va],scores[va],d['patient'][va])
        candidate=(selected['macro_f1'],selected['ez_f1'],-epoch)
        elapsed=time.perf_counter()-tick
        hist.append({'fold':fold,'method':arm,'epoch':epoch,'loss':float(np.mean(losses)),
                     'patient_updates':update_count,'pos_weight':pos,'seconds':elapsed,
                     'gamma':float(model.gamma.detach()) if arm!='S0' else 0.,
                     'raw_gradient_sum_mean':float(np.mean(grads)) if grads else None,**selected})
        if key is None or candidate>key:
            key=candidate; stale=0
            best={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,**selected}
        else:
            stale+=1
        torch_write(last,{'binding':binding,'model':model.state_dict(),'optimizer':opt.state_dict(),
                         'rng':rng_state(),'epoch':epoch,'history':hist,'best':best,'key':key,'stale':stale})
        write_json(ROOT/'RUN_STATUS.json',{'status':'TRAINING','fold':fold,'arm':arm,'epoch':epoch,
                     'selected_epoch':best['epoch'],'protocol_sha256':binding['protocol'],
                     'outer_test_accessed':False,'elapsed_last_epoch_sec':elapsed})
        print('EPOCH_COMPLETE',fold,arm,epoch,'seconds',round(elapsed,2),'selected',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:
            break
    assert best is not None
    model.load_state_dict(best['model'])
    torch_write(cell/'BEST_PRIVATE.pt',{'binding':binding,**best})
    score,disabled,records,details=predict(model,d,raw,fold,device,True)
    rows=[]
    for p,ix in records:
        rows.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':arm,
                     **metric(d['y'][ix],score[ix],best['threshold'])})
        if arm=='S1':
            rows.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':'S1_RAW_DISABLED',
                         **metric(d['y'][ix],disabled[ix],best['threshold'])})
    artifact={'binding':binding,'threshold':best['threshold'],'selected_epoch':best['epoch'],
              'scores':score,'disabled':disabled,'rows':rows,'details':details,'history':hist}
    torch_write(cell/'VALIDATION_PRIVATE.pt',artifact)
    write_json(cell/'COMPLETE.json',{'status':'PASS','binding':binding,'best_sha256':sha(cell/'BEST_PRIVATE.pt'),
                                   'validation_sha256':sha(cell/'VALIDATION_PRIVATE.pt'),
                                   'completed_epochs':len(hist),'selected_epoch':best['epoch'],'threshold':best['threshold']})
    print('CELL_COMPLETE',fold,arm,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--arm',choices=['S0','S1','S2'])
    p.add_argument('--fold',type=int)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    print('RAW_BANK_LOADING',flush=True)
    raw=RawBank()
    print('RAW_BANK_READY',flush=True)
    for arm in ([a.arm] if a.arm else ['S0','S1','S2']):
        for fold in ([a.fold] if a.fold else range(1,6)):
            train_cell(arm,fold,a.protocol,raw,a.device)
    print('REGISTERED_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    main()
