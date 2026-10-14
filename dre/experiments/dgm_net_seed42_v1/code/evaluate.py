"""Development-only inference; intervention outputs remain private."""
import numpy as np
import torch

MODES=['full','identity','shuffle','static','residual0']+[f'neutral_{k}' for k in range(7)]

def predict(model,d,indices,store,scale,mode='full',diagnostics=False):
    fields={k:np.empty(len(indices),np.float32) for k in ['logit','raw_logit','base_logit','residual']}
    gate_rows=[]; model.eval()
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][indices])):
            k=np.flatnonzero(d['patient'][indices]==pid); ii=indices[k]
            raw,mask,times,delta,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),'cuda')
            x=torch.as_tensor(d['x'][ii],device='cuda')
            final,aux,base,res,g=model(x,raw,mask,times,delta,scale,mode,perms)
            for key,v in zip(fields,[final,aux,base,res]):fields[key][k]=v.cpu().numpy()
            if mode=='residual0':assert torch.equal(final,base)
            if diagnostics:
                g=g.cpu().numpy(); distance=np.abs(g-1)
                gate_rows.append(dict(patient=str(pid),gate_mean=float(g.mean()),gate_sd=float(g.std()),gate_min=float(g.min()),gate_max=float(g.max()),gate_q05=float(np.quantile(g,.05)),gate_q50=float(np.quantile(g,.5)),gate_q95=float(np.quantile(g,.95)),gate_mean_abs_deviation=float(distance.mean()),gate_near_identity_fraction=float((distance<=1e-6).mean()),gate_near_cap_fraction=float((distance>=.249).mean())))
            del raw,mask,times,delta,perms,x,final,aux,base,res,g
    assert all(np.isfinite(v).all() for v in fields.values())
    fields['gate_rows']=gate_rows; return fields

def errors(d,full,other,tau,fold,mode):
    y=d['y'][d['val']]==0; sigmoid=lambda v:torch.sigmoid(torch.from_numpy(v)).numpy()
    fp=sigmoid(full['logit'])<tau; op=sigmoid(other['logit'])<tau
    pairs=reversals=0
    for pid in sorted(set(d['patient'][d['val']])):
        k=np.flatnonzero(d['patient'][d['val']]==pid)
        a=full['logit'][k];b=other['logit'][k]
        take=np.tril(np.ones((len(k),len(k)),bool),-1)
        reversals+=int((((a[:,None]-a[None])*(b[:,None]-b[None])<0)&take).sum());pairs+=int(take.sum())
    out=dict(fold=fold,intervention=mode,changed_decisions=int((fp!=op).sum()),corrected_by_full=int(((fp==y)&(op!=y)).sum()),spoiled_by_full=int(((fp!=y)&(op==y)).sum()),EZ_corrected_by_full=int((y&fp&~op).sum()),EZ_spoiled_by_full=int((y&~fp&op).sum()),NEZ_corrected_by_full=int((~y&~fp&op).sum()),NEZ_spoiled_by_full=int((~y&fp&~op).sum()),rank_reversals=reversals,rank_pairs=pairs,logit_change_max=float(np.max(np.abs(full['logit']-other['logit']))),mean_absolute_residual=float(np.mean(np.abs(other['logit']-other['base_logit']))),threshold=tau,threshold_source='frozen A2')
    for name,pred in [('full',fp),('intervention',op)]:
        out.update({f'{k}_{name}':int(v) for k,v in dict(TP=(pred&y).sum(),FP=(pred&~y).sum(),TN=(~pred&~y).sum(),FN=(~pred&y).sum()).items()})
    return out
