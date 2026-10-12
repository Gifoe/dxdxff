"""Unique changed paths: statistic weighting, equal forward, gradient, resume."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from runtime import args,setup


def main():
    a=args();lock,ref,c,core,original,trainer,models,banks,binding=setup(a)
    pre=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());assert pre['binding']==binding
    checks=[];params=[]
    def check(name,condition):assert condition,name;checks.append({'check':name,'status':'PASS'})
    rng=np.random.default_rng(42);x=rng.normal(size=(32,88)).astype('float32');pid=np.repeat(['f1','f2','v1','v2'],8);tr=np.arange(16);va=np.arange(16,32)
    stats,s,w=models.fit_statistics(x,pid,tr)
    altered=x.copy();altered[va]=100*altered[va]+200;stats2,_,_=models.fit_statistics(altered,pid,tr)
    check('FIT-only statistic normalizer label/held-out independent',np.array_equal(stats['mean'],stats2['mean']) and np.array_equal(stats['scale'],stats2['scale']))
    xi=np.r_[x[:8],np.repeat(x[8:16],3,axis=0)];pi=np.r_[np.repeat('f1',8),np.repeat('f2',24)]
    dup,_,_=models.fit_statistics(xi,pi,np.arange(32))
    check('patient-equal statistic fit invariant to within-patient channel replication',np.allclose(stats['mean'],dup['mean'],atol=1e-12,rtol=0) and np.allclose(stats['scale'],dup['scale'],atol=1e-12,rtol=0))
    for fold in range(1,6):
        d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);seed=42+1009*fold;norm=d['stats_normalizer']
        ms={}
        for arm in lock['arms']:
            core.seed_all(seed);ms[arm]=models.constructor(original.SourceA0,arm,norm)(seed).cuda().eval()
            n=sum(p.numel() for p in ms[arm].parameters());check(f'fold{fold} {arm} parameter count',n==lock['parameters'][arm])
            if fold==1:params.append({'arm':arm,'parameters':n,'magnitude_access':arm=='B3','source_conditioning':False})
        for arm in ['B1','B2','B3']:
            for key in ['1.weight','1.bias','4.weight','4.bias']:
                check(f'fold{fold} {arm} original shared weight {key}',torch.equal(ms[arm].network.state_dict()[key],ms['B0'].network.state_dict()[key]))
        for data_name,xx in [('random',x),('actual_VAL',d['x'][d['val'][:16]])]:
            tx=torch.tensor(xx,device='cuda')
            with torch.inference_mode():base=ms['B0'](tx)
            for arm in ['B2','B3']:
                with torch.inference_mode():pred=ms[arm](tx)
                check(f'fold{fold} {arm} initial exact A0 forward {data_name}',torch.equal(base,pred))
        check(f'fold{fold} matched adapter hidden init',torch.equal(ms['B2'].hidden.weight,ms['B3'].hidden.weight) and torch.equal(ms['B2'].hidden.bias,ms['B3'].hidden.bias))
    # Single changed-path gradient audit: output moves first, then upstream projections.
    y=np.tile([0,1],16);g=np.arange(32)%4;tx=torch.tensor(x,device='cuda');ty=torch.tensor(y,dtype=torch.float32,device='cuda')
    for arm in ['B1','B2','B3']:
        core.seed_all(1051);m=models.constructor(original.SourceA0,arm,stats)(1051).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(m(tx),ty);loss.backward()
        check(arm+' initial shared gradients',m.network[1].weight.grad.abs().sum()>0)
        if arm in ['B2','B3']:check(arm+' nonzero initial output projection gradient',m.output.weight.grad.abs().sum()>0)
        opt.step();opt.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(m(tx),ty).backward()
        check(arm+' all trainable components receive finite gradients after activation',all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()))
    directory=Path(tempfile.mkdtemp(prefix='synthetic_pma_resume_',dir=a.runtime))
    for arm in ['B1','B2','B3']:
        old=trainer.SourceA0;trainer.SourceA0=models.constructor(original.SourceA0,arm,stats)
        try:
            trainer.train(x,y,pid,g,tr,va,directory/arm/'full',1051,'synthetic',penalty=0.,max_epochs=6)
            trainer.train(x,y,pid,g,tr,va,directory/arm/'resume',1051,'synthetic',penalty=0.,max_epochs=6,stop_after=2)
            trainer.train(x,y,pid,g,tr,va,directory/arm/'resume',1051,'synthetic',penalty=0.,max_epochs=6)
        finally:trainer.SourceA0=old
        p=torch.load(directory/arm/'full/LAST_PRIVATE.pt',map_location='cpu',weights_only=False);q=torch.load(directory/arm/'resume/LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        check(arm+' exact resumed weights buffers history selection',all(torch.equal(v,q['model'][k]) for k,v in p['model'].items()) and p['history']==q['history'] and p['best_key']==q['best_key'])
        check(arm+' exact resumed optimizer RNG',all(torch.equal(v,q['optimizer']['state'][k][n]) for k,slot in p['optimizer']['state'].items() for n,v in slot.items()) and torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda'])))
    check('B2 cannot access statistics',not hasattr(ms['B2'],'statistics') and not hasattr(ms['B2'],'stat_mean'))
    check('B3 statistics are buffers, not learned normalization', 'stat_mean' in dict(ms['B3'].named_buffers()) and 'stat_mean' not in dict(ms['B3'].named_parameters()))
    pd=__import__('pandas');pd.DataFrame(params).to_csv(a.runtime/'public/PARAMETER_AUDIT.csv',index=False)
    c.write_json(a.runtime/'public/ENGINEERING_TESTS.json',{'status':'PASS','binding':binding,'checks':checks,'checks_passed':len(checks),'outer_accessed':False})
    print('ALL_PMA_ENGINEERING_GATES_PASS',len(checks),flush=True)


if __name__=='__main__':main()
