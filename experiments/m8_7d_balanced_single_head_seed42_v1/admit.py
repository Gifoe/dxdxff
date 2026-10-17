"""Replay five actual original controls; unchanged Raw and seven-coordinate path."""
import gc
from common import *
import pytest


def control(fr,fold):
    rows=[r for r in fr['selections'] if r['model']=='S3' and r['fold']==fold]
    assert len(rows)==1;r=rows[0]
    assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
    return r,torch.load(r['checkpoint'],map_location='cpu',weights_only=False),legacy.read(r['validation'])


def replay_control(fr,fold,core,fac,records,d,inputs,ss):
    r,state,v=control(fr,fold);m=make(core,fac,records,fold,'C0_CONCAT')
    assert state_hash(m.state_dict())==r['initial'] and state['epoch']>=1
    m.load_state_dict(state['model']);out=predict(m,d,d['val'],inputs,ss)
    assert np.array_equal(v['indices'],d['archive_indices'][d['val']])
    error=float(np.max(np.abs(out-v['logit'])));assert error<=1e-6
    grid=np.round(np.arange(0,1.0001,.005),3)
    assert np.array_equal(probability(out)[:,None]<grid,probability(v['logit'])[:,None]<grid)
    sel=core.select_threshold(d['y'][d['val']],probability(out),d['patient'][d['val']])
    for k,value in sel.items():assert abs(float(value)-float(state[k]))<1e-12
    del m;gc.collect();torch.cuda.empty_cache();return r,state,out,error


class Counts:
    def __init__(self):self.collected=0;self.passed=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call' and report.passed:self.passed+=1


def main():
    binding,core,fac,records,rec,banks,fr=setup();count=Counts()
    assert pytest.main([str(HERE/'test_model.py'),'-q'],plugins=[count])==0 and count.collected==count.passed==4
    inputs=store();checks=[];largest=None;counts={}
    for fold in range(1,6):
        d=bank(fold,banks);r,s,p,error=replay_control(fr,fold,core,fac,records,d,inputs,scale(fold))
        checks.append(dict(fold=fold,maximum_replay_error=error,all201_CAL_decisions_exact=True,selected_epoch=s['epoch']))
        for pid in sorted(set(d['patient'][d['train']])):
            ii=d['train'][d['patient'][d['train']]==pid];plan=inputs.raw.registry['plans'][str(pid)]
            # Census uses exact registry metadata, never labels/outcomes to choose the largest case.
            if str(pid) not in counts:
                assert sha(plan['path'])==plan['sha256']
                source=torch.load(plan['path'],map_location='cpu',weights_only=False)
                counts[str(pid)]=int(source['mask'][0].sum());del source
            windows=counts[str(pid)]
            if largest is None or windows>largest['windows']:largest=dict(fold=fold,d=d,ii=ii,pid=pid,windows=windows)
    q=largest;fold=q['fold'];d=q['d'];ii=q['ii'];ss=scale(fold)
    raw,mask,times,delta,_=inputs.get(str(q['pid']),d['channel'][ii].astype(str).tolist(),'cuda')
    assert int(mask.sum())==q['windows'];x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    tr=d['train'];pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda');audits=[];init=[]
    original=legacy.make(core,fac,records,fold,'S3');original.eval()
    with torch.inference_mode():expected=original.features(raw,mask,times,delta,ss)[0]
    for arm in ARMS:
        m=make(core,fac,records,fold,arm);m.eval();init.append(state_hash({k:v for k,v in m.state_dict().items() if k!='interaction_gain'}))
        with torch.inference_mode():
            assert torch.equal(m(x,raw,mask,times,delta,ss)['raw_summary'],expected)
            pad=raw.clone();pad[~mask]=123
            assert torch.equal(m(x,raw,mask,times,delta,ss)['logit'],m(x,pad,mask,times,delta,ss)['logit'])
            order=torch.arange(len(x)-1,-1,-1,device='cuda')
            assert torch.allclose(m(x,raw,mask,times,delta,ss)['logit'][order],m(x[order],raw[:,order],mask[:,order],times,delta[:,order],ss)['logit'],atol=1e-6,rtol=1e-5)
        opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001);norms=[]
        for step in range(3):
            opt.zero_grad(set_to_none=True);out=m(x,raw,mask,times,delta,ss);loss=MODEL.objective(out['logit'],y,pos);loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            g={n:float(p.grad.norm()) for n,p in m.named_parameters()}
            assert g['cnn.0.weight']>0 and g['head.0.weight']>0 and g['head.2.weight']>0 and g['gate.network.2.weight']>0
            if step:assert g['gate.network.0.weight']>0
            if arm=='C2_INTERACTION':assert g['interaction_gain']>0
            norms.append(g);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        audits.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),valid_windows=q['windows'],channels=len(ii),
            exact_original_Raw_summary=True,padding_ignored=True,channel_order_equivariant=True,all7_coordinates_retained=True,discarded_legal_FIT_steps=3,gradient_norms=norms))
        del m,opt;gc.collect();torch.cuda.empty_cache()
    assert len(set(init))==1
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),tests_passed=4,actual_control_replays=checks,
        largest_complete_FIT_backwards=audits,shared_CNN_Gate_head_initial_state_exact=True,formal_epochs_discarded=0,new_Outer_passes=0))
    print('BALANCED_SINGLE_HEAD_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
