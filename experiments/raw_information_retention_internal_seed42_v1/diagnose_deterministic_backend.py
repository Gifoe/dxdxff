"""One outcome-free cross-backend comparison before formal training."""
from common import *
from model import Model
from engine import direct


def main():
    binding=setup();d=legal_fit_banks(True)[0];rr=role_paths([d]+legal_fit_banks(True)[1:])[(1,1)]['roles'];tr=np.r_[rr['train'],rr['choice']]
    pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    raw,mask,times=RawStore().get(pid,d['channel'][ii]);y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    pos=torch.tensor(float((1-d['y'][tr]).sum())/max(float(d['y'][tr].sum()),1),device='cuda');reports=[]
    for arm in ['C2_WINDOW_LONG','C4_PRE_LONG_CONTEXT']:
        outputs=[];gradients=[]
        for flag in [False,True]:
            torch.backends.cudnn.deterministic=flag
            model=Model(arm,1).cuda();loss,score,z=direct(model,raw,mask,times,y,pos);loss.backward()
            outputs.append(score.detach().cpu());gradients.append({n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None})
            del model,loss,score,z;__import__('gc').collect();torch.cuda.empty_cache()
        error=float((outputs[0]-outputs[1]).abs().max());failed=[];maximum=0.
        for name in gradients[0]:
            a,b=gradients[0][name],gradients[1][name];e=float((a-b).abs().max());maximum=max(maximum,e)
            if not torch.allclose(a,b,atol=1e-6,rtol=1e-5):failed.append(dict(parameter=name,error=e))
        reports.append(dict(arm=arm,forward_max_error=error,gradient_max_error=maximum,failed_parameters=failed,
            original_gate_passed=error<=1e-6 and not failed))
        print('CROSS_BACKEND_DIAGNOSIS',arm,error,maximum,not failed,flush=True)
    write(PUB/'DETERMINISTIC_BACKEND_COMPARISON.json',dict(status='PASS' if all(r['original_gate_passed'] for r in reports) else 'FAIL',
        original_gradient_tolerances_unchanged=True,no_cuDNN_disabling=True,no_model_data_loss_or_parameter_changes=True,
        source_binding_sha256=digest(binding),formal_epochs=0,reports=reports))


if __name__=='__main__':main()
