"""Read-only trained input-isolation and original initializer checks, VAL only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

def main():
    prot,core,fac,records,rec,banks,binding=setup();fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');assert fr['binding']==binding
    assert jr(PUB/'TRAINING_VERIFICATION.json')['status']=='PASS'
    inputs=store();rows=[];initializers=[]
    for fold in range(1,6):
        d=bank(fold,banks);ss=scale(fold);va=d['val'];ii=va[d['patient'][va]==sorted(set(d['patient'][va]))[0]]
        assert set(d['patient'][ii]).isdisjoint(d['roles']['test'])
        raw,mask,times,delta,perms=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda')
        original=OLD.P3.make(core,fac,records,OLD.BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold,'M8_MACRO_AWARE')
        fresh=make(core,fac,records,fold,'S3')
        assert core.state_hash(fresh.cnn.state_dict())==core.state_hash(original.cnn.state_dict())
        assert core.state_hash(fresh.gate.state_dict())==core.state_hash(original.gate.state_dict())
        initializers.append(dict(fold=fold,original_CNN_and_Gate_initial_state_exact=True));del fresh,original
        for arm in ['S0','S2','S4']:
            r=next(q for q in fr['selections'] if q['model']==arm and q['fold']==fold);assert sha(r['checkpoint'])==r['checkpoint_sha256']
            m=make(core,fac,records,fold,arm);m.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model']);m.eval()
            before=core.state_hash(m.state_dict());xx=x.clone();dd=delta.clone()
            if arm=='S0':xx+=123;dd+=321
            else:xx[:,8:10]+=123;dd[...,4]+=321
            with torch.inference_mode():
                a=m(x,raw,mask,times,delta,ss)['logit'];b=m(xx,raw,mask,times,dd,ss)['logit']
            error=float((a-b).abs().max());assert error==0 and core.state_hash(m.state_dict())==before
            rows.append(dict(model=arm,fold=fold,maximum_logit_error=error,trained_masked_input_isolation=True));del m
        del raw,mask,times,delta,perms,x;gc.collect();torch.cuda.empty_cache()
    protected_unchanged(binding)
    write(PUB/'TRAINED_INPUT_ISOLATION_AUDIT.json',dict(status='PASS',trained_checks=rows,original_initializers=initializers,
        no_D1_in_any_new_model=True,validation_only=True,no_training_or_Outer_read=True,source_sha256=sha(__file__)))
    print('SINGLE_HEAD_ALL15_TRAINED_MASKS_AND5_INITIALIZERS_PASS',flush=True)

if __name__=='__main__':main()
