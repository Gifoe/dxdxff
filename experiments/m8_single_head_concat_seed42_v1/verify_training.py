"""Reconstruct all25 legal validation predictions and original checkpoint/tau choices."""
from common import *

def main():
    prot,core,fac,records,rec,banks,binding=setup();fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');assert fr['binding']==binding
    inputs=store();maximum=0.;epochs=0
    for r in fr['selections']:
        f=r['fold'];arm=r['model'];d=bank(f,banks);cp=Path(r['checkpoint']);assert sha(cp)==r['checkpoint_sha256']
        best=torch.load(cp,map_location='cpu',weights_only=False);last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        done=jr(cp.parent/'COMPLETE_PRIVATE.json');assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
        assert 6<=len(last['history'])<=30 and [h['epoch'] for h in last['history']]==list(range(1,len(last['history'])+1))
        winner=max(last['history'],key=lambda h:(h['macro_f1'],h['ez_f1'],-h['epoch']))
        assert winner['epoch']==best['epoch']==r['epoch'] and winner['threshold']==best['threshold']==r['threshold']
        m=make(core,fac,records,f,arm);assert core.state_hash(m.state_dict())==r['initial']
        assert not hasattr(m,'base') and all(not k.startswith('base.') for k in best['model'])
        m.load_state_dict(best['model']);logit=predict(m,d,d['val'],inputs,scale(f));old=read(r['validation'])
        assert old['checkpoint_sha256']==sha(cp) and np.array_equal(old['indices'],d['archive_indices'][d['val']])
        error=float(np.max(np.abs(logit-old['logit'])));maximum=max(maximum,error);assert error<=1e-6
        score=torch.sigmoid(torch.from_numpy(old['logit'])).numpy();sel=core.select_threshold(d['y'][d['val']],score,d['patient'][d['val']])
        assert sel['threshold']==r['threshold'];epochs+=len(last['history'])
        print('SINGLE_HEAD_VALIDATION_REPLAY_PASS',arm,f,flush=True)
        del m,d,best,last;gc.collect();torch.cuda.empty_cache()
    assert len(fr['selections'])==25;protected_unchanged(binding)
    write(PUB/'TRAINING_VERIFICATION.json',dict(status='PASS',models=25,epochs=epochs,maximum_logit_error=maximum,
        no_D1_in_any_new_model=True,all_roles_epochs_thresholds_initializers_checked=True,source=binding['source'],model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl')))

if __name__=='__main__':main()
