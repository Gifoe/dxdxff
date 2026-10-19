"""Read-only native-exit diagnosis; no formal update, no test/assessment scores.

Outside the original13-file science binding. Does not change frozen source or
overwrite checkpoints. Exact current/selected reference replay only.
"""
import json
import time
from common import *

def main():
    binding,core,fac,records,rec,banks=setup();inputs=store()
    path=ROOT/'models_private/internal/B2/seed43/fold3/LAST_PRIVATE.pt';before=sha(path)
    q=torch.load(path,weights_only=False,map_location='cpu');assert q['binding']['input']==binding
    assert q['epoch']==6 and len(q['history'])==6
    d,scale=data(3,'internal',core,rec,banks,inputs,binding)
    m=make(core,fac,records,3,'B2',43);assert state_hash(m.state_dict())==q['binding']['initial']
    m.load_state_dict(q['model']);m.train();g=MODEL.GlobalReference(m,d,inputs,scale);replayed=[];seconds=[]
    for k in range(5):
        tick=time.perf_counter();r=g.fit(m);torch.cuda.synchronize();seconds.append(time.perf_counter()-tick);replayed.append(reference_hash(r))
        assert len(set(replayed))==1;print('CURRENT_REFERENCE_REPLAY_PASS',k+1,flush=True)
    m.load_state_dict(q['best']['model']);r=g.fit(m);torch.cuda.synchronize()
    assert reference_hash(r)==reference_hash(q['best']['reference'])
    assert sha(path)==before and set(g.fit_ids)==d['roles']['train']
    result=dict(status='PASS',diagnosis='Native torch_cuda.dll access violation; no Python exception. Single bounded exact reference diagnosis, not automatic retry.',
        source=sources(),checkpoint_sha256=before,completed_epoch=q['epoch'],selected_epoch=q['best']['epoch'],current_reference_exact_replays=5,
        selected_reference_bit_exact=True,current_reference_seconds=seconds,checkpoint_unchanged=True,formal_updates=0,new_assessment_or_Outer=False,
        TRAIN_reference_patients=len(g.fit_ids),TRAIN_reference_valid_windows=g.valid_windows,torch=torch.__version__,cuda=torch.version.cuda,
        scientific_configuration_changed=False)
    write(PUB/'NATIVE_DIAGNOSIS_01.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
