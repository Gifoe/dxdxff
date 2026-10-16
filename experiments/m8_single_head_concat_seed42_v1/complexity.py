"""Actual affine MAC hooks and matched label-free complete-FIT-patient timing."""
from common import *
import time

class MACMeter:
    def __init__(self,model):
        self.count=dict(cnn=0,head=0,gate=0,other=0,valid_windows=0);self.handles=[]
        for name,module in model.named_modules():
            if isinstance(module,(torch.nn.Conv1d,torch.nn.Linear)):
                def hook(m,x,y,name=name):
                    group='cnn' if '.cnn.' in '.'+name+'.' else 'gate' if '.gate.' in '.'+name+'.' else 'head' if '.head.' in '.'+name+'.' else 'other'
                    n=y.numel()*(m.in_channels//m.groups*m.kernel_size[0] if isinstance(m,torch.nn.Conv1d) else m.in_features)
                    self.count[group]+=int(n)
                    if isinstance(m,torch.nn.Conv1d) and m.in_channels==1:self.count['valid_windows']+=int(x[0].shape[0])
                self.handles.append(module.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()
        self.count['total']=sum(self.count[k] for k in ['cnn','head','gate','other']);return dict(self.count)

class LateComponents(torch.nn.Module):
    def __init__(self,d1,raw):super().__init__();self.d1=d1;self.raw=raw
    def forward(self,x,raw,mask,times):return self.d1(x),self.raw(raw,mask,times)

def main():
    prot,core,fac,records,rec,banks,binding=setup();fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl')
    assert fr['binding']==binding and jr(PUB/'TRAINING_VERIFICATION.json')['status']=='PASS'
    d=bank(1,banks);inputs=store();ss=scale(1);tr=d['train'];ii=tr[d['patient'][tr]==sorted(set(d['patient'][tr]))[0]]
    raw,mask,times,delta,perms=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda');x=torch.as_tensor(d['x'][ii],device='cuda')
    models={};calls={}
    for arm in prot['models']:
        r=next(q for q in fr['selections'] if q['model']==arm and q['fold']==1);m=make(core,fac,records,1,arm)
        m.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model']);m.eval();models[arm]=m
        calls[arm]=lambda m=m:m(x,raw,mask,times,delta,ss)
    old=read(OLD.RUN/'MODEL_SELECTION_PRIVATE.pkl')['selections']
    for name,variant in [('REF_M8','FULL_RETRAIN'),('REF_M8_NO_VAR','DROP_4')]:
        r=next(q for q in old if q['variant']==variant and q['fold']==1)
        m,unused=OLD.build(core,fac,records,Path(r['D1_checkpoint']),1,r['keep']);del unused
        m.load_state_dict(torch.load(r['M8_checkpoint'],map_location='cuda',weights_only=False)['model']);m.eval();models[name]=m
        calls[name]=lambda m=m:m(x,raw,mask,times,delta,ss)
    m=OLD.P3.make_base(core,fac,records,OLD.BASELINE/'D1/fold1/TRAIN/BEST_PRIVATE.pt',1051).cuda().eval();models['REF_D1']=m;calls['REF_D1']=lambda:m(x)
    # Capture default arguments to avoid loop-variable closure changes below.
    calls['REF_D1']=lambda m=m:m(x)
    RawOnly=load('single_head_reference_raw_model',Path(r'D:\m8_raw_only_seed42_v1\model.py')).M8RawOnly
    orig=template(core,fac,records,1);rm=RawOnly(orig).cuda();del orig
    r=next(q for q in jr(RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json')['selections'] if q['fold']==1)
    assert sha(r['checkpoint'])==r['checkpoint_sha256'];rm.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model']);rm.eval()
    models['REF_RAW_ONLY']=rm;calls['REF_RAW_ONLY']=lambda:rm(raw,mask,times)
    lm=LateComponents(models['REF_D1'],rm).eval();models['REF_LATE_FUSION']=lm;calls['REF_LATE_FUSION']=lambda:lm(x,raw,mask,times)
    hashes={k:core.state_hash(v.state_dict()) for k,v in models.items()};counts={};timings={k:[] for k in models}
    with torch.inference_mode():
        for name,m in models.items():
            meter=MACMeter(m);calls[name]();counts[name]=meter.close()
        for _ in range(3):
            for name in models:calls[name]()
        torch.cuda.synchronize()
        order=list(models)
        for rep in range(10):
            for name in (order if rep%2==0 else list(reversed(order))):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();q=calls[name]();end.record();end.synchronize();timings[name].append(float(start.elapsed_time(end)));del q
    rows=[]
    for name,m in models.items():
        assert core.state_hash(m.state_dict())==hashes[name]
        rows.append(dict(model=name,parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
            scope='same_complete_fold1_FIT_patient_neural_forward',channels=len(ii),valid_raw_windows=int(mask.sum()),
            actual_affine_MAC=counts[name]['total'],cnn_MAC=counts[name]['cnn'],gate_MAC=counts[name]['gate'],head_MAC=counts[name]['head'],other_MAC=counts[name]['other'],
            timing_repetitions=10,forward_cuda_ms_mean=float(np.mean(timings[name])),forward_cuda_ms_median=float(np.median(timings[name])),forward_cuda_ms_sample_sd=float(np.std(timings[name],ddof=1)),
            includes_offline_physiology_extraction=False,counts_non_affine_as_FLOPs=False,late_fusion_neural_components_only=name=='REF_LATE_FUSION'))
    pd.DataFrame(rows).to_csv(PUB/'COMPLEXITY.csv',index=False)
    write(PUB/'COMPLEXITY_AUDIT.json',dict(status='PASS',models=10,actual_input_shape=list(raw.shape),complete_FIT_only=True,
        original_raw_source_unchanged=True,hooked_affine_MACs_not_totalFLOPs=True,all_models_state_unchanged=True,
        fixed3warmups_10alternating_repetitions=True,source_sha256=sha(__file__),GPU=torch.cuda.get_device_name()))
    protected_unchanged(binding);print('SINGLE_HEAD_MATCHED_FIT_COMPLEXITY_PASS',flush=True)

if __name__=='__main__':main()
