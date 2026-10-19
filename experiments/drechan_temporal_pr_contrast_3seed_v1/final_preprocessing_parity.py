"""Verify potential old B0 score reuse after runtime migration, without test I/O."""
import ctypes
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),ctypes.c_size_t(0xffff0000))
from common import *

def main():
    binding,core,fac,records,rec,banks=setup();inputs=store();checks=[]
    for fold in range(1,6):
        b=parent.bank(fold,banks,'FULL_F0');x,pre=core.prepare(rec['R14'][b['archive_indices']],b['patient'],b['train'])
        assert np.array_equal(x,b['x']) and all(np.array_equal(np.asarray(v),np.asarray(b['pre'][key])) for key,v in pre.items())
        magnitudes=[]
        for pid in sorted(b['roles']['fit']):
            ii=b['train'][b['patient'][b['train']]==pid];raw,mask,times,delta,_=inputs.get(str(pid),b['channel'][ii].astype(str).tolist(),'cpu')
            magnitudes.append(np.median(np.abs(delta[mask].numpy()),axis=0));del raw,delta
        scale=np.maximum(np.median(np.stack(magnitudes),axis=0),1e-30).astype(np.float32)
        assert np.array_equal(scale,parent.scale(fold).cpu().numpy())
        checks.append(dict(fold=fold,FIT_preprocessing_arrays_bit_exact=True,all_input_features_bit_exact=True,FIT_conditioning_scale_bit_exact=True))
    write(PUB/'FINAL_REUSE_PREPROCESSING_PARITY.json',dict(status='PASS',folds=checks,scientific_source=sources(),no_new_Outer_or_assessment_inference=True,
        no_formal_optimizer_updates=True,purpose='Prevent old B0 scores being rebound to numerically different full-FIT preprocessing after runtime migration'))
    print('ALL5_ORIGINAL_FIT_PREPROCESSORS_AND_SCALES_BIT_EXACT',flush=True)

if __name__=='__main__':main()
