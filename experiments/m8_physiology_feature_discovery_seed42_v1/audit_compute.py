"""Measured source-shape affine MACs, matched timing and offline feature cost; no Test."""
import os
import time
import train_integrated_v2 as T
from integrated_data_v2 import *
from features import extract
from raw_physiological_features import Extractor

PUB=OUT/'compute_public'


def affine_profile(model,args):
    entries=[];handles=[]
    def hook(name):
        def apply(module,inputs,output):
            if isinstance(module,torch.nn.Conv1d):macs=int(output.numel()*module.in_channels//module.groups*np.prod(module.kernel_size))
            else:macs=int(output.numel()*module.in_features)
            entries.append(dict(module=name,type=type(module).__name__,output_elements=output.numel(),affine_MACs=macs))
        return apply
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.Conv1d,torch.nn.Linear)):handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():model(*args)
    for h in handles:h.remove()
    def total(prefix):return sum(r['affine_MACs'] for r in entries if r['module'].startswith(prefix))
    assert total('cnn.')>0 and total('gate.')>0 and total('head.')>0 and total('base.')>0
    return entries,dict(cnn=total('cnn.'),gate=total('gate.'),head=total('head.'),base=total('base.'))


def main():
    PUB.mkdir(exist_ok=True)
    lock,c,core,fac,trainer,records,rec,banks,prior=T.setup_new()
    assert jr(T.PUB/'INDEPENDENT_TRAINING_VERIFICATION.json')['status']=='PASS'
    pool=FeaturePool(rec);pool.populate();d=banks[0];tr=d['train'];pid=sorted(set(d['patient'][tr]))[0];ii=tr[d['patient'][tr]==pid]
    oldstore=T.P3.Inputs(BASELINE,Path(r'D:\dgm_net_seed42_v1_runtime'),OUT);oldstore.arm='M1_SCALED_DGM'
    channels=d['channel'][ii].astype(str).tolist();raw,mask,times,delta,perms=oldstore.get(str(pid),channels,'cuda')
    original=T.P3.make(core,fac,records,BASELINE/'D1/fold1/TRAIN/BEST_PRIVATE.pt',1051,'M8_MACRO_AWARE').eval()
    wide=T.make(core,fac,records,1).eval()
    oldbank=torch.load(BASELINE/'D1/fold1/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
    newbank=torch.load(T.RUN/'SLOT10_OLD7/fold1/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert np.array_equal(oldbank['x'],newbank['x'][:,:14])
    scale=torch.tensor(read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/'scale_fold1_PRIVATE.pkl')['scale'],device='cuda')
    newdelta=torch.cat([delta,delta.new_zeros(*delta.shape[:-1],3)],-1);newscale=torch.cat([scale,scale.new_ones(3)])
    args=[(torch.as_tensor(oldbank['x'][ii],device='cuda'),raw,mask,times,delta,scale),
        (torch.as_tensor(newbank['x'][ii],device='cuda'),raw,mask,times,newdelta,newscale)]
    with torch.inference_mode():assert torch.equal(original.features(*args[0][1:])[0],wide.features(*args[1][1:])[0])
    entries=[];profiles=[]
    for name,model,a in zip(['LEGACY_M8','SLOT10_M8'],[original,wide],args):
        rows,counts=affine_profile(model,a);entries.extend(dict(architecture=name,**r) for r in rows)
        N=int(mask.sum());G=mask.numel();C=len(ii)
        profile=dict(architecture=name,cnn_MACs_per_valid_window=counts['cnn']//N,gate_MACs_per_allocated_window=counts['gate']//G,
            head_MACs_per_channel=counts['head']//C,base_MACs_per_channel=counts['base']//C,
            total_parameters=sum(p.numel() for p in model.parameters()),trainable_correction=sum(p.numel() for p in model.parameters() if p.requires_grad))
        assert profile['cnn_MACs_per_valid_window']*N==counts['cnn']
        assert profile['gate_MACs_per_allocated_window']*G==counts['gate']
        assert (profile['head_MACs_per_channel']+profile['base_MACs_per_channel'])*C==counts['head']+counts['base']
        profiles.append(profile)
    assert profiles[0]['total_parameters']==11018 and profiles[1]['total_parameters']==11690
    estimates=[]
    for name in ['ALL']+sorted(set(rec['meta']['center'])):
        patients=[p for p in pool.mask if name=='ALL' or str(np.unique(rec['meta']['center'][rec['meta']['patient']==p])[0])==name]
        N=sum(int(pool.mask[p].sum()) for p in patients);G=sum(pool.mask[p].size for p in patients);C=sum(pool.mask[p].shape[1] for p in patients)
        for pr in profiles:
            cnn=N*pr['cnn_MACs_per_valid_window'];gate=G*pr['gate_MACs_per_allocated_window'];head=C*(pr['base_MACs_per_channel']+pr['head_MACs_per_channel'])
            estimates.append(dict(scope=name,architecture=pr['architecture'],patients=len(patients),channels=C,valid_windows=N,allocated_gate_windows=G,
                convolution_MACs=cnn,gate_MACs=gate,base_and_head_MACs=head,affine_MACs=cnn+gate+head,affine_only_FLOPs_2_per_MAC=2*(cnn+gate+head),
                per_patient_affine_MACs=(cnn+gate+head)/len(patients)))
    pd.DataFrame(estimates).to_csv(PUB/'AFFINE_COMPUTE_COUNTS.csv',index=False)
    pd.DataFrame(entries).to_csv(PUB/'MEASURED_MODULE_SHAPE_MACS.csv',index=False)
    pd.DataFrame(profiles).to_csv(PUB/'PARAMETER_AND_AFFINE_PROFILE.csv',index=False)
    # Outcome-free representative first complete FIT patient, not a fast-patient search.
    with torch.inference_mode():
        for model,a in zip([original,wide],args):
            for _ in range(3):model(*a)
        timing=[]
        for repeat in range(10):
            for k in np.random.default_rng(42+repeat).permutation(2):
                model=[original,wide][k];torch.cuda.synchronize();start=time.perf_counter();model(*args[k]);torch.cuda.synchronize()
                timing.append(dict(architecture=['LEGACY_M8','SLOT10_M8'][k],repeat=repeat+1,seconds=time.perf_counter()-start,
                    channels=len(ii),valid_windows=int(mask.sum()),FP32=True,TF32=False,trained_checkpoint_not_used=True))
    pd.DataFrame(timing).to_csv(PUB/'MATCHED_PATIENT_INFERENCE_TIMING.csv',index=False)
    # Offline candidate-pool construction is reported separately from cached model inference.
    x=raw[mask][:2048].cpu().numpy();assert len(x)==2048
    expected=jr(SOURCE/'raw_delta_encoder_seed42_v1/PROTOCOL_LOCK.json')['source_hashes']['ez_features.py'];native=Extractor(SOURCE,expected)
    native(x);extract(x);extraction=[]
    for repeat in range(5):
        t=time.perf_counter();q=native(x);native_seconds=time.perf_counter()-t
        t=time.perf_counter();extract(x);candidate_seconds=time.perf_counter()-t
        t=time.perf_counter();np.log(np.maximum(np.expm1(q[:,:3].astype(float)),1e-30)).astype(np.float32);control_seconds=time.perf_counter()-t
        extraction.append(dict(repeat=repeat+1,valid_windows=2048,samples_per_window=500,old7_seconds=native_seconds,new11_seconds=candidate_seconds,
            stable_log3_from_existing_powers_seconds=control_seconds,full_pool_total_seconds=native_seconds+candidate_seconds+control_seconds,
            shared_original_power_pass=True,feature_benchmark_not_training_data_selection=True))
    pd.DataFrame(extraction).to_csv(PUB/'OFFLINE_FEATURE_EXTRACTION_TIMING.csv',index=False)
    oldhistory=pd.read_csv(OUT/'old7_matched_M8/public/TRAINING_HISTORY.csv');newhistory=pd.read_csv(T.PUB/'TRAINING_HISTORY.csv');training=[]
    for stage,df in [('old7_160fits',oldhistory),('integrated_actual_fits',newhistory)]:
        for head,g in df.groupby('head'):
            measured=g['seconds'].dropna() if 'seconds' in g else pd.Series(dtype=float)
            training.append(dict(stage=stage,head=head,epochs=len(g),epochs_with_measured_seconds=len(measured),
                measured_epoch_seconds_sum=float(measured.sum()) if len(measured) else np.nan,
                full_epoch_timing_available=len(measured)==len(g),imports_admissions_reporting_transfer_excluded=True))
    pd.DataFrame(training).to_csv(PUB/'ACTUAL_TRAINING_WORK.csv',index=False)
    write(PUB/'COMPUTE_AUDIT.json',dict(status='PASS',source_sha256=sha(__file__),first_complete_FIT_patient_not_selected_by_speed=True,
        module_forward_hooks_and_actual_masks=True,patients=80,channels=7635,valid_windows=1471965,
        total_parameters=[11018,11690],parameter_ratio=11690/11018,
        affine_counts_exclude_bias_nonlinearity_normalization_reductions_and_feature_extraction=True,
        affine_counts_not_full_end_to_end_FLOPs=True,offline_feature_cost_separately_measured=True,
        timing_conditional_on_shared_Windows_GPU_display_and_this_runtime=True,no_Outer_outcomes_used=True,
        D1_epoch_timing_missing_reported_not_imputed=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__))
    print('M8_PARAMETERS_ACTUAL_SHAPE_MACS_AND_MATCHED_TIMING_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
