"""Actual label-blind own-FIT waveform normalization audit; no clinical outcomes."""
import gc
from common import *
from model import normalize,pre_scale,local_receptive_field,ARMS


def main():
    binding=setup();banks=legal_fit_banks(labels=False);store=RawStore()
    pids=sorted(set.union(*[set(d['patient']) for d in banks]));assert len(pids)==79
    lookup={}
    for d in banks:
        for pid in set(d['patient']):
            q=d['patient']==pid
            lookup.setdefault(str(pid),(d['channel'][q].copy(),str(d['center'][q][0])))
    reports=[];unit_errors=[];protection=jr(ROOT/'INPUT_FREEZE_PRIVATE.json')
    for number,pid in enumerate(pids,1):
        channels,center=lookup[str(pid)];path=ROOT/'input_audit_private'/(private_key(pid)+'.pkl')
        bind=dict(input=binding,raw_bank_sha256=store.plans[str(pid)]['sha256'])
        if path.exists():
            q=read(path);assert q['binding']==bind
        else:
            raw,mask,times=store.get(pid,channels);flat=raw[mask];sd=flat.double().std(-1,unbiased=False);old_sd=flat.std(-1,unbiased=False)
            present=mask.any(-1);pre=mask&(times[:,None]<0);post=mask&((times[:,None]>=0)&(times[:,None]<20))
            assert (pre.sum(-1)[present]>0).all() and (post.sum(-1)[present]>0).all()
            dense=torch.zeros(mask.shape,device='cuda',dtype=torch.float64);dense[mask]=sd.square()
            prevar=(dense*pre).sum(-1)/pre.sum(-1).clamp_min(1)
            postvar=(dense*post).sum(-1)/post.sum(-1).clamp_min(1)
            valid=present&(prevar>0);ratio=(postvar[valid]/prevar[valid]).sqrt().cpu().numpy()
            window=normalize(raw,mask,times,'window');assert torch.isfinite(window).all()
            amplified=raw.clone();amplified[post]*=2
            old_amplified=normalize(amplified,mask,times,'window');qualified=old_sd>8e-6
            old_amplitude_error=float((window[qualified]-old_amplified[qualified]).abs().max().cpu()) if qualified.any() else np.nan
            if qualified.any():assert old_amplitude_error<=1e-6
            pre_failure=None;unit_error=np.nan;amplitude_change=np.nan
            try:
                pn=normalize(raw,mask,times,'pre');scale,info=pre_scale(raw,mask,times)
                unit_error=0.
                for gain in [2.**-3,2.**3]:
                    error=float((pn-normalize(raw*gain,mask,times,'pre')).abs().max().cpu())
                    unit_error=max(unit_error,error);assert error<=1e-6,'Pre normalization unit-gain gate failed'
                qpn=normalize(amplified,mask,times,'pre');amplitude_change=float((pn-qpn).abs().mean().cpu())
                assert torch.isfinite(pn).all() and torch.isfinite(qpn).all()
                fallback=int(info['constant_pre_channels'].sum().cpu())
                clipped=float((pn.abs()>=8).float().mean().cpu())
            except AssertionError as error:
                pre_failure=str(error);fallback=int((present&(prevar==0)).sum().cpu());clipped=np.nan
            q=dict(binding=bind,patient=str(pid),center=center,channels=len(channels),seizures=int(raw.shape[0]),
                valid_windows=int(mask.sum().cpu()),valid_samples=int(flat.numel()),finite_raw=True,
                old_floor_window_fraction=float((old_sd<=1e-6).double().mean().cpu()),
                old_power8_safe_fraction=float(qualified.double().mean().cpu()),
                old_post_gain_invariance_qualified_max_error=old_amplitude_error,
                pre_scale_zero_channel_records=int((present&(prevar==0)).sum().cpu()),
                pre_normalization_eligible=pre_failure is None,pre_failure=pre_failure,
                pre_fallback_channel_records=fallback,pre_unit_max_error=unit_error,
                pre_amplitude_response_mean_absolute_change=amplitude_change,pre_clipped_sample_fraction=clipped,
                post_pre_waveform_RMS_ratio_quantiles=np.quantile(ratio,[0,.25,.5,.75,1]) if len(ratio) else np.full(5,np.nan),
                patient_no_labels_used=True)
            save(path,q);del raw,mask,times,flat,window,dense,amplified,old_amplified;gc.collect()
        reports.append(q)
        print('RAW_INFORMATION_INPUT_AUDIT',number,'of79',flush=True)
    frame=pd.DataFrame([{k:v for k,v in q.items() if k not in ['binding','post_pre_waveform_RMS_ratio_quantiles','pre_failure']} for q in reports])
    save(ROOT/'INPUT_AUDIT_ROWS_PRIVATE.pkl',reports)
    fields=['old_floor_window_fraction','old_power8_safe_fraction','pre_clipped_sample_fraction',
            'pre_amplitude_response_mean_absolute_change']
    public=[]
    for center in ['ALL']+sorted(frame.center.unique()):
        g=frame if center=='ALL' else frame[frame.center==center];n=len(g)
        row=dict(center=center,unique_patients=n,suppressed=n<5,channels=int(g.channels.sum()),
                 seizures=int(g.seizures.sum()),valid_windows=int(g.valid_windows.sum()),
                 failed_pre_normalization_patients=int((~g.pre_normalization_eligible).sum()))
        for name in fields:
            for quantile,label in [(0,'minimum'),(.25,'q1'),(.5,'median'),(.75,'q3'),(1,'maximum')]:
                row[name+'_'+label]=float(g[name].quantile(quantile)) if n>=5 and g[name].notna().any() else np.nan
        public.append(row)
    pd.DataFrame(public).to_csv(PUB/'NORMALIZATION_INPUT_AUDIT.csv',index=False)
    eligible=bool(frame.pre_normalization_eligible.all())
    allowed=ARMS if eligible else [a for a in ARMS if a not in ['C1_PRE_SHORT','C3_PRE_LONG','C4_PRE_LONG_CONTEXT']]
    maximum=float(frame.pre_unit_max_error.max()) if frame.pre_unit_max_error.notna().any() else None
    write(PUB/'INPUT_ADMISSION.json',dict(status='PASS' if eligible else 'PARTIAL_PRE_FAMILY_NOT_IDENTIFIABLE',
        legal_FIT_unique_patients=79,original_FIT_patient_appearances=sum(len(set(d['patient'])) for d in banks),
        raw_valid_windows_audited=int(frame.valid_windows.sum()),pre_family_eligible=eligible,eligible_arms=allowed,
        pre_unit_max_error=maximum,labels_used=False,old_Outer_source_or_prediction_read=False,
        short_local_RF_samples=local_receptive_field(False),long_local_RF_samples=local_receptive_field(True),
        all_original_channels_preserved=True,normalization_coefficients_never_classifier_inputs=True,
        input_binding_sha256=digest(binding)))
    print('RAW_INFORMATION_INPUT_AUDIT_COMPLETE',eligible,flush=True)


if __name__=='__main__':main()
