"""Label-blind temporal geometry with genuine-window masks and canonical names."""
import numpy as np

FEATURES=['log_bp_delta','log_bp_theta','log_bp_beta','log_bp_low_gamma',
          'log_bp_high_gamma','rms','variance','line_length_per_sec','spectral_entropy']


def relative_features(x,mask):
    # x[time,channel,9]; invalid channel/window never contributes to reference.
    masked=np.where(mask[...,None],x,np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(masked,axis=1,keepdims=True)
        scale=1.4826*np.nanmedian(np.abs(masked-median),axis=1,keepdims=True)
    scale=np.where(np.isfinite(scale)&(scale>1e-8),scale,1.)
    r=np.clip((masked-median)/(scale+1e-8),-8,8)
    return np.where(mask[...,None],np.concatenate([x,r],-1),0.)


def extract(record,canonical):
    s=record['sample']; names=list(map(str,record['channel_names_norm']))
    assert len(names)==len(set(names)); assert len(canonical)==len(set(canonical))
    fn=list(s['window_feature_names']); assert len(fn)==len(set(fn))
    indices=[fn.index(n) for n in FEATURES]
    x=np.asarray(s['window_features'],dtype=np.float64)[...,indices]
    t=np.asarray(s['window_relative_centers_sec'],dtype=np.float64)
    assert x.shape==(len(t),len(names),9)
    # Reject duplicates; sort by explicit time, never by a fabricated grid.
    finite=np.isfinite(t); assert len(np.unique(t[finite]))==finite.sum()
    order=np.argsort(np.where(finite,t,np.inf),kind='stable'); t=t[order]; x=x[order]
    fs=float(s['raw_temporal_sfreq']); duration=float(s['raw_temporal_duration_sec'])
    onset=float(s['seizure_onset_sec']); start=float(s['start_sec'])
    valid_start=float(s['raw_valid_start_sample'])/fs
    valid_end=valid_start+float(s['raw_valid_samples'])/fs
    # Cache centers refer to annotation onset; derive nominal padded origin from
    # genuine acquisition start minus explicitly recorded left-padding extent.
    onset_position=onset-start+valid_start
    assert fs>0 and 0<=valid_start<valid_end<=duration+1/fs
    assert abs(onset_position-duration/2)<=max(1/float(record['sfreq']),1/fs)+1e-5
    widths=np.asarray(s['feature_scale_used_secs'],dtype=float)[order]
    assert widths.shape==t.shape and np.isfinite(widths).all() and np.all(widths==2.)
    abscenter=t+onset_position
    genuine=finite[order] & (abscenter-widths/2>=valid_start-1e-6) & (abscenter+widths/2<=valid_end+1e-6)
    selected=np.isfinite(t)&(t>=-10)&(t<20)
    mask=np.isfinite(x).all(-1)&genuine[:,None]
    z=relative_features(x,mask)
    mapping={n:i for i,n in enumerate(names)}
    out=np.zeros((len(canonical),int(selected.sum()),18),np.float32)
    om=np.zeros(out.shape[:2],bool)
    for i,ch in enumerate(canonical):
        if ch in mapping:
            out[i]=z[selected,mapping[ch]]; om[i]=mask[selected,mapping[ch]]
    times=np.broadcast_to(t[selected],om.shape).copy().astype(np.float32)
    stages=np.stack([((times>=lo)&(times<hi)&om).sum(1) for lo,hi in [(-10,0),(0,8),(8,20)]],-1)
    return out,times,om,{'stage_counts':stages,'missing_channels':sum(ch not in mapping for ch in canonical),
        'nonfinite_feature_windows':int((~np.isfinite(x).all(-1)).sum()),
        'invalid_time_windows':int((~finite).sum()),'invalid_boundary_windows':int((~genuine & finite[order]).sum()),
        'source_windows':len(t),'width_seconds':float(widths[0]),
        'interval_min':float(np.diff(t[np.isfinite(t)]).min()),'interval_max':float(np.diff(t[np.isfinite(t)]).max()),
        'source_monotonic':bool(np.all(np.diff(np.asarray(s['window_relative_centers_sec']))>0)),
        'onset_position_seconds':onset_position,'left_padding_seconds':valid_start}


def pack(records,canonical):
    extracted=[extract(r,canonical) for r in records]
    w=max(v[0].shape[1] for v in extracted); c=len(canonical); s=len(records)
    z=np.zeros((c,s,w,18),np.float32); t=np.zeros((c,s,w),np.float32); m=np.zeros((c,s,w),bool)
    for k,(a,b,d,_) in enumerate(extracted):
        z[:,k,:a.shape[1]]=a; t[:,k,:a.shape[1]]=b; m[:,k,:a.shape[1]]=d
    return {'z':z,'time':t,'mask':m},[v[3] for v in extracted]


def normalize(banks,fit_ids):
    total=np.zeros(18); squares=np.zeros(18); count=0
    for pid in sorted(fit_ids):
        b=banks[pid]; v=b['z'][b['mask']].astype(float)
        total+=v.sum(0); squares+=(v*v).sum(0); count+=len(v)
    assert count>0
    mean=total/count; std=np.sqrt(np.maximum(squares/count-mean**2,0.)); std=np.where(std>1e-8,std,1.)
    result={}
    for pid,b in banks.items():
        result[pid]={**b,'z':np.where(b['mask'][...,None],(b['z']-mean)/std,0.).astype(np.float32)}
        assert np.isfinite(result[pid]['z']).all()
    return result,{'mean':mean,'std':std,'tokens':count,'fit_patient_ids':sorted(fit_ids)}
