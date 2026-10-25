"""CPU-only analysis of immutable cached physiological summaries, never waves."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import zipfile
import numpy as np
import pandas as pd
import scipy
from statistics_core import effect,summarize,bh,permuted_effects

HERE=Path(__file__).resolve().parent
REC=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime\RECONSTRUCTION_PRIVATE.pkl')
ARCHIVE=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
REC_SHA='73095eefbb3035200894560de3c3cfdb6259a4f28c343ecd3d7195be5b94f609'
LABEL_SHA='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'
SCIENCE=['statistics_core.py','run_analysis.py','verify_analysis.py','test_statistics.py','PROTOCOL.json']
ALLOW=['feature_association.csv','source_association.csv','patient_effect_summary.csv','statistical_tests.csv',
       'patient_effect_histogram.csv','permutation_control.csv','feature_definitions.json','INPUT_AUDIT.json',
       'INDEPENDENT_VERIFICATION.json','STATUS.json']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.partial');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(path)


def checked_pickle(path):
    path=Path(path);assert sha(path)==json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
    with path.open('rb') as f:return pickle.load(f)


def inputs():
    assert sha(REC)==REC_SHA and sha(ARCHIVE)==LABEL_SHA
    rec=checked_pickle(REC);meta=rec['meta'];x=rec['R14']
    source=Path(r'E:\DRE-nips\new-pipeline\7-11')
    for key,h in rec['binding'].items():
        if key.startswith('source_'):assert sha(source/key[len('source_'):])==h
    for name,h in rec['binding']['code'].items():
        assert sha(source/'raw_delta_encoder_seed42_v1/code'/name)==h
    assert x.shape==(7635,14) and len(set(meta['patient']))==80
    assert len(set(zip(meta['patient'],meta['channel'])))==7635
    with np.load(ARCHIVE,allow_pickle=True) as z:
        for k in ['patient','channel','center']:assert np.array_equal(z[k].astype(str),meta[k])
        label=z['y'].copy()
    assert set(np.unique(label))=={0,1};ez=label==0
    assert rec['binding']['archive']==LABEL_SHA
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    cols=[c['r14_index'] for c in protocol['coordinates']]
    assert cols==[0,1,2,3,4,5,6,7,10,11,12,13]
    assert sum(len(r['records']) for r in rec['ledgers'])==256
    return rec,x[:,cols].astype(float),ez,protocol


def main(runtime):
    import psutil
    psutil.Process().cpu_affinity([0])
    assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    runtime=Path(runtime);private=runtime/'private';pub=runtime/'public'
    private.mkdir(parents=True,exist_ok=True);pub.mkdir(exist_ok=True)
    assert not (pub/'feature_association.csv').exists(),'Completed/partial outputs exist; inspect before replay'
    rec,x,ez,protocol=inputs();meta=rec['meta'];ids=sorted(set(meta['patient']))
    assert 'torch' not in sys.modules,'CPU statistics must not import neural runtime'
    source_hashes={n:sha(HERE/n) for n in SCIENCE}
    binding=dict(source=source_hashes,reconstruction=REC_SHA,label_archive=LABEL_SHA,
                 numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,python=platform.python_version())
    write(private/'INPUT_FREEZE.json',binding)
    # Reuse all80 existing label-blind per-patient summary caches, not raw extraction.
    cache_checks=0
    for n,pid in enumerate(ids,1):
        old=checked_pickle(REC.parent/'patient_reconstruction_private'/f'{n:03d}.pkl')
        assert old['patient']==pid and np.array_equal(rec['R14'][old['indices']],old['R'])
        assert np.array_equal(np.flatnonzero(meta['patient']==pid),old['indices']);cache_checks+=1
    pairs=[];matrix=[];centers=[];null_sum=np.zeros((10000,12));null_n=np.zeros((10000,12),int)
    rng=np.random.default_rng(42)
    for pid in ids:
        ix=np.flatnonzero(meta['patient']==pid);xx=x[ix];yy=ez[ix]
        center=np.unique(meta['center'][ix]);assert len(center)==1 and yy.any() and (~yy).any()
        centers.append(str(center[0]));effects=[]
        for j,c in enumerate(protocol['coordinates']):
            good=np.isfinite(xx[:,j]);d=effect(xx[:,j],yy);effects.append(d)
            pairs.append(dict(patient=pid,source=centers[-1],coordinate=c['coordinate'],effect=d,
                channels=len(ix),finite_EZ_channels=int((good&yy).sum()),finite_NEZ_channels=int((good&~yy).sum()),
                missing_channels=int((~good).sum()),constant_coordinate=bool(good.any() and np.ptp(xx[good,j])==0)))
        matrix.append(effects)
        labels=rng.permuted(np.broadcast_to(yy,(10000,len(yy))),axis=1)
        assert labels.shape==(10000,len(yy)) and np.all(labels.sum(1)==yy.sum())
        null=permuted_effects(xx,yy,labels);valid=np.isfinite(null)
        null_sum+=np.nan_to_num(null);null_n+=valid
    matrix=np.asarray(matrix);centers=np.asarray(centers);assert matrix.shape==(80,12)
    table(private/'patient_effect_summary.csv',pairs)
    np.save(private/'PATIENT_EFFECT_MATRIX.npy',matrix)
    null_means=np.divide(null_sum,null_n,out=np.full_like(null_sum,np.nan),where=null_n>0)
    np.save(private/'PERMUTATION_NULL_MEANS.npy',null_means)
    assert dict(zip(*np.unique(centers,return_counts=True)))==dict(hup=36,lzu=21,multicenter=15,pediatric=8)
    aggregate=[];sources=[];histogram=[];distribution=[]
    for j,c in enumerate(protocol['coordinates']):
        base={k:c[k] for k in ['coordinate','feature','summary','r14_index']}
        aggregate.append(dict(**base,**summarize(matrix[:,j])))
        v=matrix[:,j];v=v[np.isfinite(v)]
        distribution.append(dict(**base,valid_patients=len(v),negative_patients=int((v<0).sum()),zero_patients=int((v==0).sum()),
                                 positive_patients=int((v>0).sum()),**{name:float(np.quantile(v,p)) if len(v) else np.nan
                                 for name,p in [('q025',.025),('q25',.25),('median',.5),('q75',.75),('q975',.975)]}))
        counts,edges=np.histogram(v,bins=np.linspace(-1,1,41))
        for b,count in enumerate(counts):histogram.append(dict(coordinate=c['coordinate'],bin_left=edges[b],bin_right=edges[b+1],patients=int(count)))
        for source in protocol['sources']:
            s=summarize(matrix[centers==source,j]);s.pop('pvalue');s.pop('wilcoxon_statistic')
            sources.append(dict(**base,source=source,cohort_patients=int((centers==source).sum()),**s))
    q=bh([r['pvalue'] for r in aggregate]);permutation=[]
    for j,row in enumerate(aggregate):
        row['BH_q']=float(q[j]);row['family_tests']=12
        row['mean_CI_excludes_zero']=bool(row['ci_low']>0 or row['ci_high']<0)
        row['FDR_below_0_05']=bool(q[j]<.05)
        row['constant_patients']=sum(r['constant_coordinate'] for r in pairs if r['coordinate']==row['coordinate'])
        directed=[s for s in sources if s['coordinate']==row['coordinate']]
        row['sources_same_mean_direction']=sum(np.sign(s['mean_effect'])==np.sign(row['mean_effect']) for s in directed)
        null=null_means[:,j];null=null[np.isfinite(null)];lo,hi=np.quantile(null,[.025,.975])
        sd=float(null.std(ddof=1));mc_se=sd/np.sqrt(len(null));mu=float(null.mean())
        permutation.append(dict(coordinate=row['coordinate'],permutations=len(null),null_mean=mu,null_sd=sd,
            null_mean_MC_standard_error=mc_se,null_ci_low=float(lo),null_ci_high=float(hi),expected_mean=0.,
            null_centered_within_4MCSE=bool(abs(mu)<=4*mc_se+1e-14),
            observed_mean=row['mean_effect'],empirical_two_sided_p=float((1+np.sum(np.abs(null)>=abs(row['mean_effect'])-1e-14))/(len(null)+1)),
            labels_shuffled_within_patient=True,EZ_count_preserved=True))
    table(pub/'feature_association.csv',aggregate);table(pub/'source_association.csv',sources)
    table(pub/'patient_effect_summary.csv',distribution);table(pub/'patient_effect_histogram.csv',histogram)
    table(pub/'statistical_tests.csv',[{k:r[k] for k in ['coordinate','valid_patients','nonzero_patients','wilcoxon_statistic','pvalue','BH_q','family_tests','FDR_below_0_05']} for r in aggregate])
    table(pub/'permutation_control.csv',permutation)
    write(pub/'feature_definitions.json',protocol['coordinates'])
    times=[z['times'][z['mask'].any(1)] for a in rec['ledgers'] for z in a['records']]
    mask_count=sum(int(z['mask'].sum()) for a in rec['ledgers'] for z in a['records'])
    assert mask_count==1471965
    single=sum(len(a['records'])==1 for a in rec['ledgers'])
    audit=dict(binding=binding,patients=80,channels=7635,seizures=256,valid_channel_windows=mask_count,
        patients_by_source={s:int((centers==s).sum()) for s in protocol['sources']},coordinates=12,patient_effect_rows=len(pairs),
        finite_effects=int(np.isfinite(matrix).sum()),finite_channel_coordinate_values=int(np.isfinite(x).sum()),
        EZ_channels=int(ez.sum()),NEZ_channels=int((~ez).sum()),
        valid_window_center_min_sec=float(min(map(np.min,times))),valid_window_center_max_sec=float(max(map(np.max,times))),
        valid_pre_channel_windows=sum(int((z['mask']*(z['times']<0)[:,None]).sum()) for a in rec['ledgers'] for z in a['records']),
        valid_post_channel_windows=sum(int((z['mask']*(z['times']>=0)[:,None]).sum()) for a in rec['ledgers'] for z in a['records']),
        single_seizure_patients=single,cached_patient_summaries_bit_exact=cache_checks,
        classifier_fits=0,neural_imports=False,GPU_used=False,raw_waveform_or_window_extraction=False,
        native_amplitude_unit='Not physically certified across sources; exact native frozen cache units retained',
        original_direction_retained=True,patient_unit_not_channel_or_seizure=True)
    write(pub/'INPUT_AUDIT.json',audit)
    assert sha(REC)==REC_SHA and sha(ARCHIVE)==LABEL_SHA
    write(pub/'STATUS.json',dict(status='ANALYSIS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION',patient_effect_rows=960))
    print('ASSOCIATION_CALCULATION_COMPLETE',960,'patient effects; classifier fits=0',flush=True)


def package(runtime):
    runtime=Path(runtime);pub=runtime/'public'
    assert json.loads((pub/'STATUS.json').read_text())['status']=='COMPLETE_INDEPENDENTLY_VERIFIED_SIXFEATURE_EZ_ASSOCIATION'
    manifest=dict(schema='SIXFEATURE_EZ_ASSOCIATION_V1',files={n:sha(pub/n) for n in ALLOW},
                  source={n:sha(HERE/n) for n in SCIENCE},private_artifacts_exported=False)
    write(pub/'PACKAGE_MANIFEST.json',manifest)
    path=runtime/'public_aggregate_export_v1.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n in ALLOW+['PACKAGE_MANIFEST.json']:z.write(pub/n,n)
    print('ANONYMOUS_PACKAGE_SHA256',sha(path),path.stat().st_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['analyze','package']);parser.add_argument('--runtime',type=Path,required=True)
    args=parser.parse_args();(main if args.stage=='analyze' else package)(args.runtime)
