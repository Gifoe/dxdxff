"""FIT-only label-blind conditional-input redundancy; no candidate exclusion."""
from scipy.stats import rankdata
from common import *


def main():
    _,_,_,_,_,_,banks,_=legal();pool=read(ROOT/'POOL_PRIVATE.pkl');assert sha(ROOT/'POOL_PRIVATE.pkl')==jr(PUB/'POOL_STATUS.json')['pool_sha256']
    x=pool['summary'];pid=pool['meta']['patient'];correlations=[];quality=[]
    for fold,bank in enumerate(banks,1):
        group=[];fallback=[];const=[];finite=[]
        for patient in sorted(bank['roles']['fit']):
            a=x[pid==patient].astype(float);valid=np.isfinite(a);den=valid.sum(0)
            mu=np.divide(np.nansum(a,0),den,out=np.zeros(64),where=den>0)
            filled=np.where(valid,a,mu);std=filled.std(0)
            ranks=np.apply_along_axis(rankdata,0,filled);ranks-=ranks.mean(0);scale=np.sqrt((ranks*ranks).sum(0));outer=scale[:,None]*scale[None,:]
            c=np.divide(ranks.T@ranks,outer,out=np.full((64,64),np.nan),where=outer>0)
            # A group match preserves mean/mean and SD/SD identity, not max over coordinates.
            g=np.stack([c[0::2,0::2],c[1::2,1::2]],-1)
            group.append(g);fallback.append(std<1e-8);const.append(np.ptp(filled,0)==0);finite.append(valid.mean(0))
        group=np.asarray(group)
        for i in range(32):
            for j in range(i+1,32):
                for k,coordinate in enumerate(['seizure_mean','seizure_population_sd']):
                    z=group[:,i,j,k];z=z[np.isfinite(z)]
                    correlations.append(dict(outer_fold=fold,feature_a=NAMES[i],feature_b=NAMES[j],coordinate=coordinate,
                        finite_patients=len(z),median_spearman=float(np.median(z)) if len(z) else np.nan,
                        q25_spearman=float(np.quantile(z,.25)) if len(z) else np.nan,q75_spearman=float(np.quantile(z,.75)) if len(z) else np.nan,
                        absolute_ge_095_fraction=float((np.abs(z)>=.95).mean()) if len(z) else np.nan,
                        scope='primary_FIT' if fold==1 else 'overlapping_FIT_sensitivity',labels_used=False))
        for j,name in enumerate(NAMES):
            for k,coordinate in enumerate(['seizure_mean','seizure_population_sd']):
                z=2*j+k;quality.append(dict(outer_fold=fold,feature=name,coordinate=coordinate,FIT_patients=len(group),
                    patient_z_std_below_1e8_fraction=float(np.mean(fallback,0)[z]),constant_patient_fraction=float(np.mean(const,0)[z]),
                    median_channel_finite_fraction=float(np.median(finite,0)[z]),labels_used=False,used_to_change_selection=False))
    pd.DataFrame(correlations).to_csv(PUB/'FIT_REDUNDANCY.csv',index=False)
    pd.DataFrame(quality).to_csv(PUB/'FIT_NUMERICAL_QUALITY.csv',index=False)
    write(PUB/'REDUNDANCY_STATUS.json',dict(status='PASS',correlation_rows=len(correlations),numerical_quality_rows=len(quality),
        labels_used=False,Outer_channels_used=False,selection_exclusions_or_model_changes=False,
        interpretation='Correlation alone is not conditional redundancy; candidate replacement remains the predictive test.',source_sha256=sha(HERE/'redundancy.py')))
    print('FIT_LABEL_BLIND_REDUNDANCY_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
