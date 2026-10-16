"""Anonymous aggregate-only publication plots; Python/matplotlib backend."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE=Path(__file__).resolve().parent;DATA=HERE/'results';OUT=HERE/'figures'
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,
    'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'axes.linewidth':.6,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':600,'lines.linewidth':1.2})
width_mm=180
WIDTH=width_mm/25.4
NAMES=json.loads((DATA/'POOL_STATUS.json').read_text())['names']
LABELS=['Beta power','Low-gamma power','Partial HG power (80–125)','RMS (legacy)','Variance','Line length','Spectral entropy',
    'Log variance','Hjorth mobility (stable)','Hjorth complexity (stable)','Permutation entropy','Spectral centroid','Log gamma/beta',
    'Partial HG fraction','Spectral flatness','Slope proxy','Normalized line length','Clipped kurtosis','Log beta (stable)',
    'Log low gamma (stable)','Log partial HG (stable)','Partial alpha (10–13)','Total power (10–125)','Hjorth mobility (legacy)',
    'Hjorth complexity (legacy)','Graph degree','Graph strength','Graph clustering','Graph eigenvector','Graph PageRank','Graph k-core','Graph local efficiency']
SHORT=dict(zip(NAMES,LABELS));COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
METRICS=['auroc','auprc','macro_f1','f1','mrr','hit_at_1'];MLABELS=['Patient AUROC','Patient AP','Patient Macro-F1','Patient EZ-F1','Patient MRR','Patient Top-1']


def save(fig,name,data):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white')
    fig.savefig(OUT/(name+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    data.to_csv(OUT/(name+'_source.csv'),index=False)
    plt.close(fig)


def clean(ax):
    ax.spines[['top','right']].set_visible(False)


def physiology():
    d=pd.read_csv(DATA/'PHYSIOLOGICAL_EFFECTS.csv');d=d[d.outer_fold==1]
    phases=['pre','early0to8','late8to20','early_minus_pre','late_minus_pre']
    a=d.pivot(index='feature',columns='phase',values='mean_effect').loc[NAMES,phases]
    fig,ax=plt.subplots(figsize=(WIDTH,6.6));fig.subplots_adjust(left=.30,right=.85,bottom=.13,top=.93)
    im=ax.imshow(a,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.7,vcenter=0,vmax=.7),interpolation='nearest')
    ax.set_yticks(range(32),LABELS);ax.set_xticks(range(5),['Pre','Early\n0–8 s','Late\n8–20 s','Early − pre','Late − pre'])
    for _,r in d[d.operationally_reproducible].iterrows():ax.plot(phases.index(r.phase),NAMES.index(r.feature),'.',color='black',markersize=2)
    ax.set_title('Within-patient physiological association',loc='left',fontweight='bold',pad=12)
    ax.set_xlabel('Cache-relative phase; primary FIT cohort, 51 patients')
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.05);cb.set_label('Mean rank-biserial effect (EZ − NEZ)')
    fig.text(.30,.03,'Dot: prespecified patient/source/seizure reproducibility rule.\nBH correction: all 160 feature–phase tests. Not a selection filter.',fontsize=6)
    save(fig,'01_physiological_effects',d)
    # Six trajectories were fixed in BIOLOGY_LOCK before association analysis.
    names=['line_length_per_sec','partial_high_gamma_80_125','spectral_entropy_10_125','normalized_line_length','clipped_excess_kurtosis','rebuilt_strength_norm']
    t=pd.read_csv(DATA/'PHYSIOLOGY_TRAJECTORY.csv');t=t[t.feature.isin(names)]
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,3.8),sharex=True,sharey=True,layout='constrained')
    for ax,name in zip(axes.flat,names):
        g=t[t.feature==name].sort_values('cache_relative_display_sec');x=g.cache_relative_display_sec.to_numpy()
        ax.fill_between(x,g.ci_low,g.ci_high,color=COLORS[0],alpha=.17,linewidth=0);ax.plot(x,g.mean_effect,color=COLORS[0])
        ax.axhline(0,color='.55',linewidth=.6);ax.axvline(0,color='.4',linestyle=':',linewidth=.7);ax.set_title(SHORT[name],loc='left');clean(ax)
    axes[0,0].set_ylabel('Mean effect (EZ − NEZ)');axes[1,0].set_ylabel('Mean effect (EZ − NEZ)')
    for ax in axes[-1]:ax.set_xlabel('Cache-relative time (s)')
    fig.suptitle('Patient-unit trajectories; 95% patient-bootstrap intervals',fontsize=8)
    save(fig,'02_phase_trajectories',t)


def stability():
    d=pd.read_csv(DATA/'SELECTION_FREQUENCY.csv');a=d.pivot(index='feature',columns='k',values='frequency').loc[NAMES]
    s=pd.read_csv(DATA/'SELECTION_STABILITY.csv');g=s.groupby('k').agg(jaccard=('jaccard','mean'),adjusted=('chance_adjusted_overlap','mean')).reset_index()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(WIDTH,6.2),gridspec_kw={'width_ratios':[2.2,1]});fig.subplots_adjust(left=.29,right=.96,bottom=.13,top=.92,wspace=.45)
    im=ax.imshow(a,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest');ax.set_yticks(range(32),LABELS);ax.set_xticks(range(9),range(1,10));ax.set_xlabel('Number of measurement groups (k)');ax.set_title('a  Assessment-path selection frequency',loc='left')
    cb=fig.colorbar(im,ax=ax,location='bottom',fraction=.035,pad=.08);cb.set_label('Fraction of 15 overlapping assessment paths')
    bx.plot(g.k,g.jaccard,'o-',color=COLORS[0],markersize=3,label='Jaccard');bx.plot(g.k,g.adjusted,'s--',color=COLORS[1],markersize=3,label='Chance-adjusted')
    bx.axhline(0,color='.65',linewidth=.6);bx.set_xticks([1,3,5,7,9]);bx.set_xlabel('k');bx.set_ylabel('Mean overlap');bx.set_title('b  Path stability',loc='left');bx.legend(frameon=False,loc='upper left');clean(bx)
    fig.text(.29,.02,'Overlapping training sets are not independent replications; frequencies did not define a global Outer subset.',fontsize=6)
    save(fig,'04_feature_stability',d)


def redundancy():
    d=pd.read_csv(DATA/'FIT_REDUNDANCY.csv');d=d[d.outer_fold==1];a=np.eye(32)
    for (x,y),g in d.groupby(['feature_a','feature_b']):a[NAMES.index(x),NAMES.index(y)]=a[NAMES.index(y),NAMES.index(x)]=g.median_spearman.mean()
    fig,ax=plt.subplots(figsize=(WIDTH,6.8));fig.subplots_adjust(left=.28,right=.88,bottom=.29,top=.94)
    im=ax.imshow(a,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest');ax.set_yticks(range(32),LABELS);ax.set_xticks(range(32),[str(i+1) for i in range(32)],rotation=90)
    ax.set_xlabel('Feature index (same order as rows)');ax.set_title('Within-patient rank correlation of DELTA summaries',loc='left',pad=12)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04);cb.set_label('Mean of median mean/SD coordinate correlations')
    fig.text(.28,.15,'Primary FIT cohort; no labels used.\nConstant-coordinate correlations are unavailable, not zero.\nCorrelation is not a test of conditional predictive redundancy.',fontsize=6)
    save(fig,'06_redundancy',d)


def final():
    d=pd.read_csv(DATA/'CARDINALITY_CURVE.csv');f=pd.read_csv(DATA/'CARDINALITY_ASSESSMENT_BY_FOLD.csv')
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,4.4),layout='constrained')
    for ax,m,label in zip(axes.flat,METRICS,MLABELS):
        for fold,h in f.groupby('outer_fold'):ax.plot(h.k,h[m],color='.75',linewidth=.6,alpha=.7)
        h=d[d.metric==m].sort_values('k');ax.fill_between(h.k,h.ci_low,h.ci_high,color=COLORS[0],alpha=.17,linewidth=0);ax.plot(h.k,h['mean'],'o-',color=COLORS[0],markersize=3)
        ax.set_title(label,loc='left');ax.set_xticks([1,3,5,7,9]);ax.set_xlabel('Measurement groups (k)');clean(ax)
    fig.suptitle('Nested FIT assessment: fixed linear probe, not neural Outer performance',fontsize=8)
    save(fig,'03_feature_cardinality',d)
    p=pd.read_csv(DATA/'outer/PAIRED_COMPARISONS.csv');refs=['M8__OLD7','M8__BEST_SWAP','M8__RANDOM_A','M8__RANDOM_B','Concat__RAW_ONLY','Concat__RAW_SUPPORT30','Concat__SELECTED','D1__SELECTED']
    labels=['Matched M8 + old7','Matched M8 + best swap','Matched M8 + random A','Matched M8 + random B','Raw-only (0–20 s)','Raw-only (0–30 s)','Single-head selected','Physiology-only selected']
    d=p[(p.model=='M8__SELECTED')&p.reference.isin(refs)&p.metric.isin(METRICS)]
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,5.4),layout='constrained')
    for ax,m,label in zip(axes.flat,METRICS,MLABELS):
        h=d[d.metric==m].set_index('reference').loc[refs];y=np.arange(len(h));v=h.delta.to_numpy();lo=h.ci_low.to_numpy();hi=h.ci_high.to_numpy()
        ax.errorbar(v,y,xerr=np.vstack([v-lo,hi-v]),fmt='o',color=COLORS[0],markersize=3,capsize=2,elinewidth=.8);ax.axvline(0,color='.5',linewidth=.7)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(label,loc='left');ax.set_xlabel('Selected M8 − reference');clean(ax)
    fig.suptitle('Exploratory frozen Outer contrasts; unadjusted 95% paired patient intervals',fontsize=8)
    save(fig,'05_matched_replacement_raw_increment',d)


if __name__=='__main__':
    physiology();stability();redundancy()
    if (DATA/'INDEPENDENT_RESULTS_VERIFICATION.json').exists():
        assert json.loads((DATA/'INDEPENDENT_RESULTS_VERIFICATION.json').read_text())['status']=='PASS';final()
    print('AGGREGATE_FIGURES_RENDERED')
