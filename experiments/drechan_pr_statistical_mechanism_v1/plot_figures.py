"""Publication plots from authenticated anonymous aggregates only."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ARMS=['G0_GLOBAL','G1_CENTER','G2_SCALE','G3_FULL']
LABELS=['Global','Center only','Scale only','Full PR']
COLORS=['#77828A','#8D78A5','#D19047','#367F9B']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
    'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'pr-mechanism-v1',
    'legend.frameon':False,'lines.linewidth':1.1,'savefig.facecolor':'white'})


def table(name):return pd.read_csv(HERE/'results'/(name+'.csv'),float_precision='round_trip')


def primary(df):return df[(df.estimator=='DIRECT')&np.isclose(df.factor,1e-8,rtol=1e-12,atol=0)]


def interval(ax,y,mean,low,high,color,marker='o',style='-'):
    ax.plot([low,high],[y,y],color=color,lw=1.5,ls=style,zorder=2)
    ax.plot(mean,y,marker=marker,color=color,ms=4,zorder=3)


def label(ax,letter,title):
    ax.set_title(title,loc='left',pad=10,fontweight='normal')
    ax.text(-.12,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,va='top')


def forest(ax,df,xlabel):
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        r=df[df.arm==arm].iloc[0];interval(ax,3-i,r['mean'],r.ci_low,r.ci_high,color)
    ax.set_yticks(range(4),LABELS[::-1]);ax.set_ylim(-.6,3.6)
    ax.axvline(0,color='#BBBBBB',lw=.7,zorder=0);ax.set_xlabel(xlabel)
    ax.grid(axis='x',alpha=.14)


def save(fig,name):
    out=HERE/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None},bbox_inches='tight')
    fig.savefig(out/(name+'.svg'),metadata={'Date':None},bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight')
    plt.close(fig)


def main():
    snr=primary(table('transferable_snr'));align=primary(table('representation_alignment'))
    pairs=primary(table('paired_intervals'));null=table('negative_control_summary')
    null=null[(null.estimator=='DIRECT')&(null.diagnostic=='MATCHED_H3_LABEL_NULL')]
    fig=plt.figure(figsize=(7.20,5.55),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=[1,1.13],hspace=.16,wspace=.18)
    ax=fig.add_subplot(gs[0,0]);label(ax,'a','Patient coordinate transformation')
    xx=np.linspace(-5,5,400)
    # Deliberately schematic analytic density curves, never cohort observations.
    for offset,scale,c in [(-1.5,.65,COLORS[0]),(.7,1.25,COLORS[1]),(2.0,.85,COLORS[2])]:
        density=np.exp(-.5*((xx-offset)/scale)**2)/scale
        ax.plot(xx,density/np.max(density)*.65+1.2,color=c,alpha=.9)
    ax.plot(xx,np.exp(-.5*xx**2)*.65-.05,color=COLORS[3],lw=2)
    ax.text(0,2.0,'Heterogeneous patient coordinates',ha='center',fontsize=7)
    ax.text(0,.86,r'$x\ \longmapsto\ (x-\mu_p)/s_p$',ha='center',fontsize=10)
    ax.text(0,-.35,'Unlabeled, complete-patient reference',ha='center',fontsize=7)
    ax.text(0,-.65,'Schematic; not patient observations',ha='center',fontsize=6.5,color='#555555')
    ax.set(xlim=(-5,5),ylim=(-.8,2.2),xticks=[],yticks=[])
    for s in ax.spines.values():s.set_visible(False)
    ax=fig.add_subplot(gs[0,1]);label(ax,'b','Transferable standardized separation')
    forest(ax,snr,'Signed separation, T (mean and 95% CI)')
    r=pairs[(pairs.target=='G3_FULL')&(pairs.control=='G0_GLOBAL')&(pairs.metric=='T')].iloc[0]
    ax.text(.02,.02,f"Full − Global: {r.effect:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
        transform=ax.transAxes,fontsize=6.4,va='bottom')
    ax=fig.add_subplot(gs[1,0]);label(ax,'c','Shared-direction alignment')
    forest(ax,align,'Cosine alignment, A (mean and 95% CI)')
    r=pairs[(pairs.target=='G3_FULL')&(pairs.control=='G0_GLOBAL')&(pairs.metric=='A')].iloc[0]
    ax.text(.02,.02,f"ΔA: {r.effect:+.3f}; adjusted CI [{r.adjusted_ci_low:+.3f}, {r.adjusted_ci_high:+.3f}]",
        transform=ax.transAxes,fontsize=6.2,va='bottom')
    ax=fig.add_subplot(gs[1,1]);label(ax,'d','Midpoint displacement reduction')
    for i,arm in enumerate(['G1_CENTER','G2_SCALE','G3_FULL']):
        r=pairs[(pairs.target==arm)&(pairs.control=='G0_GLOBAL')&(pairs.metric=='MAE')].iloc[0]
        q=null[(null.target==arm)&(null.control=='G0_GLOBAL')&(null.metric=='MAE')].iloc[0]
        interval(ax,2-i+.09,r.effect,r.adjusted_ci_low,r.adjusted_ci_high,COLORS[i+1])
        interval(ax,2-i-.12,q.null_mean,q.null_reference_low,q.null_reference_high,'#999999',marker='|',style='--')
    ax.set_yticks([0,1,2],['Full PR','Scale only','Center only']);ax.set_ylim(-.7,2.6)
    ax.axvline(0,color='#BBBBBB',lw=.7);ax.grid(axis='x',alpha=.14)
    ax.set_xscale('symlog',linthresh=2,linscale=1)
    ax.set_xlim(-80,2);ax.set_xticks([-60,-20,-5,0,1])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Global − arm: mean |midpoint displacement| (symlog)')
    ax.text(.02,.01,'Solid: adjusted patient CI\nDashed: matched-label-null reference range',
        transform=ax.transAxes,fontsize=6.2,va='bottom',color='#555555')
    fig.suptitle('Patient-relative normalization: a held-out statistical probe',fontsize=10,y=1.03)
    fig.supxlabel('n = 80 unique patients; one held-out assessment per patient; fixed TRAIN-derived directions',fontsize=6.8)
    save(fig,'pr_statistical_mechanism')

    hist=table('patient_distribution_bins');q=hist[(hist.arm=='G3_MINUS_G0')&(hist.metric=='PAIRED_T')]
    assert (q['count'].sum()+q.underflow.sum()+q.overflow.sum())==80
    categories=[];counts=[];colors=[]
    if q.underflow.sum():categories.append(f"< {q.bin_low.min():g}");counts.append(q.underflow.sum());colors.append('#B58B86')
    for r in q.itertuples():
        categories.append(f'[{r.bin_low:g}, {r.bin_high:g}{"]" if r.bin_high==q.bin_high.max() else ")"}')
        counts.append(r.count);colors.append('#B58B86' if r.bin_high<=0 else COLORS[3])
    if q.overflow.sum():categories.append(f"> {q.bin_high.max():g}");counts.append(q.overflow.sum());colors.append(COLORS[3])
    fig,ax=plt.subplots(figsize=(7.20,2.75),layout='constrained')
    ax.bar(np.arange(len(counts)),counts,color=colors,width=.8)
    for i,c in enumerate(counts):ax.text(i,c+.25,str(int(c)),ha='center',fontsize=6.5)
    ax.set_xticks(np.arange(len(counts)),categories,rotation=35,ha='right')
    ax.set_ylabel('Patients');ax.set_xlabel('Paired change in T: Full PR − Global (fixed bins; widths differ)')
    ax.set_title('Patient-effect distribution; all 80 held-out patients',loc='left')
    ax.set_ylim(0,max(counts)*1.2+1);save(fig,'paired_patient_effect_distribution')

    src=primary(table('source_results'));fig,axes=plt.subplots(1,2,figsize=(7.20,3.5),layout='constrained')
    strata=['hup','lzu','multicenter','pediatric'];names=['HUP (36)','LZU (21)','Multicenter (15)','Pediatric (8)']
    for ax,metric,title in zip(axes,['T','A'],['Transferable separation','Direction alignment']):
        for i,s in enumerate(strata):
            for j,(arm,color) in enumerate(zip(ARMS,COLORS)):
                r=src[(src.source==s)&(src.arm==arm)].iloc[0]
                interval(ax,3-i+(j-1.5)*.13,r[metric+'_mean'],r[metric+'_ci_low'],r[metric+'_ci_high'],color)
        ax.set_yticks(range(4),names[::-1]);ax.set_ylim(-.55,3.55)
        ax.axvline(0,color='#BBBBBB',lw=.7);ax.set_xlabel(metric+' (mean and nominal 95% CI)')
        ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.14)
    handles=[plt.Line2D([],[],color=c,marker='o',label=l,lw=1) for c,l in zip(COLORS,LABELS)]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,fontsize=7)
    fig.suptitle('Four-source descriptive results; no subgroup selection',fontsize=9)
    save(fig,'source_stratified_geometry')


if __name__=='__main__':main()
