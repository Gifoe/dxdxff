"""Four evidence panels; only verified public tables may supply real figures."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_public import tables,row,fmt
HERE=Path(__file__).resolve().parent
COLORS={'GLOBAL':'#777777','SOURCE':'#8D6DA8','CENTER':'#24958B','SCALE':'#D99138','FULL':'#2364AA'}
def whisker(ax,x,r,color,family=False,offset=0):
    if r['mean']=='':return
    lo,hi=('family_low','family_high') if family else ('ci_low','ci_high')
    ax.plot([x+offset,x+offset],[float(r[lo]),float(r[hi])],color=color,lw=1.3);ax.plot(x+offset,float(r['mean']),'o',color=color,ms=4.5)
def render(t,out,synthetic=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'PRCOMPV1'})
    fig,ax=plt.subplots(2,2,figsize=(12.4,8.7),layout='constrained');arms=['GLOBAL','SOURCE','CENTER','SCALE','FULL'];fractions=[.25,.5,.75,1.]
    for i,a in enumerate(arms):
        for group,color,offset in [('EZ','#C27A35',-.12),('NEZ','#2364AA',.12)]:
            r=row(t['E1_GAIN'],scope='ALL',coordinate='ALL12',arm=a,group=group);whisker(ax[0,0],i,r,color,offset=offset)
    ax[0,0].axhline(0,color='#888888',ls='--',lw=.8);ax[0,0].set_xticks(range(5),['Global','Source','Center','Scale','Full']);ax[0,0].set_ylabel('Global-minus-method CRPS gain');ax[0,0].set_title('A  EZ / NEZ reference gains',loc='left',fontweight='bold')
    ax[0,0].plot([],[],'o',color='#C27A35',label='EZ');ax[0,0].plot([],[],'o',color='#2364AA',label='NEZ');ax[0,0].legend(frameon=False,ncol=2,fontsize=8)
    h=row(t['PRIMARY'],endpoint='E1_H_FULL');ax[0,0].text(.01,-.22,f"Full H = {fmt(h['mean'])}; adjusted CI [{fmt(h['family_low'])}, {fmt(h['family_high'])}]",transform=ax[0,0].transAxes,fontsize=8)
    for a in arms:
        rr=[row(t['E2_LOSS'],scope='ALL',coordinate='ALL12',fraction=f,arm=a,group='ALL') for f in fractions]
        ax[0,1].plot(np.array(fractions)*100,[float(q['mean']) for q in rr],color=COLORS[a],lw=1.2,label=a.title())
        for f,r in zip(fractions,rr):whisker(ax[0,1],100*f,r,COLORS[a])
    ax[0,1].set_xticks([25,50,75,100]);ax[0,1].set_xlabel('Reference fraction of other channels (%)');ax[0,1].set_ylabel('Held-out CRPS');ax[0,1].set_title('B  Reference-size sensitivity',loc='left',fontweight='bold');ax[0,1].legend(frameon=False,fontsize=7,ncol=3)
    d=row(t['PRIMARY'],endpoint='E2_D_FULL_25_MINUS_100');ax[0,1].text(.01,-.22,f"Full D = {fmt(d['mean'])}; adjusted CI [{fmt(d['family_low'])}, {fmt(d['family_high'])}]",transform=ax[0,1].transAxes,fontsize=8)
    rr=[row(t['E2_ANOMALY'],scope='ALL',coordinate='ALL12',fraction=f,endpoint='gap') for f in fractions]
    ax[1,0].plot(np.array(fractions)*100,[float(q['mean']) for q in rr],color='#2364AA',lw=1.3)
    for f,r in zip(fractions,rr):whisker(ax[1,0],100*f,r,'#2364AA')
    ax[1,0].axhline(0,color='#888888',ls='--',lw=.8);ax[1,0].set_xticks([25,50,75,100]);ax[1,0].set_xlabel('Reference fraction of other channels (%)');ax[1,0].set_ylabel('EZ-minus-NEZ absolute-relative gap');ax[1,0].set_title('C  Residual EZ association',loc='left',fontweight='bold')
    ax[1,0].text(.01,-.22,'Nominal 95% intervals; fraction differences have 11-family intervals.',transform=ax[1,0].transAxes,fontsize=8)
    for pct in [25,50,75]:
        r=row(t['SECONDARY'],endpoint=f'COMPOSITION_SLOPE_{pct}');whisker(ax[1,1],pct,r,'#2364AA',family=True)
        if r['mean']=='':ax[1,1].text(pct,.5,'NA',transform=ax[1,1].get_xaxis_transform(),ha='center')
    ax[1,1].axhline(0,color='#888888',ls='--',lw=.8);ax[1,1].set_xticks([25,50,75,100]);ax[1,1].set_xlim(15,110);ax[1,1].set_xlabel('Reference fraction of other channels (%)');ax[1,1].set_ylabel('CRPS change per +10pp reference EZ');ax[1,1].set_title('D  Within-query composition sensitivity',loc='left',fontweight='bold');ax[1,1].text(100,.5,'Not\nestimable',ha='center',va='center',fontsize=8,transform=ax[1,1].get_xaxis_transform())
    ax[1,1].text(.01,-.22,'Query-centered associations; 11-family adjusted intervals.',transform=ax[1,1].transAxes,fontsize=8)
    for a in ax.flat:a.grid(axis='y',alpha=.15)
    fig.suptitle('PR reference gains and composition sensitivity'+(' — SYNTHETIC QA ONLY' if synthetic else ''),fontsize=13,fontweight='bold')
    if synthetic:fig.text(.5,.5,'SYNTHETIC QA — NOT COHORT RESULTS',ha='center',rotation=22,alpha=.15,fontsize=24)
    out.mkdir(parents=True,exist_ok=True)
    for extension in ['pdf','svg','png']:
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else None
        fig.savefig(out/('pr_ez_composition.'+extension),dpi=600,bbox_inches='tight',metadata=metadata)
    plt.close(fig)
def main():render(tables(),HERE/'figures');print('FOUR_PANEL_VECTOR_AND_600DPI_FIGURE_WRITTEN')
if __name__=='__main__':main()
