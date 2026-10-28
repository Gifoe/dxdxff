"""Publication plots from anonymous aggregate exports only."""
from pathlib import Path
import csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent.parent
NAMES={'LINEAR_SVM':'Linear SVM','TIMECONV':'TimeConv-CNN','DRECHANNET':'DREChanNet'}
COLORS={'LINEAR_SVM':'#777777','TIMECONV':'#42949E','DRECHANNET':'#0F4D92'}
SOURCES=['hup','lzu','multicenter','pediatric'];SC=['#0F4D92','#42949E','#9A4D8E','#C58B47'];SL=['HUP','LZU','Multi.','Ped.']
available={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({'font.family':[f for f in ['Arial','Helvetica','DejaVu Sans'] if f in available],'font.size':7.5,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def read(pub,n):
    with (pub/(n+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/(name+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

def figure_a(pub):
    data=read(pub,'PATIENT_DISTRIBUTION_BINS');paired=read(pub,'TABLE_B_SOURCE');fig,axes=plt.subplots(2,3,figsize=(7.15,4.0),sharey=True)
    grid=np.linspace(-100,100,401);h=7.5
    for col,(model,name) in enumerate(NAMES.items()):
        for row,metric in enumerate(['auroc','auprc']):
            ax=axes[row,col]
            for x,(src,color) in enumerate(zip(SOURCES,SC)):
                rr=[r for r in data if r['model']==model and r['metric']==metric and r['source']==src]
                centers=np.array([(float(r['bin_low'])+float(r['bin_high']))*50 for r in rr]);weights=np.array([int(r['count']) for r in rr]);assert weights.sum()>0
                density=(np.exp(-.5*((grid[:,None]-centers[None,:])/h)**2)*weights).sum(1)/(weights.sum()*h*np.sqrt(2*np.pi))
                width=density/density.max()*.36;occupied=np.flatnonzero(weights)
                support=(grid>=float(rr[int(occupied[0])]['bin_low'])*100)&(grid<=float(rr[int(occupied[-1])]['bin_high'])*100)
                ax.fill_betweenx(grid[support],x-width[support],x+width[support],color=color,alpha=.50,linewidth=.6,edgecolor=color)
                point=next(r for r in paired if r['model']==model and r['source']==src and r['metric']==metric)
                mean,lo,hi=[100*float(point[k]) for k in ['mean','low','high']]
                ax.errorbar(x,mean,yerr=np.array([[mean-lo],[hi-mean]]),fmt='o',ms=2.8,color='#272727',mfc='white',mew=.6,lw=.7,capsize=1.8,zorder=4)
            ax.axhline(0,color='#777777',lw=.65,ls=(0,(3,3)),zorder=0);ax.set_ylim(-100,100);ax.set_yticks([-100,-50,0,50,100]);ax.set_xticks(range(4),SL if row else ['']*4);ax.set_xlim(-.6,3.6)
            if row==0:ax.set_title(name,color=COLORS[model],fontweight='bold',pad=7)
            if col==0:ax.set_ylabel(('AUROC' if row==0 else 'AP')+' gain (pp)')
            ax.text(-.12,1.04,chr(97+row*3+col),transform=ax.transAxes,fontweight='bold',fontsize=9)
    fig.subplots_adjust(wspace=.18,hspace=.18,left=.09,right=.995,top=.90,bottom=.08);save(fig,'Figure_A_patient_gain_distributions')

def figure_b(pub):
    data=read(pub,'COVERAGE_CURVE');fig,axes=plt.subplots(1,2,figsize=(7.15,2.65),gridspec_kw={'width_ratios':[1.25,1]})
    for model in NAMES:
        for arm in ['NO_PR','FULL_PR']:
            rr=sorted([r for r in data if r['model']==model and r['condition']==arm],key=lambda r:float(r['review_fraction']))
            x=np.array([float(r['review_fraction'])*100 for r in rr]);mean=np.array([float(r['mean'])*100 for r in rr]);lo=np.array([float(r['low'])*100 for r in rr]);hi=np.array([float(r['high'])*100 for r in rr])
            for ax in axes:
                ax.plot(x,mean,color=COLORS[model],lw=1.35,ls='-' if arm=='FULL_PR' else (0,(4,2)),alpha=1 if arm=='FULL_PR' else .75)
                if arm=='FULL_PR':ax.fill_between(x,lo,hi,color=COLORS[model],alpha=.07,lw=0)
    for j,ax in enumerate(axes):
        ax.set_xlabel('Channel review budget (%)');ax.set_ylim(0,100);ax.set_xlim(0,100 if j==0 else 20);ax.set_yticks([0,25,50,75,100]);ax.text(-.12,1.03,chr(97+j),transform=ax.transAxes,fontweight='bold',fontsize=9)
    axes[0].set_ylabel('EZ channels identified (%)');axes[0].set_xticks([0,20,40,60,80,100]);axes[1].set_xticks([0,5,10,15,20]);axes[1].set_title('Early review',fontsize=8,pad=5)
    handles=[Line2D([0],[0],color=c,lw=1.6,label=NAMES[m]) for m,c in COLORS.items()]+[Line2D([0],[0],color='#333333',lw=1.3,ls=(0,(4,2)),label='No PR'),Line2D([0],[0],color='#333333',lw=1.3,label='Full PR')]
    fig.legend(handles=handles,loc='lower center',ncol=5,bbox_to_anchor=(.52,-.01),columnspacing=1.4,handlelength=2.2)
    fig.subplots_adjust(left=.09,right=.995,top=.92,bottom=.25,wspace=.24);save(fig,'Figure_B_EZ_coverage')

def figure_c(pub):
    if not (pub/'TABLE_E.csv').exists():return
    data=read(pub,'TABLE_E');fig,axes=plt.subplots(2,3,figsize=(7.15,3.9),sharex=True)
    for col,(model,name) in enumerate(NAMES.items()):
        for row,(metric,title) in enumerate([('auroc','AUROC (%)'),('macro_f1','Macro-F1 (%)')]):
            ax=axes[row,col]
            for arm in ['NO_PR','FULL_PR']:
                lookup={r['K']:r for r in data if r['model']==model and r['condition']==arm and r['metric']==metric};rr=[lookup[k] for k in ['1','2','All']]
                a=np.array([[float(r[k])*100 for k in ['mean','low','high']] for r in rr]);x=np.arange(3)+(-.035 if arm=='NO_PR' else .035)
                ax.errorbar(x,a[:,0],yerr=np.stack([a[:,0]-a[:,1],a[:,2]-a[:,0]]),color=COLORS[model],ls=(0,(4,2)) if arm=='NO_PR' else '-',marker='o',mfc='white' if arm=='NO_PR' else COLORS[model],ms=3.5,lw=1.2,capsize=2,elinewidth=.8,alpha=.65 if arm=='NO_PR' else 1)
            ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_xlim(-.25,2.25);ax.set_xticks(range(3),['1','2','All']);ax.text(-.12,1.04,chr(97+row*3+col),transform=ax.transAxes,fontweight='bold',fontsize=9)
            if row==0:ax.set_title(name,color=COLORS[model],fontweight='bold',pad=7)
            if row==1:ax.set_xlabel('Seizures available')
            if col==0:ax.set_ylabel(title)
    handles=[Line2D([0],[0],color='#333333',ls=(0,(4,2)),marker='o',mfc='white',ms=3,label='No PR'),Line2D([0],[0],color='#333333',marker='o',ms=3,label='Full PR')]
    fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(.54,-.005));fig.subplots_adjust(left=.085,right=.995,top=.91,bottom=.16,wspace=.23,hspace=.24);save(fig,'Figure_C_seizure_sufficiency')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,default=ROOT/'results/public_abc');a=p.parse_args();figure_a(a.public);figure_b(a.public);figure_c(a.public)
    print('PUBLIC_FIGURES_EXPORTED')
