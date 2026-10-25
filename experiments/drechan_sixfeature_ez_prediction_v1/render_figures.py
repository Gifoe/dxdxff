"""Figure B from independently verified anonymous patient-level aggregates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
NAMES=['Beta','Low gamma','High gamma (80–125)','RMS','Line Length','Spectral Entropy','All six (w/o PR)','All six (+PR)']
ORDER=['BETA_PR','LOW_GAMMA_PR','HIGH_GAMMA_PR','RMS_PR','LINE_LENGTH_PR','SPECTRAL_ENTROPY_PR','ALL_SIX_NO_PR','ALL_SIX_PR']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.dpi':600})


def export(fig,name,width,height):
    out=HERE/'figures';out.mkdir(exist_ok=True);fig.canvas.draw();renderer=fig.canvas.get_renderer();bad=[]
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:bad.append(text.get_text())
    assert not bad,('TEXT_OUTSIDE_CANVAS',name,bad)
    fig.savefig(out/(name+'.pdf'),dpi=600,facecolor='white')
    fig.savefig(out/(name+'.svg'),dpi=600,facecolor='white')
    fig.savefig(out/(name+'.png'),dpi=600,facecolor='white')
    plt.close(fig)
    return dict(figure=name,width_mm=width,height_mm=height,png_dpi=600,minimum_font_pt=7,
                text_outside_canvas=bad,individual_data_used=False,editable_text=True)


def main():
    data=HERE/'results';assert json.loads((data/'STATUS.json').read_text(encoding='utf-8'))['status']=='COMPLETE_VERIFIED_SIXFEATURE_EZ_PREDICTION'
    frame=pd.read_csv(data/'main_results.csv').set_index('arm').loc[ORDER]
    assert len(frame)==8 and frame.patients.eq(80).all()
    fig,ax=plt.subplots(figsize=(89/25.4,91/25.4))
    fig.subplots_adjust(left=.415,right=.965,bottom=.19,top=.84)
    ax.axvline(.5,color='#6B7280',lw=.7,ls=(0,(3,3)),zorder=0)
    for j,arm in enumerate(ORDER):
        r=frame.loc[arm];y=7-j
        if j>=6:ax.axhspan(y-.42,y+.42,color='#F4F6F8',zorder=-2)
        color='#B67C36' if arm.endswith('_NO_PR') else '#386B95'
        marker='s' if arm.endswith('_NO_PR') else 'o'
        ax.plot([r.AUROC_ci_low,r.AUROC_ci_high],[y,y],color=color,lw=1.1,zorder=2)
        ax.plot(r.AUROC,y,marker=marker,ms=3.6,mfc='white' if arm.endswith('_NO_PR') else color,mec=color,mew=.8,zorder=3)
    ax.set_ylim(-.6,7.6);ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1])
    ax.set_yticks(range(8));ax.set_yticklabels(NAMES[::-1]);ax.tick_params(axis='y',length=0,pad=4)
    ax.tick_params(axis='x',length=2.5,width=.6);ax.spines['left'].set_visible(False)
    ax.grid(axis='x',lw=.4,color='#E5E7EB');ax.set_xlabel('Patient-average AUROC',labelpad=6)
    fig.suptitle('Physiological feature prediction',fontsize=8,y=.97)
    fig.text(.04,.9,'80 patients; five patient-held-out folds',fontsize=7)
    fig.text(.04,.035,'Dots: mean; lines: nominal 95% patient CI',fontsize=7)
    qa=[export(fig,'feature_prediction',89,91)]
    null=pd.read_csv(data/'negative_control.csv');summary=pd.read_csv(data/'negative_control_summary.csv').set_index('metric')
    assert len(null)==100
    fig,ax=plt.subplots(figsize=(89/25.4,65/25.4))
    fig.subplots_adjust(left=.17,right=.96,bottom=.22,top=.69)
    edges=np.linspace(0,1,41);ax.hist(null.AUROC,bins=edges,color='#8FA6B9',edgecolor='white',linewidth=.3)
    ax.axvline(.5,color='#6B7280',ls='--',lw=.7,label='Chance (0.5)')
    observed=float(summary.loc['AUROC','observed']);ax.axvline(observed,color='#386B95',lw=1,label='True-label full +PR')
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_ylabel('Permutation repeats');ax.set_xlabel('Patient-average AUROC')
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4,integer=True));ax.tick_params(length=2.5,width=.6)
    fig.suptitle('Training-label permutation control',fontsize=8,y=.97)
    fig.text(.17,.87,'100 repeats; true Outer labels retained',fontsize=7)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc='upper left',bbox_to_anchor=(.17,.845),
               borderaxespad=0,handlelength=1.1,fontsize=7)
    qa.append(export(fig,'negative_control',89,65))
    (HERE/'FIGURE_QA.json').write_text(json.dumps(dict(backend='Python/matplotlib',figures=qa,
        rows_plotted=8,patients_retained=80,negative_repeats_retained=100,visual_review='PENDING'),indent=2)+'\n',encoding='utf-8')
    print('PREDICTION_FIGURES_EXPORTED 2figures x PDF/SVG/600dpiPNG;manualQA pending')


if __name__=='__main__':main()
