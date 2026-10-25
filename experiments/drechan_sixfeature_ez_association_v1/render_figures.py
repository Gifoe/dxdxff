"""Publication figures from independently verified anonymous aggregates only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures';DATA=HERE/'results'
MM=1/25.4
BLUE='#386B95';ORANGE='#B67C36';GRAY='#6B7280'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'savefig.dpi':600})
SHORT=['Beta','Low gamma','Partial high gamma','RMS','Line Length','Spectral Entropy']
LABELS=[f'{name}: {suffix}' for name in SHORT for suffix in ['mean','seizure SD']]
QA=[]


def save(fig,name,width,height):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bad=[]
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text():continue
        box=item.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            bad.append(item.get_text())
    assert not bad,('TEXT_OUTSIDE_CANVAS',name,bad)
    fig.savefig(OUT/f'{name}.pdf',dpi=600,facecolor='white')
    fig.savefig(OUT/f'{name}.svg',dpi=600,facecolor='white')
    fig.savefig(OUT/f'{name}.png',dpi=600,facecolor='white')
    QA.append(dict(figure=name,width_mm=width,height_mm=height,png_dpi=600,coordinates=12,
        text_outside_canvas=bad,editable_svg_text=True,pdf_fonttype=42,individual_data_used=False))
    plt.close(fig)


def style(ax):
    ax.axvline(0,color=GRAY,lw=.7,ls=(0,(3,3)),zorder=0)
    ax.set_xlim(-1,1);ax.set_xticks([-1,-.5,0,.5,1]);ax.tick_params(length=2.5,width=.6)
    ax.spines['left'].set_visible(False);ax.grid(axis='x',color='#E5E7EB',lw=.4)


def marks(ax,table,order,show_q=False):
    for j,key in enumerate(order):
        r=table.loc[key];color=BLUE if j%2==0 else ORANGE;marker='o' if j%2==0 else 's';y=11-j
        ax.plot([r.ci_low,r.ci_high],[y,y],color=color,lw=1.1,zorder=2)
        face='white' if 'BH_q' in r and r.BH_q>=.05 else color
        ax.plot(r.mean_effect,y,marker=marker,ms=4.2,mfc=face,mec=color,mew=.9,zorder=3)
        if show_q:ax.text(1.055,y,f'{r.BH_q:.2g}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=7.2)
    ax.set_ylim(-.65,11.65);ax.set_yticks(range(12));ax.set_yticklabels(LABELS[::-1])
    ax.tick_params(axis='y',length=0,pad=5)


def main():
    assert json.loads((DATA/'STATUS.json').read_text(encoding='utf-8'))['status']=='COMPLETE_INDEPENDENTLY_VERIFIED_SIXFEATURE_EZ_ASSOCIATION'
    defs=json.loads((DATA/'feature_definitions.json').read_text(encoding='utf-8'));order=[r['coordinate'] for r in defs]
    assoc=pd.read_csv(DATA/'feature_association.csv').set_index('coordinate').loc[order]
    source=pd.read_csv(DATA/'source_association.csv');hist=pd.read_csv(DATA/'patient_effect_histogram.csv')
    assert len(assoc)==12 and len(source)==48 and len(hist)==480
    OUT.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(178*MM,118*MM))
    fig.subplots_adjust(left=.285,right=.825,bottom=.17,top=.87)
    for j in range(6):
        if j%2==0:ax.axhspan(10-2*j-.42,11-2*j+.42,color='#F4F6F8',zorder=-1)
    style(ax);marks(ax,assoc,order,True)
    ax.text(1.055,1.035,'BH q',transform=ax.transAxes,fontsize=8,ha='left')
    ax.set_xlabel('Signed rank-biserial effect (EZ higher →)',labelpad=7)
    fig.suptitle('Physiological Differences Between EZ and NEZ Channels',fontsize=10,y=.965)
    fig.text(.285,.9,'Mean patient effect and nominal 95% bootstrap CI; n = 80',fontsize=7)
    handles=[Line2D([],[],color=BLUE,marker='o',lw=1,label='Across-seizure mean'),
             Line2D([],[],color=ORANGE,marker='s',lw=1,label='Across-seizure SD')]
    fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(.53,.012),frameon=False)
    save(fig,'feature_association',178,118)

    fig,axes=plt.subplots(2,2,figsize=(178*MM,173*MM),sharex=True,sharey=True)
    fig.subplots_adjust(left=.215,right=.965,bottom=.10,top=.91,wspace=.22,hspace=.27)
    names={'hup':'HUP','lzu':'LZU','multicenter':'Multicenter','pediatric':'Pediatric'}
    for ax,z in zip(axes.flat,['hup','lzu','multicenter','pediatric']):
        sub=source[source.source==z].set_index('coordinate').loc[order]
        style(ax);marks(ax,sub,order)
        ax.set_title(f'{names[z]} (n = {int(sub.cohort_patients.iloc[0])})',loc='left',fontweight='bold')
    fig.suptitle('Source-specific physiological associations',fontsize=10,y=.97)
    fig.supxlabel('Signed rank-biserial effect',y=.035,fontsize=8)
    save(fig,'source_association',178,173)

    fig,axes=plt.subplots(6,2,figsize=(178*MM,215*MM),sharex=True)
    fig.subplots_adjust(left=.10,right=.97,bottom=.07,top=.93,hspace=.64,wspace=.22)
    for j,(ax,key) in enumerate(zip(axes.flat,order)):
        sub=hist[hist.coordinate==key].sort_values('bin_left');color=BLUE if j%2==0 else ORANGE
        assert sub.patients.sum()==assoc.loc[key,'valid_patients']
        ax.bar(sub.bin_left,sub.patients,width=sub.bin_right-sub.bin_left,align='edge',color=color,alpha=.85,linewidth=0)
        ax.axvline(0,color=GRAY,lw=.6,ls='--');ax.set_xlim(-1,1);ax.set_xticks([-1,0,1])
        ax.set_title(LABELS[j],loc='left',fontsize=8,pad=3)
        ax.tick_params(length=2,labelsize=7);ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3,integer=True))
    fig.suptitle('Within-patient effect distributions (all 12 coordinates)',fontsize=10,y=.98)
    fig.text(.10,.95,'Fixed 0.05-wide bins; all 80 patients retained; no individual values exported',fontsize=7)
    fig.supxlabel('Signed patient rank-biserial effect',y=.018,fontsize=8)
    fig.supylabel('Patients per bin',x=.015,fontsize=8)
    save(fig,'patient_effect_distributions',178,215)
    (HERE/'FIGURE_QA.json').write_text(json.dumps(dict(backend='Python/matplotlib',figures=QA,
        minimum_font_points=7,all_requested_coordinates_shown=True,missing_data_exclusions=0,
        visual_review='PENDING'),indent=2),encoding='utf-8')
    print('FIGURE_EXPORTS_COMPLETE',len(QA),'figures x PDF/SVG/600dpiPNG')


if __name__=='__main__':main()
