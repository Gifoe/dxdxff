"""Publication figure from authenticated aggregate CSV only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from report_public import table,select,num
HERE=Path(__file__).resolve().parent
COLORS={'GLOBAL':'#777777','SOURCE':'#8D6DA8','CENTER':'#24958B','SCALE':'#D99138','FULL':'#2364AA'}
def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    bg=table('BACKGROUND');ct=table('CONTRASTS');an=table('ANOMALY')
    fig,axs=plt.subplots(2,2,figsize=(12.8,8.6),layout='constrained');ax=axs[0,0];ax.axis('off')
    ax.text(0,1.04,'A  Disjoint-channel reference transfer',fontweight='bold',fontsize=11,transform=ax.transAxes)
    boxes=[(.02,.62,.38,.19,'Reference channels A\nUnlabeled mean and SD'),(.59,.62,.38,.19,'Query channels B\nEvaluate frozen forecast'),(.12,.21,.77,.2,'Exchange A and B; repeat 100 fixed splits\nAll 80 patients and 7,635 channels retained')]
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='#F1F5FA',edgecolor='#6A849C',transform=ax.transAxes));ax.text(x+w/2,y+h/2,t,ha='center',va='center',transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch((.42,.715),(.57,.715),arrowstyle='->',mutation_scale=15,color='#2364AA',transform=ax.transAxes))
    ax.text(.5,.50,'Gaussian marginal probes in common TRAIN units',ha='center',transform=ax.transAxes,fontsize=8)
    ax.text(.02,.04,'Design schematic; no simulated data shown.',transform=ax.transAxes,color='#555555',fontsize=8)
    ax=axs[0,1];arms=['GLOBAL','SOURCE','CENTER','SCALE','FULL']
    for i,a in enumerate(arms):
        r=select(bg,scope='ALL',coordinate='ALL12',arm=a);v,l,h=map(float,[r['mean'],r['ci_low'],r['ci_high']])
        ax.errorbar(v,i,xerr=[[max(0,v-l)],[max(0,h-v)]],fmt='o',color=COLORS[a],capsize=3,markersize=6)
    ax.set_yticks(range(5),['Global','Same source','Center only','Scale only','Full PR']);ax.invert_yaxis();ax.set_xlabel('Held-out CRPS (lower is better)');ax.set_title('B  Background prediction',loc='left',fontweight='bold',fontsize=11);ax.grid(axis='x',alpha=.2)
    ax.text(.02,-.25,'Equal-patient means; nominal 95% patient-bootstrap intervals.',fontsize=8,transform=ax.transAxes)
    ax=axs[1,0];contrasts=['GLOBAL-CENTER','GLOBAL-SCALE','CENTER-FULL','SCALE-FULL','SOURCE-FULL']
    for i,c in enumerate(contrasts):
        r=select(ct,scope='ALL',coordinate='ALL12',contrast=c);v,l,h=map(float,[r['mean'],r['family_low'],r['family_high']])
        ax.plot([l,h],[i,i],color='#2364AA',linewidth=1.5);ax.plot(v,i,'o',color='#2364AA')
    ax.axvline(0,color='#666666',linestyle='--',linewidth=1);ax.set_yticks(range(5),['Global − Center','Global − Scale','Center − Full','Scale − Full','Source − Full']);ax.invert_yaxis();ax.set_xlabel('CRPS reduction (positive favors right method)');ax.set_title('C  Contribution of location and scale',loc='left',fontweight='bold',fontsize=11);ax.grid(axis='x',alpha=.2)
    ax.text(.02,-.25,'Six-family Bonferroni intervals; same paired patients.',fontsize=8,transform=ax.transAxes)
    ax=axs[1,1]
    for i,e in enumerate(['NEZ','EZ']):
        r=select(an,scope='ALL',coordinate='ALL12',endpoint=e);v,l,h=map(float,[r['mean'],r['ci_low'],r['ci_high']]);ax.plot([i,i],[l,h],color=['#777777','#2364AA'][i],lw=1.8);ax.plot(i,v,'o',color=['#777777','#2364AA'][i],ms=7)
    ax.set_xticks([0,1],['NEZ','EZ']);ax.set_xlim(-.6,1.6);ax.set_ylabel('Mean absolute relative deviation');ax.set_title('D  Residual EZ association',loc='left',fontweight='bold',fontsize=11);ax.grid(axis='y',alpha=.2)
    gap=select(an,scope='ALL',coordinate='ALL12',endpoint='gap');ax.text(.02,-.25,f"EZ − NEZ = {num(gap['mean'])}\nAdjusted CI [{num(gap['family_low'])}, {num(gap['family_high'])}]",transform=ax.transAxes,fontsize=8)
    fig.suptitle('Patient-specific reference transfer before classification',fontsize=14,fontweight='bold')
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(out/('pr_reference_transfer.'+ext),dpi=600,bbox_inches='tight')
    plt.close(fig);print('PUBLICATION_FIGURE_PDF_SVG_PNG_WRITTEN')
if __name__=='__main__':main()
