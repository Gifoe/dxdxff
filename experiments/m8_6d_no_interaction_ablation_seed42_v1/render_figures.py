"""Plot only independently verified anonymous contrasts, without private imports."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent;PUB=HERE/'results'
LABELS={'WO_EARLY_FILM':'w/o early FiLM','WO_BOTH_GUIDANCE':'w/o both guidance','WO_BOTH_PR':'w/o both PR','WO_RAW_PRE_REFERENCE':'w/o Raw pre-reference',
    'EARLY_FILM_MISMATCH':'Early FiLM mismatch','RAW_ONLY':'Raw-only','PHYSIOLOGY_ONLY':'Physiology-only',
    'DROP_BETA':'Beta','DROP_LOW_GAMMA':'Low gamma','DROP_PARTIAL_HIGH_GAMMA':'Partial high gamma (80-125 Hz)','DROP_RMS':'RMS','DROP_LINE_LENGTH':'Line length','DROP_SPECTRAL_ENTROPY':'Spectral entropy (10-125 Hz)'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))['status']=='COMPLETE_VERIFIED_NI_14_GROUP_FIVE_FOLD_OUTER'
    assert json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))['status']=='PASS'
    lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));arms=lock['arms']
    with (PUB/'INTERVALS.csv').open(encoding='utf-8',newline='') as f:ci=list(csv.DictReader(f))
    dest=PUB/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    for name,selected,title in [('CORE_PAIRED_INTERVALS',arms[1:8],'Six-feature model without head interaction: mechanisms'),
        ('FEATURE_CONTRIBUTION',arms[8:],'Conditional contribution of the six physiological measurements')]:
        fig,axes=plt.subplots(1,3,figsize=(14,max(4.6,.42*len(selected)+1.7)),sharey=True)
        for ax,m,heading,letter in zip(axes,['macro_f1','auroc','auprc'],['Macro-F1','AUROC','AP'],['a','b','c']):
            rr=[next(r for r in ci if r['kind']=='paired' and r['contrast']=='FULL_minus_'+a and r['metric']==m) for a in selected]
            lo=np.array([float(r['low']) for r in rr]);hi=np.array([float(r['high']) for r in rr]);mean=np.array([float(r['mean']) for r in rr]);yy=np.arange(len(selected))[::-1]
            ax.hlines(yy,lo,hi,color='#666666',lw=1.7,zorder=2);ax.scatter(mean,yy,c='#0072B2',s=31,zorder=3)
            ax.axvline(0,color='#222222',ls='--',lw=.9);ax.grid(axis='x',color='#D9D9D9',lw=.55)
            lower=min(float(lo.min()),0);upper=max(float(hi.max()),0);pad=max((upper-lower)*.1,.002)
            ax.set_xlim(lower-pad,upper+pad);ax.set_title(f'({letter}) {heading}',fontweight='bold');ax.set_xlabel('Main minus ablation')
            ax.set_yticks(yy);ax.set_yticklabels([LABELS[a] for a in selected]);ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=10)
            for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
        fig.suptitle(title,fontsize=13,fontweight='bold',y=.98)
        fig.text(.5,.035,'Nominal 95% paired intervals; 80 patients, 10,000 draws. Exploratory repeated-test.',ha='center',fontsize=10,color='#444444')
        fig.subplots_adjust(left=.27,right=.985,top=.86,bottom=.2,wspace=.34)
        for ext in ['svg','pdf','png']:fig.savefig(dest/(name+'.'+ext),dpi=180)
        plt.close(fig)
    manifest=dict(status='GENERATED_PENDING_VISIBLE_QA',script_sha256=sha(Path(__file__)),source_sha256=sha(PUB/'INTERVALS.csv'),
        files={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
    (PUB/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('NI_TWO_ACTUAL_AGGREGATE_FIGURES_RENDERED')

if __name__=='__main__':main()
