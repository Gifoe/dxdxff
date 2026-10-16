"""Executed-only quantitative figures; frozen-CNN screens and final CNN kept apart."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False,'axes.linewidth':.65})

LABELS={'B0_ORIGINAL_POOL_MACRO_LOSS':'B0: original pool/loss',
        'B1_ORIGINAL_POOL_EZ_LOSS':'B1: EZ-directed loss',
        'S1_SELECTED_POOL_EZ_LOSS':'S1: selected pooling',
        'A1_SELECTED_POOL_ATTENTION_EZ_LOSS':'A1: latent attention',
        'SELECTED_POLICY':'Development-selected'}


def export(fig,stem):
    fig.savefig(str(stem)+'.svg',facecolor='white')
    fig.savefig(str(stem)+'.pdf',facecolor='white')
    fig.savefig(str(stem)+'.png',dpi=600,facecolor='white')
    fig.savefig(str(stem)+'.tiff',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


def factorial(root,out):
    data=pd.read_csv(root/'FACTORIAL_ASSESSMENT_RESULTS.csv',dtype={'recipe':str})
    chosen=pd.read_csv(root/'OWN_FIT_POOL_SELECTION.csv',dtype={'recipe':str})
    data['recipe']=data.recipe.str.zfill(4);chosen['recipe']=chosen.recipe.str.zfill(4)
    recipes=sorted(data.recipe.unique());table=data.pivot(index='recipe',columns='fold',values='f1').loc[recipes]
    fig=plt.figure(figsize=(183/25.4,140/25.4),layout='constrained')
    grid=fig.add_gridspec(1,2,width_ratios=[1.35,1.]);ax=fig.add_subplot(grid[0,0]);right=grid[0,1].subgridspec(2,1,height_ratios=[1,1])
    image=ax.imshow(table.to_numpy(),aspect='auto',cmap='Blues',vmin=0,vmax=max(.65,float(table.to_numpy().max())))
    ax.set_xticks(range(5),['Fold '+str(i) for i in range(1,6)])
    ax.set_yticks(range(16),[r+'  '+('original' if r=='1111' else '') for r in recipes])
    ax.set_ylabel('Pre-reference / stages / seizure SD / patient Z')
    ax.set_title('a  Complete factorial, frozen encoders',loc='left',weight='bold',fontsize=8)
    for row in chosen.itertuples():
        j=recipes.index(row.recipe);ax.add_patch(Rectangle((row.fold-1-.46,j-.46),.92,.92,fill=False,edgecolor='#D58A32',linewidth=1.2))
    for j in range(16):
        for i in range(5):ax.text(i,j,f'{table.iloc[j,i]:.2f}',ha='center',va='center',fontsize=5.5,
                                  color='white' if table.iloc[j,i]>.45 else '#222222')
    fig.colorbar(image,ax=ax,fraction=.04,pad=.025,label='Patient-average EZ-F1')
    ax=fig.add_subplot(right[0]);counts=[chosen.recipe.str[i].eq('1').sum() for i in range(4)]
    ax.barh(range(4),counts,color='#7CA5BF',height=.6)
    ax.set_yticks(range(4),['Pre-reference','Two phases','Seizure SD','Patient Z']);ax.invert_yaxis();ax.set_xlim(0,5.5)
    ax.set_xticks(range(6));ax.set_xlabel('Selected own-FIT recipes (of 5)')
    ax.set_title('b  Fold-local component selection',loc='left',weight='bold',fontsize=8)
    for i,n in enumerate(counts):ax.text(n+.1,i,str(n),va='center',fontsize=7)
    ax=fig.add_subplot(right[1]);effect=pd.read_csv(root/'FACTOR_CONDITIONAL_EFFECTS.csv')
    for j,metric in enumerate(['f1','auroc']):
        q=effect[effect.metric==metric].groupby('factor').paired_delta.agg(['mean','min','max']).reindex(range(4))
        y=np.arange(4)+(j-.5)*.18
        ax.errorbar(q['mean'],y,xerr=np.vstack([q['mean']-q['min'],q['max']-q['mean']]),fmt='o',ms=3,
                    color=['#356B8B','#C48A45'][j],capsize=2,label=['EZ-F1','AUROC'][j])
    ax.axvline(0,color='#777777',lw=.7,ls=':');ax.set_yticks(range(4),['Pre-reference','Two phases','Seizure SD','Patient Z']);ax.invert_yaxis()
    ax.set_xlabel('On − off across 8 matched contexts')
    ax.set_title('c  Context dependence',loc='left',weight='bold',fontsize=8);ax.legend(fontsize=6,loc='best')
    export(fig,out/'factorial_screen')
    return dict(name='factorial_screen',width_mm=183,height_mm=140,archetype='quantitative_grid',
        source_data=['FACTORIAL_ASSESSMENT_RESULTS.csv','OWN_FIT_POOL_SELECTION.csv','FACTOR_CONDITIONAL_EFFECTS.csv'],
        n='255 patient-fold assessment appearances,79 unique patients,15 assessment encoder paths;5 additional choice-only paths are not plotted as assessment',
        intervals='Panel c: min/max over8 matched factor contexts, NOT confidence intervals',
        selection_data='All16 assessments used for own-FIT recipe choice; not unbiased final CNN evidence',
        exclusions='No factorial combinations or requested metrics omitted; distinct panels show EZ-F1 and AUROC, other metrics in source tables')


def final(root,out):
    absolute=pd.read_csv(root/'ABSOLUTE_INTERVALS.csv');paired=pd.read_csv(root/'PAIRED_INTERVALS.csv')
    models=list(LABELS);fig,axes=plt.subplots(1,3,figsize=(183/25.4,83/25.4),layout='constrained',sharey=True)
    for j,(metric,title) in enumerate([('f1','EZ-F1'),('auroc','AUROC')]):
        ax=axes[j];q=absolute[absolute.metric==metric].set_index('model').loc[models]
        colors=['#A7ADB4','#7FA6BC','#5C899C','#315F78','#BC8541']
        for i,model in enumerate(models):
            r=q.loc[model];ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=colors[i],capsize=2,ms=4)
        if metric=='f1':
            ax.axvline(.65,color='#A45E4B',lw=.8,ls='--')
            ax.text(.645,.03,'Target 0.65',rotation=90,ha='right',va='bottom',color='#A45E4B',
                    fontsize=6,transform=ax.get_xaxis_transform())
        ax.set_title(chr(97+j)+'  Patient-average '+title,fontsize=8,weight='bold',loc='left');ax.set_xlabel(title)
    ax=axes[2];q=paired[(paired.right==models[0])&(paired.metric=='f1')].set_index('left')
    for i,model in enumerate(models[1:],1):
        r=q.loc[model];ax.errorbar(r.delta,i,xerr=[[r.delta-r.low],[r.high-r.delta]],fmt='o',color='#356B8B',capsize=2,ms=4)
    ax.axvline(0,color='#777777',lw=.8,ls=':');ax.set_xlabel('ΔEZ-F1 versus B0')
    ax.set_title('c  Paired patient contrasts',fontsize=8,weight='bold',loc='left')
    axes[0].set_yticks(range(5),[LABELS[m] for m in models]);axes[0].invert_yaxis()
    export(fig,out/'end_to_end_results')
    return dict(name='end_to_end_results',width_mm=183,height_mm=83,archetype='quantitative_grid',
        source_data=['ABSOLUTE_INTERVALS.csv','PAIRED_INTERVALS.csv'],n='80 patients,7635 channels,5 held-out folds,seed42',
        center='Patient-equal mean',intervals='Unadjusted95% percentile intervals,10000 common paired patient bootstrap draws',
        scientific_status='Exploratory repeated-test, not independent confirmation',
        exclusions='Alternate threshold sensitivities are in tables, not substituted for primary model results')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).parent/'results');a=p.parse_args()
    out=a.results.parent/'figures';out.mkdir(exist_ok=True);contracts=[factorial(a.results,out)]
    if (a.results/'ABSOLUTE_INTERVALS.csv').exists():contracts.append(final(a.results,out))
    (out/'FIGURE_QA.json').write_text(json.dumps(dict(backend='Python_matplotlib',editable_text=True,dpi=600,
        formats=['SVG','PDF','PNG','LZW_TIFF'],contracts=contracts,rendered_inspection='PENDING'),indent=2),encoding='utf-8')
