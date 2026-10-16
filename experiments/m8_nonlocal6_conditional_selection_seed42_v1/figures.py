"""Actual aggregate-only six-group screen; no predictions or invented values."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;DATA=HERE/'results';OUT=HERE/'figures'
CONTEXTS=['PHYS','PHYS_OLD7','RAW','RAW_OLD7']
LABELS=['New physiology','Original7 + new','Frozen Raw + new','Raw + original7 + new']
COLORS=['#8B9DAF','#356B9B','#9B83AF','#684C83']
GROUPS=['early_partial_HG_80_125_slope','early_LL_slope','baseline_crossing_partial_HG_80_125',
        'baseline_crossing_LL','post_top20_partial_HG_80_125','post_top20_LL']


def clean_svg_export(path):
    """Mechanical whitespace normalization of matplotlib's vector export."""
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((DATA/'INDEPENDENT_EXPERIMENT_VERIFICATION.json').read_text())['status']=='PASS'
    main=pd.read_csv(DATA/'MAIN_RESULTS.csv').set_index(['context','arm']);fold=pd.read_csv(DATA/'FOLD_RESULTS.csv')
    freq=pd.read_csv(DATA/'SELECTION_FREQUENCY.csv');pairs=pd.read_csv(DATA/'PAIRED_INTERVALS.csv')
    curve=[]
    for c in CONTEXTS:
        for k in range(7):
            part=fold[(fold.context==c)&(fold.arm==f'K{k}')];assert len(part)==5
            curve.append(dict(context=c,k=k,patient_appearances=255,unique_patients=79,
                **{m:float(main.loc[(c,f'K{k}'),m]) for m in ['auroc','macro_f1']},
                **{m+'_fold_sd':float(part[m].std(ddof=1)) for m in ['auroc','macro_f1']}))
    curve=pd.DataFrame(curve);curve.to_csv(OUT/'SOURCE_CARDINALITY.csv',index=False)
    heat=freq[freq.scope=='final_paths'].copy();assert len(heat)==24
    heat.to_csv(OUT/'SOURCE_FINAL_FREQUENCY.csv',index=False)
    relevant=((pairs.context==pairs.reference_context)&(pairs.context!='PHYS')&(pairs.reference_arm=='BASE'))|((pairs.context=='PHYS')&(pairs.reference_context=='PHYS_OLD7')&(pairs.reference_arm=='BASE'))
    increment=pairs[relevant & pairs.metric.isin(['auroc','macro_f1'])].copy()
    assert len(increment)==8;increment.to_csv(OUT/'SOURCE_INCREMENT.csv',index=False)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'legend.fontsize':6.5,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.25})
    fig=plt.figure(figsize=(183/25.4,145/25.4),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1.07,1.])
    axes=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    for ax,metric,title in zip(axes[:2],['auroc','macro_f1'],['Patient AUROC','Patient Macro-F1']):
        for c,label,color in zip(CONTEXTS,LABELS,COLORS):
            g=curve[curve.context==c];x=g.k.to_numpy();y=g[metric].to_numpy();sd=g[metric+'_fold_sd'].to_numpy()
            ax.plot(x,y,marker='o',markersize=2.8,color=color,label=label)
            ax.fill_between(x,y-sd,y+sd,color=color,alpha=.09,linewidth=0)
        ax.set(xlabel='Number of new record-feature groups',ylabel=title,xticks=range(7))
        ax.grid(axis='y',alpha=.15,linewidth=.5)
    axes[0].legend(loc='lower right',handlelength=1.3,labelspacing=.35)
    matrix=np.array([[heat[(heat.context==c)&(heat.feature==g)].frequency.iloc[0] for g in GROUPS] for c in CONTEXTS])
    im=axes[2].imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    axes[2].set_xticks(range(6),['HG\nslope','LL\nslope','HG\ncrossing','LL\ncrossing','HG\ntop20%','LL\ntop20%'])
    axes[2].set_yticks(range(4),['New only','Original7','Raw','Raw + original7'])
    axes[2].set_title('Final-path selection frequency',loc='left',pad=8)
    axes[2].tick_params(length=0)
    for i in range(4):
        for j in range(6):axes[2].text(j,i,f'{matrix[i,j]:.0%}',ha='center',va='center',fontsize=6.5,color='white' if matrix[i,j]>.55 else '#202020')
    bar=fig.colorbar(im,ax=axes[2],fraction=.04,pad=.02);bar.set_ticks([0,.5,1]);bar.set_label('Fraction of five paths',fontsize=6.5)
    ax=axes[3]
    for metric,offset,color,marker,name in [('auroc',-.13,'#356B9B','o','AUROC'),('macro_f1',.13,'#B56F59','s','Macro-F1')]:
        for i,c in enumerate(CONTEXTS):
            r=increment[(increment.context==c)&(increment.metric==metric)].iloc[0]
            ax.errorbar(r.delta,i+offset,xerr=[[r.delta-r.ci_low],[r.ci_high-r.delta]],color=color,fmt=marker,markersize=3,
                capsize=2,linewidth=.9,label=name if i==0 else None)
    ax.axvline(0,color='#777777',linewidth=.7,linestyle='--')
    ax.set_yticks(range(4),['New only vs original7','+ new | original7','+ new | Raw','+ new | Raw + original7']);ax.invert_yaxis()
    ax.set(xlabel='Paired difference',title='Replacement / conditional increment')
    ax.legend(loc='lower left',handlelength=1.2);ax.grid(axis='x',alpha=.15,linewidth=.5)
    for label,ax in zip('abcd',axes):ax.text(-.12,1.055,label,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
    fig.savefig(OUT/'01_nonlocal6_cardinality_stability.svg')
    clean_svg_export(OUT/'01_nonlocal6_cardinality_stability.svg')
    fig.savefig(OUT/'01_nonlocal6_cardinality_stability.pdf')
    fig.savefig(OUT/'01_nonlocal6_cardinality_stability.png',dpi=600)
    fig.savefig(OUT/'01_nonlocal6_cardinality_stability.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;outside=[]
    not_drawn=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low-1e-12<=tick.get_loc()<=high+1e-12:not_drawn.update([id(tick.label1),id(tick.label2)])
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text() or id(text) in not_drawn:continue
        box=text.get_window_extent(renderer)
        if box.x0<bounds.x0-2 or box.y0<bounds.y0-2 or box.x1>bounds.x1+2 or box.y1>bounds.y1+2:outside.append(text.get_text())
    qa=dict(status='PASS' if not outside else 'FAIL',backend='Python matplotlib',width_mm=183,height_mm=145,raster_dpi=600,
        source_cardinality_rows=28,source_frequency_cells=24,source_interval_rows=8,text_outside_canvas=outside,
        curves_use_registered_appearance_equal_mean=True,shading='five-fold sample SD, not CI',
        intervals='10000 paired patient-ID cluster percentile95%, exploratory',private_identifiers_plotted=False,
        final_size_visual_inspection='PENDING')
    (HERE/'FIGURE_QA.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8');assert not outside,outside


if __name__=='__main__':main()
