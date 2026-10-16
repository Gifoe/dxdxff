"""All actual12 record-coordinate effects and four source strata."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;DATA=HERE/'results';OUT=HERE/'figures'
GROUPS=['early_partial_HG_80_125_slope','early_LL_slope','baseline_crossing_partial_HG_80_125',
        'baseline_crossing_LL','post_top20_partial_HG_80_125','post_top20_LL']
NAMES=['HG slope','LL slope','HG crossing','LL crossing','HG top20%','LL top20%']
SOURCES=['hup','lzu','multicenter','pediatric']


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((DATA/'INDEPENDENT_ATLAS_VERIFICATION.json').read_text())['status']=='PASS'
    table=pd.read_csv(DATA/'NEW12_PATIENT_ASSOCIATIONS.csv').set_index('coordinate')
    source=pd.read_csv(DATA/'NEW12_SOURCE_ASSOCIATIONS.csv');order=[];labels=[]
    for j,(group,name) in enumerate(zip(GROUPS,NAMES)):
        suffix=['observed_position','crossing_rate'] if j in [2,3] else ['seizure_mean','seizure_SD']
        display=['position*','rate'] if j in [2,3] else ['mean','SD']
        for a,b in zip(suffix,display):order.append(group+'__'+a);labels.append(name+' · '+b)
    rows=table.loc[order].reset_index();rows.to_csv(OUT/'SOURCE_PATIENT_EFFECTS.csv',index=False)
    source.to_csv(OUT/'SOURCE_CENTER_EFFECTS.csv',index=False)
    matrix=np.array([[source[(source.coordinate==name)&(source.center==c)].mean_effect.iloc[0] for c in SOURCES] for name in order])
    counts=np.array([[source[(source.coordinate==name)&(source.center==c)].finite_patients.iloc[0] for c in SOURCES] for name in order])
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
        'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'axes.linewidth':.65})
    fig,(a,b)=plt.subplots(1,2,figsize=(183/25.4,150/25.4),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
    for i,name in enumerate(order):
        r=table.loc[name];color='#74548B' if 'HG' in name else '#356B9B'
        a.errorbar(r.mean_effect,i,xerr=[[r.mean_effect-r.ci_low],[r.ci_high-r.mean_effect]],fmt='o',markersize=3.2,
            color=color,capsize=2,linewidth=1)
        a.text(.76,i,f'{int(r.patients)}',ha='center',va='center',fontsize=6.5)
    a.axvline(0,color='#777777',linewidth=.7,linestyle='--')
    a.set_yticks(range(12),labels);a.set_ylim(11.6,-.8);a.set_xlim(-.55,.82)
    a.set_xticks([-.4,-.2,0,.2,.4,.6]);a.grid(axis='x',alpha=.15,linewidth=.5)
    a.set(xlabel='Mean within-patient signed rank-biserial effect',title='EZ/NEZ association with patient uncertainty')
    a.text(.76,-.58,'n',ha='center',va='center',fontsize=6.5,fontweight='bold')
    im=b.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto',origin='upper')
    b.set_xticks(range(4),['HUP','LZU','Multi','Pediatric']);b.set_title('Source means and finite n')
    b.tick_params(axis='y',left=False);b.set_ylim(11.6,-.8)
    for i in range(12):
        for j in range(4):
            b.text(j,i,f'{matrix[i,j]:.2f}\nn={counts[i,j]}',ha='center',va='center',fontsize=6.5,
                color='white' if abs(matrix[i,j])>.65 else '#202020')
    cb=fig.colorbar(im,ax=b,fraction=.055,pad=.035);cb.set_ticks([-1,-.5,0,.5,1]);cb.set_label('Mean signed effect',fontsize=6.5)
    for label,ax in zip('ab',[a,b]):ax.text(-.10,1.025,label,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
    fig.savefig(OUT/'01_record_physiological_associations.svg')
    svg=OUT/'01_record_physiological_associations.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    fig.savefig(OUT/'01_record_physiological_associations.pdf')
    fig.savefig(OUT/'01_record_physiological_associations.png',dpi=600)
    fig.savefig(OUT/'01_record_physiological_associations.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;outside=[];not_drawn=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low-1e-12<=tick.get_loc()<=high+1e-12:not_drawn.update([id(tick.label1),id(tick.label2)])
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text() or id(text) in not_drawn:continue
        box=text.get_window_extent(renderer)
        if box.x0<bounds.x0-2 or box.y0<bounds.y0-2 or box.x1>bounds.x1+2 or box.y1>bounds.y1+2:outside.append(text.get_text())
    qa=dict(status='PASS' if not outside else 'FAIL',backend='Python matplotlib',width_mm=183,height_mm=150,raster_dpi=600,
        patient_coordinate_rows=12,source_coordinate_cells=48,primary_patients=51,finite_patient_range=[int(rows.patients.min()),int(rows.patients.max())],
        all_sources_including_n4_pediatric_shown=True,no_significance_stars=True,
        interval='10000 patient-bootstrap marginal95% percentile, exploratory',text_outside_canvas=outside,
        no_patient_identifiers_plotted=True,final_size_visual_inspection='PENDING')
    (HERE/'FIGURE_QA.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8');assert not outside,outside


if __name__=='__main__':main()
