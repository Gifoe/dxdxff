"""Mechanical formatting of actual anonymous CSVs into companion Markdown."""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent;R=HERE/'results'
METRICS=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1']
HEADERS=['AUROC','AP','Macro-F1','EZ-F1','NEZ-F1','ACC','MRR','Top-1']


def table(frame):
    def text(v):
        if isinstance(v,float):return f'{v:.6f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','|'+'|'.join(['---']*len(frame.columns))+'|']+['| '+' | '.join(text(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def metric_table(frame):
    return frame[['model']+METRICS].rename(columns=dict(zip(METRICS,HEADERS)))


def pair_table(p,comparisons):
    rows=[]
    for a,b in comparisons:
        row={'Model − reference':a+' − '+b}
        for m,label in zip(METRICS,HEADERS):
            r=p[(p.model==a)&(p.reference==b)&(p.metric==m)].iloc[0]
            row[label]=f'{r.delta:+.6f} [{r.ci_low:+.6f}, {r.ci_high:+.6f}]'
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    assert json.loads((R/'INDEPENDENT_RESULTS_VERIFICATION.json').read_text())['status']=='PASS'
    outer=pd.read_csv(R/'outer/MAIN_RESULTS.csv');dev=pd.read_csv(R/'development/MAIN_RESULTS.csv');p=pd.read_csv(R/'outer/PAIRED_COMPARISONS.csv')
    sections=['# Complete anonymous result tables\n\nGenerated from the accompanying actual CSV files. All Outer rows are exploratory repeated-test, not independent confirmation. Values are equal-patient means; Top-1 is a fraction. Every main model covers80 patients/7635 channels. Development has65 patient-fold appearances/47 unique patients and is epoch/threshold-selection conditioned. No best model is selected here.']
    sections += ['## All21 Outer procedures/references\n\n'+table(metric_table(outer)), '## All17 new Development procedures\n\n'+table(metric_table(dev))]
    c=pd.read_csv(R/'CARDINALITY_CURVE.csv');a=c.pivot(index='k',columns='metric',values='mean').reset_index()
    sections += ['## Nested FIT cardinality curve\n\nEach k:255 assessment appearances/79 unique patients. Each group has mean+SD coordinates. Not neural Outer performance.\n\n'+table(a[['k']+METRICS].rename(columns=dict(zip(METRICS,HEADERS))))]
    for family in ['D1','M8','Concat']:
        pairs=[(family+'__SELECTED',family+'__'+b) for b in ['OLD7','BEST_SWAP','RANDOM_A','RANDOM_B']]
        sections += [f'## {family}: equal-k replacement and old7 reference\n\nPaired10000-draw unadjusted95% intervals; OLD7 is not equal k.\n\n'+table(pair_table(p,pairs))]
    rawpairs=[(f+'__SELECTED','Concat__'+r) for f in ['M8','Concat'] for r in ['RAW_ONLY','RAW_SUPPORT30']]
    sections += ['## Raw increment\n\n'+table(pair_table(p,rawpairs))]
    sd=pd.read_csv(R/'outer/FOLD_STABILITY.csv');g=sd.pivot(index='model',columns='metric',values='fold_sample_sd').reset_index()
    sections += ['## Fold sample SD for all procedures\n\nSD of five equal-fold patient means, not patient SD or a replacement for the main patient-weighted mean.\n\n'+table(g[['model']+METRICS].rename(columns=dict(zip(METRICS,HEADERS))))]
    s=pd.read_csv(R/'outer/SOURCE_RESULTS.csv');keep=['M8__SELECTED','M8__OLD7','Concat__SELECTED','Concat__OLD7','Concat__RAW_ONLY','Concat__RAW_SUPPORT30','D1__SELECTED','D1__OLD7']
    for center,h in s[s.model.isin(keep)].groupby('center'):
        sections += [f'## Source group {center}; n={int(h.unique_patients.iloc[0])}\n\n'+table(metric_table(h))]
    sections += ['## Frozen feature allocations\n\n'+table(pd.read_csv(R/'FROZEN_FEATURE_SELECTION.csv')[['outer_fold','arm','k','features']]),
        '## Access to full precision and uncertainty\n\nAll source-specific paired intervals, all9 endpoint coverage counts, fixed-score compensation diagnostics, absolute intervals and full fold/source results are in `results/`. The selected-M8 versus matched-old7 familywise intervals are in `outer/PAIRED_COMPARISONS.csv`. This table formatter does not refit, rescore, resample or select.']
    (HERE/'RESULT_TABLES.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8',newline='\n')
    print('ACTUAL_RESULT_TABLES_FORMATTED')


if __name__=='__main__':main()
