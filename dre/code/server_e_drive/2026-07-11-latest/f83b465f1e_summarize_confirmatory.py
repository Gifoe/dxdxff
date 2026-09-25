#!/usr/bin/env python3
"""Write concise paper tables only from complete formal multi-seed outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from task1_confirmatory.reporting import write_report

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--output-root',required=True); a=p.parse_args(); root=Path(a.output_root); source=root/'statistics'/'multiseed_overall.csv'
    if not source.is_file():
        print(write_report(root)); return
    table=pd.read_csv(source); metrics=['patient_macro_f1','patient_ez_f1','patient_nez_f1','patient_ez_auprc','patient_ez_auroc','patient_ez_mrr']; selected=table[table.metric.isin(metrics)].copy(); selected['mean_std']=selected.apply(lambda row:f"{row['mean']:.4f} +/- {row['std']:.4f}",axis=1); wide=selected.pivot(index='model',columns='metric',values='mean_std').reset_index(); target=root/'reports'/'tables'; target.mkdir(parents=True,exist_ok=True); wide.to_csv(target/'task1_multiseed_main.csv',index=False); (target/'task1_multiseed_main.tex').write_text(wide.to_latex(index=False,escape=True),encoding='utf-8'); print(write_report(root)); print(target/'task1_multiseed_main.csv')
if __name__=='__main__': main()
