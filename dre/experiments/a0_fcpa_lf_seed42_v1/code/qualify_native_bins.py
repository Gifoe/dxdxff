"""Metadata correction of a published diagnostic, preserving numerical values.

The residue total-minus-low is not exactly13..40 on all native Welch grids.
This qualifier does not change the stopped data gate or recompute any signal.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from audit_source import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 source=a.output/'LF_PSD_COMPARISON_QUALIFIED.csv';df=pd.read_csv(source)
 m=df.band=='mid13_40';df.loc[m,'band']='total_minus_low_residual_PROXY'
 df['native_bin_semantics']='Fixed2s Welch,nperseg=round(2fs),native-bin closed masks; edges not universally exact0.5Hz'
 df.loc[m,'native_bin_semantics']='Residual of separately masked integrals; NOT exact13to40 for nonaligned native grids; not a primary gate metric'
 df.to_csv(a.output/'LF_PSD_COMPARISON_QUALIFIED_V2.csv',index=False)
 (a.output/'NATIVE_BIN_QUALIFICATION.json').write_text(json.dumps(dict(
  original_diagnostic_sha256=sha(source),qualifier_sha256=sha(Path(__file__)),
  correction='Semantic qualification only; numerical values preserved',
  gate_changed=False,signals_recomputed=False,new_labels_or_model_outputs_read=False,
  dense_PSD_shape_or_exact_native13_40_proof=False),indent=2),encoding='utf-8')
 print('FCPA_LF_NATIVE_BIN_REPORT_QUALIFICATION_COMPLETE',flush=True)


if __name__=='__main__':main()
