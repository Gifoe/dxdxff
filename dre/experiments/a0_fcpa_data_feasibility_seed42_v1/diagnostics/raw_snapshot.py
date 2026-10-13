"""Interim anonymous raw PSD summary, using already frozen audit definitions."""
import json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit import check,summary
from discover import ROOT,sha,write


def main():
 p,fr=check()
 with (ROOT/'RAW_PSD_PRIVATE.pkl').open('rb') as f:rows=pickle.load(f)
 out=[]
 total=np.concatenate([r['powers']['total'].ravel() for r in rows])
 for band in ['delta','theta','alpha','alpha_low','alpha_high','beta']:
  values=np.concatenate([r['powers'][band].ravel() for r in rows])
  out.append(dict(band=band,**summary(values,total,p)))
 with (ROOT/'ALIGNMENT_PRIVATE.pkl').open('rb') as f:alignment=pickle.load(f)
 report=dict(status='RAW_MEASURED_PROVENANCE_PENDING',bands=out,
   complete_feature_windows=sum(r['complete_feature_windows'] for r in alignment),
   invalid_feature_boundary_windows=sum(r['invalid_feature_boundary_windows'] for r in alignment),
   nominal59_invalid_windows=sum(r['nominal59_invalid_windows'] for r in alignment),
   raw_PSD_sha256=sha(ROOT/'RAW_PSD_PRIVATE.pkl'),snapshot_code_sha256=sha(Path(__file__)))
 write(ROOT/'public/INTERIM_RAW_BANDPOWER.json',report)
 print(json.dumps(report),flush=True)


if __name__=='__main__':main()
