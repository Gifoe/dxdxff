"""Trusted-only reporting derivative; never releases individual endpoint rows."""
import csv,hashlib,json
from pathlib import Path
R=Path(r'D:\Codex_Research\experiments\drechan_pr_ranking_linear_residual_abc_v1_runtime')
def main():
    src=R/'private/patient_endpoints.csv'
    with src.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
    lookup={(r['model'],r['condition'],r['patient']):r for r in rr};out=[]
    metrics=['auroc','auprc','mrr','hit_at_1','hit_at_3','hit_at_5','macro_f1','f1','nez_f1','accuracy','ez_fraction_bias']
    for r in rr:
        if r['condition']!='FULL_PR':continue
        for arm in ['NO_PR','SCORE_Z']:
            ref=lookup[r['model'],arm,r['patient']]
            row={k:r[k] for k in ['model','fold','patient','source','channels']}
            pr='PR_LINEAR_RESIDUAL' if r['model']=='TIMECONV' else 'FULL_PR'
            row['contrast']=pr+'_MINUS_'+arm
            row.update({k:float(r[k])-float(ref[k]) for k in metrics});out.append(row)
    assert len(out)==480
    target=R/'private/patient_paired_differences.csv';assert not target.exists()
    with target.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    seal=dict(rows=480,patients=80,metrics=11,source_endpoints_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),paired_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),private_only=True)
    (R/'PRIVATE_PAIRED_REPORT_SEAL.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
    print('TRUSTED_PRIVATE_PAIRED_REPORT_COMPLETE',len(out))
if __name__=='__main__':main()
