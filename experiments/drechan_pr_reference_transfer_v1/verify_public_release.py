"""Release-only checks from public aggregates and actual archive-bound source."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
from release_check import check
import contract

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    payload=check(args.package,args.sha256)
    for name,b in payload.items():assert (HERE/'results'/name).read_bytes()==b,name
    tables={n:list(csv.DictReader(io.StringIO(b.decode('utf-8')))) for n,b in payload.items() if n.endswith('.csv')}
    expected={'BACKGROUND.csv':325,'CONTRASTS.csv':390,'ANOMALY.csv':195,'SUPPORT.csv':130,'PERMUTATION.csv':5000,'PERMUTATION_SUMMARY.csv':5}
    assert {n:len(v) for n,v in tables.items()}==expected
    counts={'ALL':80,'hup':36,'lzu':21,'multicenter':15,'pediatric':8}
    family=primary=0
    for name,rows in tables.items():
        for r in rows:
            if 'patients' in r:assert int(r['patients'])==counts[r['scope']]
            if 'family_low' in r and r['family_low']:
                family+=1;assert r['scope']=='ALL' and r['coordinate']=='ALL12'
                assert float(r['family_low'])<=float(r['ci_low'])<=float(r['ci_high'])<=float(r['family_high'])
            if r.get('primary')=='True':primary+=1;assert r['contrast']=='GLOBAL-FULL' and r['scope']=='ALL' and r['coordinate']=='ALL12'
    assert family==6 and primary==1
    for r in tables['PERMUTATION_SUMMARY.csv']:
        v=np.array([float(q['gap']) for q in tables['PERMUTATION.csv'] if q['scope']==r['scope']]);assert len(v)==1000
        expected_null=[v.mean(),np.quantile(v,.025),np.quantile(v,.975),(1+(v>=float(r['observed'])).sum())/1001]
        np.testing.assert_allclose(expected_null,[float(r[k]) for k in ['null_mean','null_low','null_high','conditional_plus_one_upper']],atol=1e-12,rtol=1e-12)
    import pymupdf as fitz
    from PIL import Image
    pdf=fitz.open(HERE/'figures/pr_reference_transfer.pdf');assert len(pdf)==1
    page=pdf[0];assert all(t in page.get_text() for t in ['Background prediction','Residual EZ association','0.958464'])
    for font in page.get_fonts():assert len(pdf.extract_font(font[0])[3])>0,'Unembedded font'
    for block in page.get_text('blocks'):
        assert block[0]>=-1 and block[1]>=-1 and block[2]<=page.rect.width+1 and block[3]<=page.rect.height+1,'Clipped PDF text'
    im=Image.open(HERE/'figures/pr_reference_transfer.png');assert im.width>7000 and im.height>5000 and abs(im.info['dpi'][0]-600)<.1
    svg=(HERE/'figures/pr_reference_transfer.svg').read_text(encoding='utf-8');assert '<text' in svg
    result=dict(status='PASS',source_commit=json.loads(payload['MANIFEST.json'])['source_commit'],package_sha256=args.sha256,anonymous_payloads=12,scientific_files=8,parent_files=2,patients=80,channels=7635,primary_tests=1,secondary_family_tests=6,public_rows=expected,figure_dimensions=list(im.size),figure_dpi=600,PDF_embedded_fonts=True,SVG_editable_text=True,figures={p.name:contract.sha(p) for p in sorted((HERE/'figures').iterdir())})
    prior=HERE/'RELEASE_VERIFICATION.json'
    if prior.exists():assert json.loads(prior.read_text(encoding='utf-8'))==result,'Release metadata or figure bytes changed'
    if args.write:
        p=HERE/'RELEASE_VERIFICATION.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
