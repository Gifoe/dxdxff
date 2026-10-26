"""Verify extracted public bytes,numerics,legacy retention andpublication files."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
import io_contract as contract
from release_check import check,COUNTS
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--write',action='store_true');a=parser.parse_args()
    payload=check(a.package,a.sha256)
    for n,b in payload.items():assert (HERE/'results'/n).read_bytes()==b,n
    tables={n[:-4]:list(csv.DictReader(io.StringIO(b.decode('utf-8')))) for n,b in payload.items() if n.endswith('.csv')}
    primary=tables['PRIMARY'];secondary=tables['SECONDARY'];assert len(primary)==2 and len(secondary)==11
    for family,rows in [(2,primary),(11,secondary)]:
        for r in rows:
            if r['status']=='estimable':assert float(r['family_low'])<=float(r['ci_low'])<=float(r['ci_high'])<=float(r['family_high'])
    for r in tables['H_PERMUTATION_SUMMARY']:
        v=np.array([float(q['H']) for q in tables['H_PERMUTATION'] if q['scope']==r['scope'] and q['arm']==r['arm']]);assert len(v)==1000
        obs=float(r['observed_H']);tie=1e-10+1e-9*abs(obs)
        np.testing.assert_allclose([v.mean(),np.quantile(v,.025),np.quantile(v,.975),(1+(v>=obs-tie).sum())/1001,(1+(v<=obs+tie).sum())/1001],[float(r[k]) for k in ['null_mean','null_low','null_high','conditional_plus_one_upper','conditional_plus_one_lower']],rtol=1e-12,atol=1e-12)
    for r in tables['E1_H']:
        if r['arm']=='GLOBAL':assert float(r['mean'])==0 and r['identity']=='True'
    for source in ['ALL','hup','lzu','multicenter','pediatric']:
        for arm in ['GLOBAL','SOURCE']:
            for coordinate in ['ALL12']+__import__('core').COORDS:
                for group in ['ALL','EZ','NEZ']:
                    q=[float(r['mean']) for r in tables['E2_LOSS'] if r['scope']==source and r['coordinate']==coordinate and r['arm']==arm and r['group']==group]
                    assert len(q)==4 and max(q)==min(q)
    for r in tables['COMPOSITION_SLOPE']:
        if float(r['fraction'])==1:assert r['status']=='not_estimable' and r['mean']==''
    import pymupdf
    from PIL import Image
    pdf=pymupdf.open(HERE/'figures/pr_ez_composition.pdf');assert len(pdf)==1;page=pdf[0];text=page.get_text()
    assert all(s in text for s in ['EZ / NEZ','Reference-size sensitivity','Residual EZ association','Within-query composition']) and 'SYNTHETIC' not in text
    for f in page.get_fonts():assert len(pdf.extract_font(f[0])[3])>0
    for b in page.get_text('blocks'):assert b[0]>=-1 and b[1]>=-1 and b[2]<=page.rect.width+1 and b[3]<=page.rect.height+1
    image=Image.open(HERE/'figures/pr_ez_composition.png');image.load();assert image.width>7000 and image.height>5000 and abs(image.info['dpi'][0]-600)<.1
    assert '<text' in (HERE/'figures/pr_ez_composition.svg').read_text(encoding='utf-8')
    proof=dict(status='PASS',source_commit=json.loads(payload['MANIFEST.json'])['source_commit'],package_sha256=a.sha256,anonymous_payloads=len(contract.PAYLOADS),scientific_files=len(contract.SCIENCE),patients=80,channels=7635,primary_slots=2,secondary_slots=11,table_rows=COUNTS,legacy_bytes_unchanged=True,figure_dimensions=list(image.size),figure_dpi=600,PDF_embedded_fonts=True,SVG_editable_text=True,figures={p.name:contract.sha(p) for p in sorted((HERE/'figures').iterdir())})
    target=HERE/'RELEASE_VERIFICATION.json'
    if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==proof
    if a.write:
        assert not target.exists();target.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
