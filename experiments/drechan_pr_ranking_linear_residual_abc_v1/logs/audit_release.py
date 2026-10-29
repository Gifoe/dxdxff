"""Verify actual exported payload bytes, frozen math ASTs and figure formats."""
import ast,hashlib,json
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent.parent
def audit():
    pub=ROOT/'results/public';man=json.loads((pub/'PUBLIC_MANIFEST.json').read_text())
    assert set(p.name for p in pub.iterdir())==set(man['files'])|{'PUBLIC_MANIFEST.json'}
    for n,meta in man['files'].items():
        b=(pub/n).read_bytes();assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256']
    parent=ROOT.parent/'drechan_pr_ranking_robustness_v1/scripts'
    a=ast.parse((parent/'run_abc.py').read_text());b=ast.parse((ROOT/'scripts/core_abc.py').read_text())
    for name in ['endpoints','summary']:
        x=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name)
        y=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name)
        assert ast.dump(x)==ast.dump(y)
    assert (ROOT/'scripts/analysis_math.py').read_bytes()==(parent/'analysis_math.py').read_bytes()
    count=0
    for name in ['Figure_A_patient_gain_distributions','Figure_B_EZ_coverage']:
        with Image.open(ROOT/'figures'/(name+'.png')) as im:assert all(abs(q-600)<1 for q in im.info['dpi'])
        pdf=PdfReader(ROOT/'figures'/(name+'.pdf'));assert len(pdf.pages)==1 and len(pdf.pages[0].images)==0
        assert 'TimeConv' in pdf.pages[0].extract_text()
        assert '<text' in (ROOT/'figures'/(name+'.svg')).read_text(encoding='utf-8');count+=1
    print('ACTUAL_RELEASE_PAYLOAD_HASHES_MATH_AST_AND_VECTOR_600DPI_PASS',len(man['files']),count)
if __name__=='__main__':audit()
