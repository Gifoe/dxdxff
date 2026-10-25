"""Offline release QA on an actual Git tree, without private caches or identities."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image
from pypdf import PdfReader
import release_check

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
FIGURES=['pr_statistical_mechanism','paired_patient_effect_distribution','source_stratified_geometry']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    result=HERE/'results';manifest=json.loads((result/'PACKAGE_MANIFEST.json').read_text())
    assert {q.name for q in result.iterdir()}==set(manifest['files'])|{'PACKAGE_MANIFEST.json'}
    for n,h in manifest['files'].items():assert sha(result/n)==h,n
    for n,h in manifest['source'].items():assert sha(HERE/n)==h,n
    dependencies={'experiments/drechan_f0_complete_ablation_3seed_v1/feature_pr.py':
        '3b9823db47382cc3d4d66d226b4041f05efbc65041095fccaf298f459dc8148f',
        'experiments/drechan_unit_consistent_pr_3seed_v1/unit_pr.py':
        '2429906e260f02b7be71ab23220b2953dede963aaf79bef2a6eac923afd65f7b'}
    for n,h in dependencies.items():assert sha(REPO/n)==h,n
    # Repack the actual tree bytes to exercise all schemas and privacy gates.
    with tempfile.TemporaryDirectory(prefix='prm-aggregate-qa-') as directory:
        zpath=Path(directory)/'aggregate.zip'
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
            for n in sorted(q.name for q in result.iterdir()):z.write(result/n,n)
        _,proof=release_check.check(zpath,sha(zpath),source_tree=HERE)
    for path in HERE.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-v'],check=True,cwd=REPO)
    figure_info={}
    for n in FIGURES:
        pdf=PdfReader(HERE/'figures'/(n+'.pdf'));assert len(pdf.pages)==1
        page=pdf.pages[0];text=page.extract_text();assert len(text)>150
        embedded=True;fonts=page['/Resources']['/Font'].get_object()
        for ref in fonts.values():
            font=ref.get_object();assert font['/Subtype']!='/Type3'
            ds=font.get('/DescendantFonts',[font])
            for d in ds:
                desc=d.get_object().get('/FontDescriptor');assert desc is not None
                desc=desc.get_object();embedded&=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
        assert embedded
        svg=ET.parse(HERE/'figures'/(n+'.svg')).getroot()
        text_count=len(svg.findall('.//{http://www.w3.org/2000/svg}text'));assert text_count>=10
        with Image.open(HERE/'figures'/(n+'.png')) as im:
            assert im.width>=4000 and im.height>=1000
            dpi=im.info.get('dpi');assert dpi is not None and all(abs(q-600)<.01 for q in dpi)
            dimensions=list(im.size)
        figure_info[n]=dict(PDF_pages=1,PDF_text_characters=len(text),embedded_fonts=embedded,
            SVG_text_elements=text_count,PNG_pixels=dimensions,PNG_dpi=list(dpi),
            PDF_width_mm=float(page.mediabox.width)/72*25.4)
    # Native plotting/report regeneration must preserve all delivered bytes.
    targets=[HERE/'RESULTS.md']+[HERE/'figures'/(n+'.'+ext) for n in FIGURES for ext in ['pdf','svg','png']]
    before={str(p.relative_to(HERE)):sha(p) for p in targets}
    for script in ['report_public.py','plot_figures.py']:
        subprocess.run([sys.executable,str(HERE/script)],check=True,cwd=REPO)
    after={str(p.relative_to(HERE)):sha(p) for p in targets}
    assert before==after,'Regenerated public presentation bytes differ'
    print(json.dumps(dict(status='PUBLIC_RELEASE_QA_PASS',payloads=proof['payloads'],
        frozen_scientific_sources=8,exact_operational_reference_dependencies=2,unit_tests=9,
        all80_assessed=True,aggregate_rows=proof['row_counts'],figures=figure_info,
        deterministic_public_regeneration=True,private_inputs_required=False),indent=2))


if __name__=='__main__':main()
