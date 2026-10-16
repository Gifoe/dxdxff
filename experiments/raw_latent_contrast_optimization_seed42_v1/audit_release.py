"""Local anonymous-only release audit; manual figure review is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_json(value):
    forbidden={'patient','patient_id','patient_ids','channel','channels_ids','channel_ids','indices','archive_indices',
               'probability','logit','logit_nez','score_ez','raw','features','embedding','channel_embedding','password','secret','token'}
    if isinstance(value,dict):
        for key,item in value.items():
            assert key.lower() not in forbidden,('Identifying/individual JSON field',key)
            check_json(item)
    elif isinstance(value,list):
        for item in value:check_json(item)
    elif isinstance(value,str):
        assert not value.startswith(('C:\\','D:\\','E:\\','F:\\')), 'Private host path in public JSON'


def main(root,accept_review):
    results=root/'results';status=json.loads((results/'INDEPENDENT_RESULTS_VERIFICATION.json').read_text())
    assert status['status']=='PASS'
    scientific=json.loads((results/'SCIENTIFIC_SOURCE_AUDIT.json').read_text())
    assert scientific['status']=='PASS'
    for mapping in scientific['source'].values():
        for name,value in mapping.items():assert sha(root/name)==value,('Frozen scientific source changed',name)
    artifacts=[];csv_rows=0
    for path in sorted(results.iterdir()):
        assert path.is_file() and path.suffix in ['.csv','.json'],path.name
        if path.suffix=='.json':check_json(json.loads(path.read_text(encoding='utf-8')))
        else:
            frame=pd.read_csv(path);csv_rows+=len(frame)
            banned={'patient','patient_id','channel','channel_id','archive_index','archive_indices','logit','probability','y','prediction'}
            assert not (set(map(str.lower,frame.columns))&banned),path.name
        artifacts.append(dict(file=str(path.relative_to(root)).replace('\\','/'),sha256=sha(path),bytes=path.stat().st_size))
    assert accept_review,'Inspect both PNGs at final size before accepting visual review'
    figures=root/'figures';qa=json.loads((figures/'FIGURE_QA.json').read_text())
    assert len(qa['contracts'])==2
    inspected=[]
    for contract in qa['contracts']:
        stem=contract['name'];png=figures/(stem+'.png');svg=figures/(stem+'.svg')
        with Image.open(png) as image:
            width,height=image.size
            assert abs(width-contract['width_mm']/25.4*600)<=2 and abs(height-contract['height_mm']/25.4*600)<=2
        tree=ET.parse(svg);assert len(tree.findall('.//{http://www.w3.org/2000/svg}text'))>10
        for suffix in ['svg','pdf','png','tiff']:
            p=figures/(stem+'.'+suffix);assert p.stat().st_size>1000
            artifacts.append(dict(file=str(p.relative_to(root)).replace('\\','/'),sha256=sha(p),bytes=p.stat().st_size))
        inspected.append(dict(figure=stem,png_sha256=sha(png),width_pixels=width,height_pixels=height,
            final_size_visual_review='PASS',no_label_clipping=True,source_tables_checked=True,editable_SVG_text=True))
    qa['rendered_inspection']='PASS';qa['manual_review']=inspected
    (figures/'FIGURE_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    check_json(qa)
    report=root/'REPORT.md';assert report.exists()
    report_text=report.read_text(encoding='utf-8')
    assert 'TBD' not in report_text and 'PENDING' not in report_text and '探索性重复测试' in report_text
    manifest=dict(status='PASS',publication_scope='CODE_AND_ANONYMOUS_AGGREGATES_ONLY',
        result_files=len(list(results.iterdir())),aggregate_csv_rows=csv_rows,
        individual_rows_or_weights_transferred=False,report_sha256=sha(report),artifacts=artifacts,
        visual_reviews=inspected,code={p.name:sha(p) for p in sorted(root.glob('*.py'))})
    (root/'RELEASE_AUDIT.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('ANONYMOUS_RAW_RELEASE_AUDIT_PASS',len(artifacts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent)
    p.add_argument('--accept-visual-review',action='store_true');a=p.parse_args();main(a.root,a.accept_visual_review)
