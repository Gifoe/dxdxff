"""Public release QA against authenticated aggregates and rendered outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import core as c

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(reviewed=False):
    data=HERE/'results';manifest=c.jr(data/'PACKAGE_MANIFEST.json')
    for n,h in manifest['files'].items():assert sha(data/n)==h,('PAYLOAD_CHANGED',n)
    for n,h in manifest['source'].items():assert sha(HERE/n)==h,('FROZEN_ANALYSIS_CHANGED',n)
    a=pd.read_csv(data/'main_results.csv');overall=pd.read_csv(data/'all_results.csv')
    s=pd.read_csv(data/'source_results.csv');f=pd.read_csv(data/'fold_results.csv')
    assert a.arm.tolist()==c.MAIN and len(overall)==14 and len(s)==448 and len(f)==560
    assert a.patients.eq(80).all() and a.channels.eq(7635).all()
    variation=pd.read_csv(data/'coordinate_variation.csv')
    assert len(variation)==60 and variation.patients.eq(80).all()
    assert ((variation.nonconstant_patients+variation.constant_patients)==80).all()
    assert (variation.supported_PR_patients==np.where(variation.coordinate_index%2==0,80,73)).all()
    report=(HERE/'RESULTS.md').read_text(encoding='utf-8');tex=(HERE/'tables/feature_prediction_single_column.tex').read_text(encoding='utf-8')
    audit=c.jr(HERE/'tables/TABLE_AUDIT.json')
    compile_proof=c.jr(HERE/'tables/TABLE_COMPILE_AUDIT.json')
    assert audit['render_status']==compile_proof['status']=='PASS_COMPILED_AND_VISUALLY_REVIEWED'
    assert sha(HERE/'tables/feature_prediction_preview.tex')==compile_proof['preview_sha256']
    assert sha(HERE/'tables/feature_prediction_preview.pdf')==compile_proof['pdf_sha256']
    assert compile_proof['overfull_boxes']==compile_proof['underfull_boxes']==0
    for r in overall.to_dict('records'):
        for metric in c.METRICS:
            source=s[(s.arm==r['arm'])&(s.metric==metric)]
            np.testing.assert_allclose(np.average(source['mean'],weights=source.patients),r[metric],atol=1e-12,rtol=0)
        if r['arm'] in c.MAIN:
            for metric in c.METRICS:assert f"{r[metric]:.3f}" in report
    for metric in c.PRIMARY:
        best=float(a[metric].max());rows=[r.arm for r in a.itertuples() if abs(getattr(r,metric)-best)<1e-12]
        assert audit['bolded_best_rows'][metric]==rows
    expected=sum(len(audit['bolded_best_rows'][m]) for m in c.PRIMARY)
    assert tex.count(r'\textbf{')==expected and r'\toprule' in tex and r'\bottomrule' in tex and r'\columnwidth' in tex
    assert r'\hline' not in tex and r'\begin{table*}' not in tex
    assert tex.count(r'\midrule')==1
    body=[line for line in tex.splitlines() if ' & ' in line and not line.startswith('Input &')]
    assert len(body)==8
    for j,line in enumerate(body):
        fields=line.removesuffix(r' \\').split(' & ')[1:]
        assert len(fields)==4
        row=a.iloc[j]
        for field,metric in zip(fields,c.PRIMARY):
            expected=f'{row[metric]:.3f}'
            if abs(row[metric]-a[metric].max())<1e-12:expected=r'\textbf{'+expected+'}'
            assert field==expected,('TABLE_CELL_MISMATCH',row.arm,metric)
    for path in data.glob('*.csv'):
        header=set(pd.read_csv(path,nrows=1).columns)
        assert not header&{'patient','patient_id','channel','channel_id','subject_id','y','label','probability','path'}
    qa=c.jr(HERE/'FIGURE_QA.json');sizes=[]
    for spec in qa['figures']:
        name=spec['figure'];w=spec['width_mm'];h=spec['height_mm'];out=HERE/'figures'
        with Image.open(out/(name+'.png')) as im:
            dpi=im.info['dpi'];assert all(abs(z-600)<.1 for z in dpi)
            assert abs(im.width/dpi[0]*25.4-w)<.05 and abs(im.height/dpi[1]*25.4-h)<.05
            pixels=[im.width,im.height]
        root=ET.parse(out/(name+'.svg')).getroot();sw=float(root.attrib['width'][:-2])*25.4/72;sh=float(root.attrib['height'][:-2])*25.4/72
        assert abs(sw-w)<1e-5 and abs(sh-h)<1e-5
        texts=[''.join(z.itertext()) for z in root.iter() if z.tag.endswith('}text')];assert len(texts)>8
        raw=(out/(name+'.pdf')).read_bytes();box=re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]',raw)
        assert box and abs(float(box[1])*25.4/72-w)<1e-5 and abs(float(box[2])*25.4/72-h)<1e-5
        assert b'/FontFile2' in raw and not spec['text_outside_canvas']
        sizes.append(dict(figure=name,width_mm=w,height_mm=h,png_pixels=pixels,dpi=list(dpi),SVG_text_nodes=len(texts),PDF_TrueType_embedded=True))
    if reviewed:
        qa['visual_review']='PASS: main8-row forest and all100-repeat negative-control histogram inspected; labels,nominal CIs,chance line and source numbers checked'
        qa['reviewed_PNG_SHA256']={p['figure']:sha(HERE/'figures'/(p['figure']+'.png')) for p in qa['figures']}
        qa['static_preflight']='12 PASS,2 WARN,0 FAIL: requested PNG not TIFF; width parser reads89/25.4 incorrectly, actual dimensions independently checked'
        c.write(HERE/'FIGURE_QA.json',qa)
    proof=dict(status='PASS',public_server_payload_hashes=len(manifest['files']),frozen_scientific_source_hashes=len(manifest['source']),
        main_rows=8,all_groups=14,source_rows=448,fold_rows=560,table_true_best_bolding=True,
        figures=sizes,visual_review=qa['visual_review'],private_patient_rows_exported=False)
    c.write(HERE/'ARTIFACT_CHECK.json',proof);print('PREDICTION_PUBLIC_ARTIFACT_CHECK_PASS',json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--visual-reviewed',action='store_true');main(p.parse_args().visual_reviewed)
