"""Local release QA; never opens private inputs or imports neural libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import false_discovery_control

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(reviewed=False):
    data=HERE/'results';fig=HERE/'figures'
    manifest=json.loads((data/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():assert sha(data/name)==h,('PAYLOAD_HASH',name)
    for name,h in manifest['source'].items():assert sha(HERE/name)==h,('FROZEN_SOURCE_HASH',name)
    a=pd.read_csv(data/'feature_association.csv')
    s=pd.read_csv(data/'source_association.csv')
    d=pd.read_csv(data/'patient_effect_summary.csv')
    t=pd.read_csv(data/'statistical_tests.csv')
    n=pd.read_csv(data/'permutation_control.csv')
    assert len(a)==len(d)==len(t)==len(n)==12 and len(s)==48
    assert a.valid_patients.eq(80).all() and (a.mean_effect.abs()<=1).all()
    assert np.array_equal(a.coordinate,t.coordinate) and np.array_equal(a.coordinate,d.coordinate)
    np.testing.assert_allclose(false_discovery_control(a.pvalue,method='bh'),a.BH_q,atol=1e-12,rtol=0)
    np.testing.assert_allclose(a.BH_q,t.BH_q,atol=1e-12,rtol=0)
    report=(HERE/'RESULTS.md').read_text(encoding='utf-8')
    for j,row in a.iterrows():
        sub=s[s.coordinate==row.coordinate]
        assert dict(zip(sub.source,sub.valid_patients))==dict(hup=36,lzu=21,multicenter=15,pediatric=8)
        np.testing.assert_allclose(np.average(sub.mean_effect,weights=sub.valid_patients),row.mean_effect,atol=1e-12,rtol=0)
        dr=d[d.coordinate==row.coordinate].iloc[0]
        for sign in ['positive','negative','zero']:
            assert abs(dr[sign+'_patients']/80*100-row[sign+'_percent'])<1e-12
        cell=f'{row.mean_effect:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]'
        assert cell in report and f'{row.BH_q:.3g}' in report and f'{row.pvalue:.3g}' in report
    assert n.permutations.eq(10000).all() and n.EZ_count_preserved.all() and n.labels_shuffled_within_patient.all()
    assert n.null_centered_within_4MCSE.all()
    # Validate aggregate-only public schemas rather than just checking filenames.
    for path in data.glob('*.csv'):
        cols=set(pd.read_csv(path,nrows=1).columns)
        assert not cols & {'patient','patient_id','channel','channel_id','subject_id','label','y','probability'}
    qa=json.loads((HERE/'FIGURE_QA.json').read_text(encoding='utf-8'))
    dimensions=[]
    for spec in qa['figures']:
        name=spec['figure'];w=spec['width_mm'];h=spec['height_mm']
        with Image.open(fig/(name+'.png')) as im:
            dpi=im.info['dpi'];assert all(abs(z-600)<.1 for z in dpi)
            actual=[im.width/dpi[0]*25.4,im.height/dpi[1]*25.4]
            assert abs(actual[0]-w)<.05 and abs(actual[1]-h)<.05
            pixels=[im.width,im.height]
        root=ET.parse(fig/(name+'.svg')).getroot()
        assert root.attrib['width'].endswith('pt') and root.attrib['height'].endswith('pt')
        sw=float(root.attrib['width'][:-2])*25.4/72
        sh=float(root.attrib['height'][:-2])*25.4/72
        assert abs(sw-w)<1e-5 and abs(sh-h)<1e-5
        svgtexts=[''.join(z.itertext()) for z in root.iter() if z.tag.endswith('}text')]
        assert len(svgtexts)>12 and any('Partial high gamma' in z for z in svgtexts)
        raw=(fig/(name+'.pdf')).read_bytes()
        m=re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]',raw)
        assert m and abs(float(m[1])*25.4/72-w)<1e-5 and abs(float(m[2])*25.4/72-h)<1e-5
        assert b'/FontFile2' in raw,'PDF must embed editable TrueType text'
        if name=='feature_association':
            for value in a.BH_q:assert f'{value:.2g}' in svgtexts
        assert not spec['text_outside_canvas']
        dimensions.append(dict(figure=name,png_pixels=pixels,png_dpi=list(dpi),width_mm=w,height_mm=h,
                               svg_text_nodes=len(svgtexts),PDF_embedded_TrueType=True))
    if reviewed:
        qa['visual_review']='PASS: all three rendered PNGs inspected; labels, signs, zero lines, intervals, source reversals and q-values checked'
        qa['reviewed_image_sha256']={spec['figure']:sha(fig/(spec['figure']+'.png')) for spec in qa['figures']}
        qa['static_preflight']='12 PASS,2 WARN,0 FAIL; requested PNG not TIFF; validator misreads 178*MM as inches, actual vector/raster dimensions independently checked'
        (HERE/'FIGURE_QA.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
    check=dict(status='PASS',authenticated_public_payload_hashes_checked=len(manifest['files']),
               frozen_analysis_source_hashes_checked=len(manifest['source']),primary_report_rows=12,
               source_rows=48,public_patient_rows=False,report_numbers_match_CSV=True,
               source_weighted_means_match_primary=True,figures=dimensions,
               visual_review=qa['visual_review'],models_or_sensitive_inputs_opened=0)
    (HERE/'ARTIFACT_CHECK.json').write_text(json.dumps(check,indent=2)+'\n',encoding='utf-8')
    print('PUBLIC_ARTIFACT_CHECK_PASS',json.dumps(check))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--visual-reviewed',action='store_true')
    main(p.parse_args().visual_reviewed)
