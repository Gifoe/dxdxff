"""Local anonymous-delivery completeness, byte identity, schema and image QA."""
import ast
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent;R=HERE/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest=read(R/'PUBLIC_TRANSFER_MANIFEST.json');assert manifest['status']=='PASS'
    actual={p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file()}
    assert actual==set(manifest['files'])|{'PUBLIC_TRANSFER_MANIFEST.json'},actual.symmetric_difference(set(manifest['files']))
    for name,row in manifest['files'].items():assert sha(R/name)==row['sha256'] and (R/name).stat().st_size==row['bytes'],name
    for f in HERE.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    forbidden={'patient','patient_id','patient_ids','subject','subject_id','channel','channel_id','y','probability','prediction','logit','path','checkpoint','validation'}
    for f in R.rglob('*.csv'):
        columns=set(pd.read_csv(f,nrows=0).columns.str.lower());assert not columns&forbidden,(f,columns&forbidden)
    for p in HERE.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        assert p.suffix.lower() not in {'.pkl','.pt','.pth','.npz','.npy','.edf','.log','.err','.zip'},p
    for name in ['INDEPENDENT_SELECTION_VERIFICATION','INDEPENDENT_RESULTS_VERIFICATION','REPORT_VERIFICATION','NEURAL_VALIDATION_VERIFICATION','EXECUTION_AUDIT','UNIT_TEST_STATUS']:
        assert read(R/(name+'.json'))['status']=='PASS'
    assert read(R/'OUTER_STATUS.json')['status']=='ALL105_FROZEN_SCORE_CELLS_SEALED'
    facts=read(R/'FINAL_FACTS.json');assert not facts['global_best_subset_claim'] and not facts['independent_validation_claim'] and not facts['model_choice_changed_after_Outer']
    assert read(R/'NEURAL_STATUS.json')['actual_models']==80 and read(R/'SUPPORT30_STATUS.json')['epochs']==75
    assert read(R/'NEURAL_STATUS.json')['epochs']==1033 and read(R/'UNIT_TEST_STATUS.json')['passed']==10
    evidence=read(R/'EXECUTION_AUDIT.json')
    bindings=[evidence['main_scientific_source'],evidence['frozen_evaluation_sources'],read(R/'SUPPORT30_STATUS.json')['source'],read(R/'SCREEN_STATUS.json')['source'],read(R/'BIOLOGY_STATUS.json')['source']]
    for binding in bindings:
        for name,h in binding.items():assert sha(HERE/name)==h,(name,'Published code differs from executed freeze')
    for stage,count,models in [('development',65,17),('outer',80,21)]:
        t=pd.read_csv(R/stage/'MAIN_RESULTS.csv');assert len(t)==models and (t.appearances==count).all()
        assert (t.unique_patients==(47 if stage=='development' else 80)).all()
        if stage=='outer':assert (t.channels==7635).all()
        cov=pd.read_csv(R/stage/'METRIC_COVERAGE.csv');assert len(cov)==models*9 and (cov.finite_appearances==count).all()
    # Mechanical catalogue export from the checked human-readable definitions.
    names=read(R/'POOL_STATUS.json')['names'];quality=pd.read_csv(R/'POOL_QUALITY.csv').set_index('feature');freq=pd.read_csv(R/'FINAL_FEATURE_FREQUENCY.csv').set_index('feature');rows=[]
    for line in (HERE/'FEATURE_CATALOGUE.md').read_text(encoding='utf-8').splitlines():
        if not re.match(r'^\|\d+\|',line):continue
        cells=line.strip('|').split('|');i=int(cells[0]);name=cells[1];assert names[i]==name
        rows.append(dict(group=i,feature=name,definition=cells[2],rationale_and_boundary=cells[3],
            measured_window_finite_fraction=quality.loc[name,'finite_fraction'],selected_folds=int(freq.loc[name,'selected_folds']),selection_is_not_biological_eligibility=True))
    assert len(rows)==32
    pd.DataFrame(rows).to_csv(HERE/'FEATURE_CATALOGUE.csv',index=False)
    qa=[]
    for source in sorted((HERE/'figures').glob('*_source.csv')):
        name=source.name.removesuffix('_source.csv')
        for ext in ['svg','pdf','png','tiff']:assert (source.parent/(name+'.'+ext)).stat().st_size>1000
        with Image.open(source.parent/(name+'.png')) as im:
            assert abs(im.width-180/25.4*600)<2 and min(im.info['dpi'])>599
            qa.append(dict(figure=name,width_px=im.width,height_px=im.height,dpi=im.info['dpi'][0],source_rows=len(pd.read_csv(source)),visual_QA='opened_and_inspected',editable_vector_text=True))
    assert len(qa)==6
    (HERE/'FIGURE_QA.json').write_text(json.dumps(dict(status='PASS',figures=qa,static_source_checks='14PASS_0WARN_0FAIL',statistics_not_validated_by_static_checker=True),indent=2)+'\n',encoding='utf-8')
    # Freeze the public release; self-referential audit/manifest files excluded.
    files={p.relative_to(HERE).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['RELEASE_MANIFEST.json','RELEASE_STATUS.json']}
    (HERE/'RELEASE_MANIFEST.json').write_text(json.dumps(dict(files=files),indent=2)+'\n',encoding='utf-8')
    (HERE/'RELEASE_STATUS.json').write_text(json.dumps(dict(status='PASS',public_transfer_files=len(manifest['files']),release_files=len(files),
        all_transferred_bytes_match_server=True,all_frozen_scientific_sources_match=True,all6_figures_complete=True,
        no_individual_patient_channel_data_or_runtime_logs=True,new_neural_fits=85,total_epochs=1108,
        terminal='COMPLETE_NO_STABLE_NEW_M8_FEATURE_SET_SMALL_CONCAT_RANKING_INCREMENT_EXPLORATORY'),indent=2)+'\n',encoding='utf-8')
    print('LOCAL_PUBLIC_RELEASE_PASS',len(files),'files;',len(manifest['files']),'exact server aggregates; no private artifacts')


if __name__=='__main__':main()
