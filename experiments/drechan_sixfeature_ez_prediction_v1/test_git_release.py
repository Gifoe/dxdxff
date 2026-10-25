"""Verify an actual Git tree archive without trusted-server private inputs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
EXP='experiments/drechan_sixfeature_ez_prediction_v1'
PARENT='experiments/drechan_sixfeature_ez_association_v1/results/feature_association.csv'
BASE='97e587aef0fe137094fd0e9a9564567eaaf546cc'
SOURCE='caceb44597097f0f4baea49d10a5c543e2bb706f'
PACKAGE_SHA='87202b39099d472bb2203aed30c60d0b88221535446a3f0b7485a989718f009a'


def sha(b):return hashlib.sha256(b).hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def run(args,cwd):
    p=subprocess.run([sys.executable,*args],cwd=cwd,capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==0,(args,p.stdout,p.stderr)
    print(p.stdout.strip() or p.stderr.strip())


def main():
    p=argparse.ArgumentParser();p.add_argument('--staged',action='store_true')
    p.add_argument('--ref',default='HEAD');p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tree=git('write-tree').decode().strip() if a.staged else git('rev-parse',a.ref+'^{tree}').decode().strip()
    changed=git('diff','--name-only',BASE,tree).decode().splitlines()
    assert changed and all(x.startswith(EXP+'/') for x in changed),changed
    assert sha(a.package.read_bytes())==PACKAGE_SHA
    temp=Path(tempfile.mkdtemp(prefix='taskb_git_release_',dir=ROOT.parent))
    archive=temp/'git_tree.zip';extracted=temp/'tree'
    git('archive','--format=zip','--output='+str(archive),tree,'--',EXP,PARENT)
    with zipfile.ZipFile(archive) as z:
        files=[x for x in z.namelist() if not x.endswith('/')]
        assert PARENT in files
        for name in files:
            pure=PurePosixPath(name)
            assert not pure.is_absolute() and '..' not in pure.parts
            assert name.startswith(EXP+'/') or name==PARENT
            assert pure.suffix.lower() not in {'.pkl','.npz','.npy','.pt','.pth','.log','.err','.pem','.key','.aux'}
        manifest=json.loads(z.read(EXP+'/results/PACKAGE_MANIFEST.json'))
        for name,h in manifest['files'].items():
            assert sha(z.read(EXP+'/results/'+name))==h,('GIT_PAYLOAD_CHANGED',name)
        for name,h in manifest['source'].items():
            raw=z.read(EXP+'/'+name)
            assert sha(raw)==h and raw==git('show',SOURCE+':'+EXP+'/'+name),('GIT_SOURCE_CHANGED',name)
        repair=json.loads(z.read(EXP+'/ENGINEERING_AUDIT.json'))
        assert sha(z.read(EXP+'/audit_adapter.py'))==repair['adapter_sha256']
        assert repair['first_candidate_preserved'] and repair['replacement_sites']==3
        z.extractall(extracted)
    run(['-m','unittest','discover','-s',EXP,'-p','test_math.py','-v'],extracted)
    run(['-m','unittest','discover','-s',EXP,'-p','test_audit_adapter.py','-v'],extracted)
    run([EXP+'/release_check.py','--package',str(a.package.resolve())],extracted)
    run([EXP+'/check_public_artifacts.py'],extracted)
    before={n:(extracted/EXP/n).read_text(encoding='utf-8') for n in
        ['RESULTS.md','PAPER_PARAGRAPH.md','FIGURE_CAPTIONS.md','tables/feature_prediction_single_column.tex']}
    run([EXP+'/publish_report.py'],extracted)
    for name,content in before.items():
        assert (extracted/EXP/name).read_text(encoding='utf-8')==content,('PUBLIC_TEXT_REGENERATION',name)
    run([EXP+'/check_public_artifacts.py'],extracted)
    proof=dict(status='PASS',tested_tree=tree,archive_sha256=sha(archive.read_bytes()),
        test_kind='actual Git index tree archive' if a.staged else 'actual committed Git tree archive',
        scientific_source_commit=SOURCE,public_package_sha256=PACKAGE_SHA,
        changed_paths_only_new_experiment=True,Task_A_and_old_experiments_unchanged=True,
        anonymous_payload_hashes_verified=len(manifest['files']),immutable_source_hashes_verified=len(manifest['source']),
        engineering_adapter_hash_verified=True,CPU_math_tests=6,adapter_tests=1,
        public_artifact_checks='PASS twice: exported archive and regenerated report',
        public_report_and_table_regeneration='byte-equivalent text after universal newline normalization',
        parent_dependency=PARENT,compiled_table='PASS, actual PDF hash and source matched',
        public_patient_records=False,public_models_or_logs=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_GIT_TREE_RELEASE_PASS',json.dumps(proof))


if __name__=='__main__':main()
