"""Verify pushed branch identity and exact Git blob bytes, not only working files."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
BRANCH='codex/m8-6d-no-variance-ablation-baselines-seed42-v1'


def git(*args):
    return subprocess.run(['git',*args],cwd=HERE,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout


def main():
    repo=Path(git('rev-parse','--show-toplevel').decode().strip())
    head=git('rev-parse','HEAD').decode().strip()
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==head
    manifest=json.loads((HERE/'results/PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    source=json.loads((HERE/'results/SOURCE_PUSH_BEFORE_TRAIN.json').read_text(encoding='utf-8'))
    render=json.loads((HERE/'results/RENDER_MANIFEST.json').read_text(encoding='utf-8'))
    expected={HERE/'results'/n:h for n,h in manifest['files'].items()}
    expected.update({HERE/n:h for n,h in source['source'].items()})
    expected[HERE/'render_report.py']=render['renderer_sha256']
    mp=HERE/'results/PACKAGE_MANIFEST.json'
    expected[mp]=hashlib.sha256(mp.read_bytes()).hexdigest()
    for path,digest in expected.items():
        relative=path.relative_to(repo).as_posix()
        assert hashlib.sha256(git('show',head+':'+relative)).hexdigest()==digest,relative
    files=git('ls-tree','-r','--name-only',head,'--',HERE.relative_to(repo).as_posix()).decode().splitlines()
    forbidden={'.pt','.pkl','.pickle','.npz','.npy','.edf','.log','.err','.zip','.bin'}
    assert all(Path(name).suffix.lower() not in forbidden for name in files)
    print('SIXD_PUSHED_REMOTE_TIP_AND_EXACT_GIT_BLOBS_PASS')
    print(json.dumps(dict(branch=BRANCH,commit=head,verified_blobs=len(expected),private_artifact_extensions_present=False)))


if __name__=='__main__':main()
