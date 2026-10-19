"""Check remote branch tip and exact committed source/aggregate/figure bytes."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
BRANCH='codex/m8-6d-no-interaction-ablation-seed42-v1'

def git(*args):return subprocess.run(['git',*args],cwd=HERE,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
def main():
    repo=Path(git('rev-parse','--show-toplevel').decode().strip());head=git('rev-parse','HEAD').decode().strip()
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==head
    manifest=json.loads((HERE/'results/PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    pushed=json.loads((HERE/'results/SOURCE_PUSH_BEFORE_TRAIN.json').read_text(encoding='utf-8'))
    expected={HERE/'results'/n:h for n,h in manifest['files'].items()}
    expected.update({HERE/n:h for n,h in pushed['source'].items()})
    figures=json.loads((HERE/'results/FIGURE_MANIFEST.json').read_text(encoding='utf-8'))
    expected.update({HERE/'results/figures'/n:h for n,h in figures['files'].items()})
    expected[HERE/'render_figures.py']=figures['script_sha256']
    for n in ['PACKAGE_MANIFEST.json','FIGURE_MANIFEST.json']:
        p=HERE/'results'/n;expected[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    for path,h in expected.items():
        relative=path.relative_to(repo).as_posix();assert hashlib.sha256(git('show',head+':'+relative)).hexdigest()==h,relative
    files=git('ls-tree','-r','--name-only',head,'--',HERE.relative_to(repo).as_posix()).decode().splitlines()
    forbidden={'.pt','.pth','.pkl','.pickle','.npz','.npy','.edf','.log','.err','.zip','.bin'}
    assert all(Path(n).suffix.lower() not in forbidden for n in files)
    print('NI_PUSHED_REMOTE_TIP_AND_EXACT_GIT_BLOBS_PASS')
    print(json.dumps(dict(branch=BRANCH,commit=head,verified_blobs=len(expected),private_artifact_extensions_present=False)))

if __name__=='__main__':main()
