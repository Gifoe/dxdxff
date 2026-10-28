"""Verify actual index/committed blobs, not merely working-copy payloads."""
from pathlib import Path
import subprocess,json,hashlib,argparse
REPO=Path(__file__).resolve().parents[3]
RANK='experiments/drechan_pr_ranking_robustness_v1/'
MAIN='experiments/drechan_pr_representation_vs_score_v1/'
def check(revision):
    def blob(path):return subprocess.check_output(['git','show',revision+':'+path],cwd=REPO)
    checked=0
    for prefix in [RANK+'results/public_abc/',RANK+'results/public_abcd/',MAIN+'results/public/',MAIN+'preview/results/public/']:
        manifest=json.loads(blob(prefix+'PUBLIC_MANIFEST.json'))
        for name,s in manifest['files'].items():
            if name.startswith('public/'):name=name[len('public/'):]
            b=blob(prefix+name);assert hashlib.sha256(b).hexdigest()==s['sha256'] and len(b)==s['bytes'],prefix+name;checked+=1
    root=json.loads(blob(RANK+'manifest.json'))
    for name,h in root['artifact_files_sha256'].items():assert hashlib.sha256(blob(RANK+name)).hexdigest()==h,name
    for path,commit,prefix,key in [
        (RANK+'results/public_abcd/manifest.json','3eac5b44073a9c18de54d74f7e2f32b1f08f2a0f',RANK,'source'),
        (RANK+'results/public_abcd/D_ADMISSION.json','4d45a5b02524e4c0eb2d01c45454e780e261bfb9',RANK,'source'),
        (MAIN+'results/public/SOURCE_LOCK_PUBLIC.json','eede217c80985bbb4f91f8df5580d4fcfbef0f6b',MAIN,'source')]:
        for name,h in json.loads(blob(path))[key].items():
            b=subprocess.check_output(['git','show',commit+':'+prefix+name],cwd=REPO);assert hashlib.sha256(b).hexdigest()==h,(commit,name)
    print('ACTUAL_GIT_BLOBS_ALL_PUBLIC_PAYLOADS_PASS',revision,checked,'artifact_files',len(root['artifact_files_sha256']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',default='');check(p.parse_args().revision)
