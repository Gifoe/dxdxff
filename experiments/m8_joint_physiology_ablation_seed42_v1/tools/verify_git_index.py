"""Ensure Git's staged bytes preserve scientific source/aggregate SHA256s."""
import hashlib
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]

def main():
    count=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ['.py','.json','.csv','.ps1']:continue
        rel=path.relative_to(REPO).as_posix()
        blob=subprocess.check_output(['git','show',':'+rel],cwd=REPO)
        assert hashlib.sha256(blob).digest()==hashlib.sha256(path.read_bytes()).digest(),rel
        count+=1
    print('JOINT_GIT_INDEX_BYTE_PRESERVATION_PASS',count)

if __name__=='__main__':main()
