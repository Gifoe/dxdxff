"""Documentary repository inventory, not a numerical result revalidation."""
import hashlib
import subprocess
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
if __name__=='__main__':
    files=subprocess.check_output(['rg','--files','-g','*REPORT*.md','-g','*FINDINGS*.md','-g','*AUDIT*.md','-g','!**/node_modules/**'],cwd=REPO,text=True).splitlines()
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines())
    rows=[]
    for name in sorted(files):
        p=REPO/name
        if HERE in p.parents:continue
        if p.relative_to(REPO).as_posix() not in tracked:continue
        value=p.read_bytes();text=value.decode('utf-8-sig',errors='replace')
        rows.append(dict(repository_path=p.relative_to(REPO).as_posix(),sha256=hashlib.sha256(value).hexdigest(),bytes=len(value),
            indexed_lines=len(text.splitlines()),scope='documentary_full_text_inventory_not_numerical_revalidation'))
    pd.DataFrame(rows).to_csv(HERE/'PROJECT_REPORT_INVENTORY.csv',index=False)
    print('PROJECT_REPORT_AUDIT_INVENTORY',len(rows))
