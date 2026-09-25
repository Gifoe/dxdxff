from __future__ import annotations
import hashlib,json,os
from pathlib import Path
def identity(path):
 p=Path(path); s=p.stat(); return {'path':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def signature(**value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def atomic_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,default=str),encoding='utf8');t.replace(p)
