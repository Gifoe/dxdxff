from __future__ import annotations
import csv,json,math
from pathlib import Path

def _safe(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_safe(v) for v in x]
    return x
def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(_safe(obj),indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
__all__=["write_json","write_csv"]
