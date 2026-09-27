"""Immutable private FIT model-selection manifest before any new target outcomes."""
from __future__ import annotations

import hashlib
import json

import teacher_core as tc
from fit_select import context_key


def selected_paths():
    out=[]
    for fold in range(1,6):
        keys=list(dict.fromkeys((int(r["selected_epoch"]),float(r["selected_threshold"])) for r in tc.selected_rows(fold)))
        for epoch,tau in keys:
            key=context_key({"fold":fold,"epoch":epoch,"tau":tau})
            out.append(tc.RUNTIME/"private"/tc.FIT_SELECTION_FOLDER/(key+".pkl"))
        for d in tc.DIMS:
            out.append(tc.RUNTIME/"private"/tc.META_FOLDER/f"fold_{fold}_d{d}.pkl")
        out.append(tc.RUNTIME/"private"/tc.META_FOLDER/f"fold_{fold}_prototype.pkl")
    return out


def hashes():
    out={}
    for path in selected_paths():
        if not path.is_file():raise RuntimeError(f"Unfinished FIT selection: {path}")
        out[str(path.relative_to(tc.RUNTIME)).replace("\\","/")]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def manifest_path():
    return tc.RUNTIME/"private"/"FIT_MODEL_SELECTION_FROZEN.json"


def verify(paths=None):
    path=manifest_path()
    if not path.is_file():raise RuntimeError("FIT_MODEL_SELECTION_NOT_FROZEN")
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj["lock_sha"]!=tc.LOCK_SHA or obj["model_files"]!=hashes():
        raise RuntimeError("FIT_MODEL_SELECTION_HASH_MISMATCH")
    return obj


def main():
    tc.preflight()
    path=manifest_path()
    if path.exists():
        obj=verify();print(f"[FIT_FREEZE_RESUME] files={len(obj['model_files'])}",flush=True);return
    target=tc.RUNTIME/"private"/"target"
    if target.exists() and any(target.rglob("*.pkl")):
        raise RuntimeError("Cannot first-freeze FIT models after target outcomes")
    model_files=hashes()
    obj={"lock_sha":tc.LOCK_SHA,"model_files":model_files,"all_fit_models_selected_before_target_outcomes":True,
         "target_cell_cache_count_at_freeze":0}
    tc.afc.write_json(path,obj)
    verify()
    print(f"[FIT_MODEL_SELECTION_FROZEN] files={len(model_files)}",flush=True)


if __name__=="__main__":main()
