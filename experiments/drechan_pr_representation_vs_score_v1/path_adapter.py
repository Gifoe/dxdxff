"""Process-local legacy path view; immutable parents, no root junctions."""
import builtins
import io
import os
import sys
import _io
from pathlib import Path

_INSTALLED=False

def install(mapping_file, writable):
    global _INSTALLED
    if _INSTALLED:return
    import json
    native_open=builtins.open
    with native_open(mapping_file,encoding='utf-8-sig') as f: rows=json.load(f)
    pairs=sorted([(str(q['Old']).rstrip('\\/'),str(q['Target']).rstrip('\\/')) for q in rows],key=lambda q:-len(q[0]))
    writable=[os.path.normcase(os.path.abspath(str(p))).rstrip('\\/') for p in writable]
    protected=os.path.normcase(r'D:\Codex_Research\experiments')
    original_stat=os.stat
    def translated(p):
        if isinstance(p,int):return p
        s=os.fsdecode(p);v=s.replace('/','\\');vl=v.lower()
        for a,b in pairs:
            if vl==a.lower() or vl.startswith(a.lower()+'\\'):
                r=b+v[len(a):]
                return os.fsencode(r) if isinstance(p,bytes) else r
        return p
    def may_write(p):
        t=translated(p)
        if isinstance(t,int):return t
        s=os.path.normcase(os.path.abspath(os.fsdecode(t))).rstrip('\\/')
        if (s==protected or s.startswith(protected+'\\')) and not any(s==a or s.startswith(a+'\\') for a in writable):
            raise PermissionError('Frozen parent write prohibited: '+os.fsdecode(p))
        return t
    def opening(fn):
        def call(file,mode='r',*args,**kwargs):
            target=may_write(file) if any(x in mode for x in 'wax+') else translated(file)
            return fn(target,mode,*args,**kwargs)
        return call
    builtins.open=opening(native_open);io.open=opening(io.open)
    old_code=_io.open_code
    _io.open_code=lambda p:old_code(translated(p))
    # importlib uses the built-in nt module directly, rather than os aliases.
    native=__import__('nt') if os.name=='nt' else os
    for name in ['stat','lstat','listdir','scandir','access','readlink']:
        fn=getattr(os,name,None)
        if fn is None:continue
        def wrapper(path,*a,_fn=fn,**kw):return _fn(translated(path),*a,**kw)
        setattr(os,name,wrapper);setattr(native,name,wrapper)
    # Python 3.13 captures scandir as a static class attribute at import time.
    import glob
    if hasattr(glob,'_StringGlobber'):
        glob._StringGlobber.scandir=staticmethod(os.scandir)
    for name in ['_getfinalpathname','_getfullpathname']:
        fn=getattr(native,name,None)
        if fn is None:continue
        def resolved(path,*a,_fn=fn,**kw):
            target=translated(path);out=_fn(target,*a,**kw)
            if os.fsdecode(target)!=os.fsdecode(path):
                clean=os.fsdecode(out).removeprefix('\\\\?\\')
                for old,new in pairs:
                    if clean.lower()==new.lower() or clean.lower().startswith(new.lower()+'\\'):
                        return old+clean[len(new):]
            return out
        setattr(native,name,resolved)
    for name in ['unlink','remove','rmdir','chmod','utime']:
        fn=getattr(os,name,None)
        if fn is None:continue
        def mutating(path,*a,_fn=fn,**kw):return _fn(may_write(path),*a,**kw)
        setattr(os,name,mutating);setattr(native,name,mutating)
    for name in ['rename','replace']:
        fn=getattr(os,name)
        def moving(a,b,*args,_fn=fn,**kw):return _fn(may_write(a),may_write(b),*args,**kw)
        setattr(os,name,moving);setattr(native,name,moving)
    mkdir=os.mkdir
    def guarded_mkdir(path,*a,**kw):
        try:target=may_write(path)
        except PermissionError:
            target=translated(path)
            # Existing-parent mkdir(exist_ok=True) remains a genuine no-op.
            if original_stat(target):raise FileExistsError(17,'File exists',os.fsdecode(path))
            raise
        return mkdir(target,*a,**kw)
    os.mkdir=guarded_mkdir;native.mkdir=guarded_mkdir
    old_os_open=os.open
    def descriptor_open(path,flags,*a,**kw):
        write_flags=os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND
        return old_os_open(may_write(path) if flags&write_flags else translated(path),flags,*a,**kw)
    os.open=descriptor_open;native.open=descriptor_open
    sys.dont_write_bytecode=True
    _INSTALLED=True

