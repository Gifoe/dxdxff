"""Execution-only restriction of this owned process; no clock/BIOS changes.

Logical CPUs16-31 only. Child processes inherit the restriction. This is a
bounded environment workaround to be admitted by exact math/checkpoint replay,
not a claim that the underlying system-wide native faults have been fixed.
"""
import ctypes
import runpy
import sys
from pathlib import Path

k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
if not k.SetProcessAffinityMask(k.GetCurrentProcess(),ctypes.c_size_t(0xffff0000)):
    raise ctypes.WinError(ctypes.get_last_error())
allowed={'admit_execution.py','runtime_parity.py','runtime_integrity.py','runtime_integrity_tagged.py','train.py','verify.py','assess.py','final_outer.py','package_public.py','test_contrast.py','test_model.py'}
target=sys.argv.pop(1)
if target=='pip':runpy.run_module('pip',run_name='__main__')
else:
    assert target in allowed
    path=Path(__file__).resolve().parent/target;sys.argv[0]=str(path)
    runpy.run_path(str(path),run_name='__main__')
