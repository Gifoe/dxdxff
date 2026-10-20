"""Inherited bounded native-runtime workaround; restrict this owned process only."""
import ctypes
import runpy
import sys
from pathlib import Path

k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
if not k.SetProcessAffinityMask(k.GetCurrentProcess(),ctypes.c_size_t(0xffff0000)):
    raise ctypes.WinError(ctypes.get_last_error())
target=sys.argv.pop(1)
assert target in {'admit.py','test_preprocessing.py','train.py','verify.py','assess.py','package_public.py'}
path=Path(__file__).resolve().parent/target;sys.argv[0]=str(path);runpy.run_path(str(path),run_name='__main__')
