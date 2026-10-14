"""Paired unique-ID cluster bootstrap, retaining repeated fold appearances."""
from runtime_support import original_modules

def contrasts(c,frame,metrics):
    _,original=original_modules(); rows=[]
    for right in ['A0','A2_identity','A2_shuffle','A2_static']:
        rows.extend(dict(comparison='A2 minus '+right,**r) for r in original.compare(c,frame[frame.arm=='A2'],frame[frame.arm==right],metrics))
    return rows
