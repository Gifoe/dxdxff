"""Explicit post-failure numeric amendment, preserving all original sources."""
import qualify as q
from constant_safe import patient_z


def main():
    q.ROOT=q.ROOT/'constant_safe';q.PUB=q.ROOT/'public'
    q.ROOT.mkdir(exist_ok=True);q.PUB.mkdir(exist_ok=True)
    original=q.freeze
    def amended_freeze(path,value):
        if path.name=='INPUT_FREEZE_PRIVATE.json':
            value=dict(value,numeric_amendment={n:q.sha(q.HERE/n) for n in ['NUMERIC_AMENDMENT.json','constant_safe.py','run_constant_safe.py','test_constant_safe.py']})
        original(path,value)
    q.freeze=amended_freeze;q.patient_z=patient_z
    q.main()


if __name__=='__main__':main()
