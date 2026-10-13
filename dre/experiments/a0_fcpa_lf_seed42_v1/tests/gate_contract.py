"""Pure synthetic Stage A gate admission, no scientific inputs or models."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit_source import decide
assert decide(.95,[True]*4,True,True,True)
assert not decide(.94999,[True]*4,True,True,True)
assert not decide(.99,[True,True,False,True],True,True,True)
assert not decide(1,[True]*4,False,True,True)
assert not decide(1,[True]*4,True,False,True)
assert not decide(1,[True]*4,True,True,False)
print('FCPA_LF_SYNTHETIC_GATE_TESTS_PASS')
