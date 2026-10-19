"""Configure a freshly initialized frozen-parent candidate without consuming RNG."""
import copy
from torch import nn

ARMS=['FULL','WO_EARLY_FILM','WO_LATE_GATE','WO_BOTH_GUIDANCE','WO_RAW_PRE_REFERENCE','EARLY_FILM_MISMATCH']
PARAMETERS=dict(FULL=11553,WO_EARLY_FILM=10505,WO_LATE_GATE=10505,WO_BOTH_GUIDANCE=9457,WO_RAW_PRE_REFERENCE=11553,EARLY_FILM_MISMATCH=11553)

def configure(m,arm):
    assert arm in ARMS and m.feature_pr and m.raw_pr and m.raw_context and m.feature_context and not m.interaction
    assert m.early and m.late and m.pre_reference
    m.core_arm=arm;m.arm='EARLY_FILM_MISMATCH' if arm=='EARLY_FILM_MISMATCH' else 'FULL_F0'
    if arm in ['WO_EARLY_FILM','WO_BOTH_GUIDANCE']:
        m.cnn=nn.Sequential(*copy.deepcopy(list(m.cnn.stem.children())),copy.deepcopy(m.cnn.pool));m.early=False
    if arm in ['WO_LATE_GATE','WO_BOTH_GUIDANCE']:
        del m.gate;m.late=False
    if arm=='WO_RAW_PRE_REFERENCE':m.pre_reference=False
    return m
