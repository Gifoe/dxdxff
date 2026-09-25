import pandas as pd
import pytest
import numpy as np

from neuroez_c.p2_v3_conservative_fusion import require_locked_weights
from neuroez_c.p2_v3_fusion_protocol import canonicalize_p2_fusion_ledger


def test_label_semantics_and_ez_score_conversion():
    raw = pd.DataFrame({"subject_id": ["hup:s"], "center": ["hup"], "fold_idx": [1], "channel_name": ["A"], "true_ez": [1], "score_ez": [.8]})
    out = canonicalize_p2_fusion_ledger(raw, split_role="validation")
    assert out.label_nez.iloc[0] == 0 and out.label_ez.iloc[0] == 1 and np.isclose(out.score_nez.iloc[0], .2)


def test_primary_nonlocked_weight_fails_but_diagnostic_is_marked():
    with pytest.raises(ValueError, match="locked"):
        require_locked_weights(.8, .2)
    assert require_locked_weights(.8, .2, diagnostic_allow_nonlocked_weights=True) == "DIAGNOSTIC_WEIGHT_ABLATION_NOT_PRIMARY"
