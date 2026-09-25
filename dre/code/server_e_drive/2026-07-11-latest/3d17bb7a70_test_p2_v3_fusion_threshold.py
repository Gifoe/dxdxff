import numpy as np
import pandas as pd

from neuroez_c.p2_v3_fusion_reporting import THRESHOLD_CANDIDATES, select_validation_threshold


def _validation(labels=(0, 1), scores=(.2, .8)):
    return pd.DataFrame({"subject_id": ["s1", "s1"], "center": ["hup", "hup"], "outer_fold": [1, 1], "channel_name": ["A", "B"], "label_nez": labels, "label_ez": 1 - np.asarray(labels), "score": scores})


def test_threshold_grid_and_single_selection():
    threshold, search = select_validation_threshold(_validation(), score_nez_column="score")
    assert np.isclose(THRESHOLD_CANDIDATES[1] - THRESHOLD_CANDIDATES[0], .005)
    assert int(search.selected.sum()) == 1 and threshold == search.loc[search.selected, "threshold"].iloc[0]


def test_test_labels_and_scores_cannot_change_validation_threshold():
    validation = _validation()
    first, _ = select_validation_threshold(validation, score_nez_column="score")
    test = _validation(labels=(1, 0), scores=(.99, .01))
    del test  # The threshold function receives validation only.
    second, _ = select_validation_threshold(validation, score_nez_column="score")
    assert first == second
