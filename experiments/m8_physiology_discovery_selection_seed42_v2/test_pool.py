import numpy as np


def test_finite_reference_and_padding():
    from build_pool import finite_delta
    v = np.array([[[[2., np.nan], [4., 3.], [8., 7.], [999., 999.]]]], np.float32)
    mask = np.array([[[True, True, True, False]]])
    times = np.array([[-2., -1., 1., 2.]])
    d, s = finite_delta(v, mask, times)
    np.testing.assert_array_equal(d[0, 0, :3, 0], [-1., 1., 5.])
    assert d[0, 0, 3].sum() == 0
    assert np.isnan(d[0, 0, 0, 1])
    assert s.shape == (1, 4)


def test_nonconstant_group_not_based_on_label():
    from common import NAMES, METRICS
    assert len(NAMES) == 32 and len(set(NAMES)) == 32
    assert len(METRICS) == 9


def test_graph_alignment_axis_order():
    target = np.full((2, 4, 5, 7), np.nan)
    values = np.arange(5 * 3 * 7).reshape(5, 3, 7)
    ix = np.array([3, 0, 2])
    target[1, ix[:, None], np.arange(5)[None, :]] = values.transpose(1, 0, 2)
    for t in range(5):
        for c in range(3):
            np.testing.assert_array_equal(target[1, ix[c], t], values[t, c])


def test_lock_excludes_outer_selection():
    import json
    from pathlib import Path
    p = json.loads((Path(__file__).parent/'PROTOCOL_LOCK.json').read_text())
    assert p['selection']['cardinalities'] == list(range(1, 10))
    assert p['final_training']['no_post_development_rescue']
    assert p['scientific_status'].startswith('EXPLORATORY_REPEATED_TEST')
