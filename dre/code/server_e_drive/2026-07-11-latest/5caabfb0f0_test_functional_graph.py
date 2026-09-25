import numpy as np

from neuroez_c.task2.functional_graph import GraphConfig, build_phase_graph, is_brain_channel, normalize_channel_name, sparsify_topk


def test_channel_normalization_and_exclusion():
    assert normalize_channel_name(" la1 ") == "LA01"
    assert not is_brain_channel("ECG1")
    assert not is_brain_channel("LA1", {"status": "bad"})
    assert is_brain_channel("LA1", {"status": "good"})


def test_raw_aec_graph_contract_and_topk_union():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(6, 2500)).astype(np.float32)
    raw[1] += 0.5 * raw[0]
    graph = build_phase_graph(raw, 250.0, ["LA1", "LA2", "LB1", "LB2", "LC1", "EKG1"], config=GraphConfig(topk=2))
    matrix = graph["adjacency"]
    assert graph["graph_valid"]
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.all(matrix >= 0.0)
    assert not graph["valid_channel_mask"][-1]
    dense = np.ones((5, 5), dtype=np.float32) - np.eye(5, dtype=np.float32)
    sparse = sparsify_topk(dense, 2)
    assert np.allclose(sparse, sparse.T) and np.allclose(np.diag(sparse), 0.0)


def test_graph_invalid_when_nyquist_or_nodes_insufficient():
    graph = build_phase_graph(np.ones((3, 500)), 100.0, ["A1", "A2", "A3"])
    assert not graph["graph_valid"]
    graph = build_phase_graph(np.ones((4, 500)), 80.0, ["A1", "A2", "A3", "A4"])
    assert not graph["graph_valid"] and "nyquist" in graph["invalid_reason"]

