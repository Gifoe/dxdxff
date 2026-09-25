from neuroez_c.task2.trace_rawmil.trainer import better_checkpoint

def test_auroc_is_primary_checkpoint_metric():
    assert better_checkpoint(.71, 9.0, .70, .1)
    assert not better_checkpoint(.70, .10005, .70, .1)

def test_bce_breaks_only_auroc_ties():
    assert better_checkpoint(.70, .0998, .70, .1)
    assert not better_checkpoint(.69, .0001, .70, .1)
