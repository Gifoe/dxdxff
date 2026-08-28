from reva_dlm.splits import assign_dev_holdout, make_g2_train_val


def test_hash_split_is_deterministic_and_disjoint():
    ids = [f"question_{index:04d}" for index in range(1319)]
    first = {case_id: assign_dev_holdout(case_id, 20260827) for case_id in ids}
    second = {case_id: assign_dev_holdout(case_id, 20260827) for case_id in reversed(ids)}
    assert first == second
    dev = {case_id for case_id, split in first.items() if split == "DEV"}
    holdout = set(ids) - dev
    assert dev.isdisjoint(holdout)
    assert dev | holdout == set(ids)


def test_g2_split_partitions_dev_without_holdout():
    ids = [f"question_{index:04d}" for index in range(100)]
    strata = ["A" if index % 4 else "B" for index in range(100)]
    train, val = make_g2_train_val(ids, strata, 20260827)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(ids)
