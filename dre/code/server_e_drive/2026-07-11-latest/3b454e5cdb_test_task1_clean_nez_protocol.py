from task1_clean_nez.protocol import make_protocol

def test_each_subject_is_test_once():
    folds=make_protocol([f"p{i}" for i in range(10)],5,.2,42); assert sorted(x for f in folds for x in f["test_subjects"])==[f"p{i}" for i in range(10)]
