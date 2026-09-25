from task1_clean_nez.metrics import select_validation_threshold

def test_threshold_does_not_read_test_labels():
    val=[{"subject_id":"v","true_nez":1,"score_nez_probability":.8},{"subject_id":"v","true_nez":0,"score_nez_probability":.2}]
    t1,_=select_validation_threshold(val); test=[{"subject_id":"x","true_nez":0,"score_nez_probability":.9}]; test[0]["true_nez"]=1; t2,_=select_validation_threshold(val)
    assert t1==t2
