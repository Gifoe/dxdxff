import pytest
from task1_clean_nez.protocol import audit_protocol
def base():
    idx={"p1":{"outcome_group":"success"},"p2":{"outcome_group":"success"}}; rows=[]
    for fold,sid in ((1,"p1"),(2,"p2")): rows.append({"subject_id":sid,"channel_name":"A1","score_nez_probability":.8,"candidate_risk":.2,"threshold":.5,"predicted_nez":1,"predicted_ez":0,"true_nez":1,"true_ez":0})
    folds=[{"train_subjects":["p2"],"validation_subjects":[],"test_subjects":["p1"],"normalizer_fit_subjects":["p2"],"threshold_selection_subjects":[],"threshold_source":"validation"},{"train_subjects":["p1"],"validation_subjects":[],"test_subjects":["p2"],"normalizer_fit_subjects":["p1"],"threshold_selection_subjects":[],"threshold_source":"validation"}]
    return folds,rows,idx
def test_overlap_and_test_threshold_fail():
    f,r,i=base(); f[0]["validation_subjects"]=["p1"]
    with pytest.raises(ValueError,match="overlap"):audit_protocol(f,["p1","p2"],r,i,2)
    f,r,i=base(); f[0]["threshold_source"]="test"
    with pytest.raises(ValueError,match="test-derived"):audit_protocol(f,["p1","p2"],r,i,2)
def test_duplicate_oof_row_fails():
    f,r,i=base()
    with pytest.raises(ValueError,match="duplicate"):audit_protocol(f,["p1","p2"],r+r[:1],i,2)
