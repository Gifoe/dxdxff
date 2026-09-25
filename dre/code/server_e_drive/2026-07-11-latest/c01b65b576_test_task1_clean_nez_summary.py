from task1_clean_nez.metrics import build_oof_summary
def test_oof_summary_is_aggregated_dict():
    rows=[]
    for sid,center in (("p1","a"),("p2","b")):
      rows += [{"subject_id":sid,"center":center,"true_nez":1,"true_ez":0,"score_nez_probability":.8,"candidate_risk":.2,"predicted_nez":1,"predicted_ez":0},{"subject_id":sid,"center":center,"true_nez":0,"true_ez":1,"score_nez_probability":.2,"candidate_risk":.8,"predicted_nez":0,"predicted_ez":1}]
    folds=[{"fold_idx":1,"patient_macro_f1":.8,"a_anchor":.05,"a_early":.04,"a_recurrence":.03},{"fold_idx":2,"patient_macro_f1":1.,"a_anchor":.07,"a_early":.06,"a_recurrence":.05}]
    s=build_oof_summary(rows,folds,{}, {}, {}); assert isinstance(s["oof_metrics"],dict) and s["fold_mean"]["patient_macro_f1"]==.9 and s["fold_std"]["patient_macro_f1"]>0
