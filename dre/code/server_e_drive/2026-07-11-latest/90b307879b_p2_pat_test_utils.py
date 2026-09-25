from __future__ import annotations
import pandas as pd


def channel_frame(subjects=("hup:a","lzu:b","pediatric:c"), fold=1, threshold=.45):
    rows=[]
    for index,subject in enumerate(subjects):
        center=subject.split(":")[0]
        for channel,label,logit in (("A",0,-.8+index*.1),("B",0,-.2+index*.1),("C",1,.2+index*.1),("D",1,.8+index*.1)):
            rows.append({"subject_id":subject,"center":center,"outer_fold":fold,"channel_name":channel,"label_nez":label,"label_ez":1-label,"base_nez_logit":logit,"selected_threshold":threshold,"threshold_source":"outer_validation","true_count_used_for_prediction":False,"q10_nez_probability":.3+index*.05,"anchor_evidence":.1+index*.02,"temporal_delta_norm":.2+index*.03,"valid_seizure_count":1+index%3})
    return pd.DataFrame(rows)
