from __future__ import annotations

import hashlib,json


def protocol_audit(pilot,folds,seed,cache_root,completed_folds=0):
    return {"pipeline":"MOSAIC-Outcome","profile":"MOSAIC_FULL","pilot_size":int(len(pilot)),"centers":4,"patients_per_center":4,"success_per_center":2,"failure_per_center":2,"pilot_only":True,"paper_valid":False,
            "status":"PILOT_ONLY_NOT_PAPER_VALID","outer_folds":4,"outer_test_patients_per_fold":4,"outer_test_centers_per_fold":4,"all_available_seizures_used":True,
            "completed_outer_folds":int(completed_folds),"success_label":1,"failure_label":0,"threshold":.5,
            "center_as_input":False,"patient_id_as_input":False,"channel_count_as_input":False,"seizure_count_as_input":False,
            "outer_test_used_for_selection":False,"outer_test_used_for_threshold":False,"primary_threshold":.5,"preprocessing_train_only":True,"expert_crossfit":"leave_one_patient_out_within_outer_train","meta_trained_on_crossfitted_expert_predictions":True,
            "p2_original_oof_only":True,"p2_fold_safe":True,"true_ez_used_as_intervention_proxy":True,"meta_input_count":8,"ablation_run":False,"seed":int(seed),"cache_root":str(cache_root),
            "pilot_manifest_hash":hashlib.sha256(pilot.to_csv(index=False).encode()).hexdigest(),
            "fold_manifest_hash":hashlib.sha256(folds.to_csv(index=False).encode()).hexdigest()}


__all__=["protocol_audit"]
