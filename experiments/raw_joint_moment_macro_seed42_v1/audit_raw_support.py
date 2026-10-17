"""Post-seal full80 valid-support census; no fitting or new predictions."""
import numpy as np
import final_stage as f
c=f.c
torch=c.torch


def main():
    f.setup();status=c.jr(f.PUBLIC/'OUTER_STATUS.json');verified=c.jr(f.PUBLIC/'INDEPENDENT_FINAL_VERIFICATION.json')
    assert status['status']=='ALL30_FROZEN_JOINT_RAW_SCORE_CELLS_SEALED' and verified['status']=='PASS'
    assert status['score_seal_sha256']==verified['score_seal_sha256']==c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    seal=c.jr(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');assert seal['all30_scores_before_complete_Outer_truth'] and len(seal['cells'])==30
    meta,_=c.io.identity_and_banks(labels=False);store=c.RawStore()
    counts=dict(patients=0,canonical_channels=0,seizure_records=0,valid_windows=0,pre_windows=0,phase_0to8_windows=0,
        phase_8to20_windows=0,at_or_after20_windows=0,channel_seizure_units=0,available_channel_seizure_units_missing_pre=0,
        channels_missing_0to8_in_all_seizures=0,channels_missing_8to20_in_all_seizures=0,invalid_window_slots=0,
        invalid_slots_with_nonfinite_times=0,valid_windows_with_nonfinite_times=0,valid_windows_with_nonfinite_samples=0)
    minimum=maximum=None
    for pid in sorted(set(meta['patient'])):
        ii=np.flatnonzero(meta['patient']==pid);raw,mask,times=store.get(pid,meta['channel'][ii],device='cpu')
        bad_time=(~torch.isfinite(times))[:,None,:].expand_as(mask);bad_raw=~torch.isfinite(raw).all(-1)
        assert raw.shape[-1]==500
        counts['invalid_window_slots']+=int((~mask).sum());counts['invalid_slots_with_nonfinite_times']+=int((bad_time&~mask).sum())
        counts['valid_windows_with_nonfinite_times']+=int((bad_time&mask).sum());counts['valid_windows_with_nonfinite_samples']+=int((bad_raw&mask).sum())
        assert not (bad_time&mask).any() and not (bad_raw&mask).any()
        pre=mask&(times[:,None,:]<0);a=mask&((times[:,None,:]>=0)&(times[:,None,:]<8))
        b=mask&((times[:,None,:]>=8)&(times[:,None,:]<20));late=mask&(times[:,None,:]>=20)
        assert torch.equal(pre|a|b|late,mask);available=mask.any(-1)
        counts['patients']+=1;counts['canonical_channels']+=len(ii);counts['seizure_records']+=len(raw);counts['valid_windows']+=int(mask.sum())
        for key,value in [('pre_windows',pre),('phase_0to8_windows',a),('phase_8to20_windows',b),('at_or_after20_windows',late)]:counts[key]+=int(value.sum())
        counts['channel_seizure_units']+=int(available.sum());counts['available_channel_seizure_units_missing_pre']+=int((available&~pre.any(-1)).sum())
        counts['channels_missing_0to8_in_all_seizures']+=int((~a.any(-1).any(0)).sum());counts['channels_missing_8to20_in_all_seizures']+=int((~b.any(-1).any(0)).sum())
        t=times[:,None,:].expand_as(mask)[mask];lo,hi=float(t.min()),float(t.max());minimum=lo if minimum is None else min(minimum,lo);maximum=hi if maximum is None else max(maximum,hi)
    assert [counts[k] for k in ['patients','canonical_channels','seizure_records','valid_windows']]==[80,7635,256,1471965]
    assert counts['available_channel_seizure_units_missing_pre']==counts['channels_missing_0to8_in_all_seizures']==counts['channels_missing_8to20_in_all_seizures']==0
    assert sum(r['valid_windows'] for r in status['coverage'])==counts['valid_windows']
    c.write(f.PUBLIC/'RAW_SUPPORT_AUDIT.json',dict(status='PASS',counts=counts,valid_input_samples_forwarded_per_fresh_procedure=counts['valid_windows']*500,
        fixed_pre_reference_and_two_phase_support_windows=counts['pre_windows']+counts['phase_0to8_windows']+counts['phase_8to20_windows'],
        minimum_valid_window_time=minimum,maximum_valid_window_time=maximum,raw_registry_sha256=c.io.EXPECTED_REG,reconstruction_sha256=c.io.EXPECTED_REC,
        score_seal_sha256=c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),audit_source_sha256=c.sha(c.Path(__file__)),
        inference_valid_window_count_matches_independent_mask_census=True,late_encoded_windows_not_claimed_as_classifier_support=True,
        full_truth_archive_not_opened_by_audit=True,no_new_classification_metrics_computed=True,legacy_mixed_metadata_physically_unpickled_but_no_y_returned=True,
        no_training_or_prediction=True,no_masks_or_data_changed=True))
    print('JOINT_RAW_INDEPENDENT_SUPPORT_CENSUS_PASS',counts,flush=True)


if __name__=='__main__':main()
