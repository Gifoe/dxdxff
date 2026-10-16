"""Post-seal, label-free Raw mask/support census; no model fit or prediction."""
import torch
import final_stage as f
c=f.c


def main():
    binding,_,_=f.setup();status=c.jr(f.PUBLIC/'OUTER_STATUS.json')
    assert status['status']=='ALL35_FROZEN_FRESH_RAW_SCORE_CELLS_SEALED'
    assert status['score_seal_sha256']==c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    seal=c.jr(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');assert seal['all35_scores_before_complete_Outer_truth'] and len(seal['cells'])==35
    meta,_=c.io.identity_and_banks(labels=False);store=c.RawStore()
    counts=dict(patients=0,canonical_channels=0,seizure_records=0,valid_windows=0,pre_windows=0,
        phase_0to8_windows=0,phase_8to20_windows=0,at_or_after20_windows=0,channel_seizure_units=0,
        available_channel_seizure_units_missing_pre=0,channels_missing_0to8_in_all_seizures=0,channels_missing_8to20_in_all_seizures=0)
    minimum=maximum=None
    for pid in sorted(set(meta['patient'])):
        ii=c.np.flatnonzero(meta['patient']==pid);raw,mask,times=store.get(pid,meta['channel'][ii],device='cpu')
        assert torch.isfinite(raw).all() and torch.isfinite(times).all() and raw.shape[-1]==500
        pre=mask&(times[:,None,:]<0);a=mask&((times[:,None,:]>=0)&(times[:,None,:]<8))
        b=mask&((times[:,None,:]>=8)&(times[:,None,:]<20));late=mask&(times[:,None,:]>=20)
        assert torch.equal(pre|a|b|late,mask);available=mask.any(-1)
        counts['patients']+=1;counts['canonical_channels']+=len(ii);counts['seizure_records']+=len(raw);counts['valid_windows']+=int(mask.sum())
        for key,value in [('pre_windows',pre),('phase_0to8_windows',a),('phase_8to20_windows',b),('at_or_after20_windows',late)]:counts[key]+=int(value.sum())
        counts['channel_seizure_units']+=int(available.sum())
        counts['available_channel_seizure_units_missing_pre']+=int((available&~pre.any(-1)).sum())
        counts['channels_missing_0to8_in_all_seizures']+=int((~a.any(-1).any(0)).sum())
        counts['channels_missing_8to20_in_all_seizures']+=int((~b.any(-1).any(0)).sum())
        t=times[:,None,:].expand_as(mask)[mask];lo,hi=float(t.min()),float(t.max())
        minimum=lo if minimum is None else min(minimum,lo);maximum=hi if maximum is None else max(maximum,hi)
        del raw,mask,times
    assert counts['patients']==80 and counts['canonical_channels']==7635 and counts['seizure_records']==256 and counts['valid_windows']==1471965
    assert counts['available_channel_seizure_units_missing_pre']==counts['channels_missing_0to8_in_all_seizures']==counts['channels_missing_8to20_in_all_seizures']==0
    encoded=sum(r['valid_windows'] for r in status['coverage']);assert encoded==counts['valid_windows']
    c.write(f.PUBLIC/'RAW_SUPPORT_AUDIT.json',dict(status='PASS',counts=counts,valid_input_samples_forwarded_per_fresh_procedure=counts['valid_windows']*500,
        fixed_pre_reference_and_two_phase_support_windows=counts['pre_windows']+counts['phase_0to8_windows']+counts['phase_8to20_windows'],
        minimum_valid_window_time=minimum,maximum_valid_window_time=maximum,
        raw_registry_sha256=c.io.EXPECTED_REG,reconstruction_sha256=c.io.EXPECTED_REC,
        score_seal_sha256=c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),audit_source_sha256=c.sha(c.Path(__file__)),
        inference_valid_window_count_matches_independent_mask_census=True,
        interpretation='Allvalidwindowsforwarded; original pooling uses pre and0to20 support. Anyat-or-after20 windows are not claimed to contribute to finalclassifications.',
        full_truth_archive_not_opened_by_audit=True,no_new_classification_metrics_computed=True,
        legacy_mixed_metadata_physically_unpickled_but_no_y_returned=True,
        no_training_or_prediction=True,no_masks_or_data_changed=True))
    print('FRESH_RAW_INDEPENDENT_SUPPORT_CENSUS_PASS',counts,flush=True)


if __name__=='__main__':main()
