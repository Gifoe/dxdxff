"""Read-only fresh waveform-bank/mask coverage check, not a prediction pass."""
import numpy as np
import pandas as pd
import final_stage as f
c=f.c


def main():
    f.setup();meta,banks=c.io.identity_and_banks(labels=False);store=c.RawStore()
    seal=c.jr(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');rows=[];seen=set()
    for fold in range(1,6):
        ix=np.flatnonzero(np.isin(meta['patient'],sorted(banks[fold-1]['roles']['test'])))
        counts=dict(fold=fold,patients=0,channels=0,seizures=0,valid_windows=0,zero_pre_channel_seizure_units=0,
                    fallback_seizures=0,encoded_samples=0,clipped_samples=0)
        for pid in sorted(set(meta['patient'][ix])):
            assert pid not in seen;seen.add(pid);ai=ix[meta['patient'][ix]==pid]
            raw,mask,times=store.get(pid,meta['channel'][ai],device='cpu')
            assert raw.shape[-1]==500 and mask.shape==raw.shape[:-1] and len(times)==len(raw)
            cache=f.FINAL/'outer_patient_scores_private'/f'fold{fold}'/(c.io.private_key(pid)+'.pkl')
            q=c.read(cache);assert q['binding']['freeze']==seal['freeze']
            assert q['binding']['archive_indices']==ai.tolist() and q['binding']['waveform_sha256']==store.plans[str(pid)]['sha256']
            assert q['valid_windows']==int(mask.sum())
            counts['patients']+=1;counts['channels']+=len(ai);counts['seizures']+=len(raw);counts['valid_windows']+=int(mask.sum())
            audit=q['normalization_audit']
            counts['zero_pre_channel_seizure_units']+=audit['zero_pre_channels']
            for k in ['fallback_seizures','encoded_samples','clipped_samples']:counts[k]+=audit[k]
        rows.append(counts)
    sums={k:sum(r[k] for r in rows) for k in rows[0] if k!='fold'}
    assert len(seen)==sums['patients']==80 and sums['channels']==7635 and sums['seizures']==256 and sums['valid_windows']==1471965
    assert sums['encoded_samples']==sums['valid_windows']*500
    pd.DataFrame(rows).to_csv(f.PUBLIC/'RAW_COVERAGE.csv',index=False)
    c.write(f.PUBLIC/'RAW_COVERAGE_VERIFICATION.json',dict(status='PASS',totals=sums,all80_actual_Raw_banks_freshly_hash_verified=True,
          original_full_patient_channels_and_masks_preserved=True,new_training_or_inference=False,
          source_sha256=c.sha(c.HERE/'verify_raw_coverage.py')))
    print('PRECOMPAND_ACTUAL_RAW_COVERAGE_PASS',sums,flush=True)


if __name__=='__main__':main()
