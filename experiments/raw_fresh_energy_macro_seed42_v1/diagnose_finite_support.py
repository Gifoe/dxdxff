"""Aggregate valid-versus-padding finiteness diagnostic after score sealing."""
import numpy as np
import torch
import final_stage as f
c = f.c


def main():
    f.setup()
    status = c.jr(f.PUBLIC / 'OUTER_STATUS.json')
    assert status['status'] == 'ALL35_FROZEN_FRESH_RAW_SCORE_CELLS_SEALED'
    meta, _ = c.io.identity_and_banks(labels=False)
    store = c.RawStore()
    counts = dict(patients=0, valid_windows=0, invalid_slots=0,
                  valid_windows_with_nonfinite_samples=0,
                  invalid_slots_with_nonfinite_samples=0,
                  valid_windows_with_nonfinite_times=0,
                  invalid_slots_with_nonfinite_times=0,
                  non500_sample_banks=0)
    for pid in sorted(set(meta['patient'])):
        ii = np.flatnonzero(meta['patient'] == pid)
        raw, mask, times = store.get(pid, meta['channel'][ii], device='cpu')
        bad_raw = ~torch.isfinite(raw).all(-1)
        bad_time = (~torch.isfinite(times))[:, None, :].expand_as(mask)
        counts['patients'] += 1
        counts['valid_windows'] += int(mask.sum())
        counts['invalid_slots'] += int((~mask).sum())
        counts['non500_sample_banks'] += int(raw.shape[-1] != 500)
        for suffix, bad in [('samples', bad_raw), ('times', bad_time)]:
            counts['valid_windows_with_nonfinite_' + suffix] += int((bad & mask).sum())
            counts['invalid_slots_with_nonfinite_' + suffix] += int((bad & ~mask).sum())
    print('FULL80_FINITE_SUPPORT_DIAGNOSTIC', counts, flush=True)


if __name__ == '__main__':
    main()
