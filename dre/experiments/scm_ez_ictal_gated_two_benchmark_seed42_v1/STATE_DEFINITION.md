# SCM-EZ state definition

Every state is assigned from the source cache's explicit `window_relative_centers_sec`. Array ordinal is never used as time.

| State | Frozen interval |
| --- | --- |
| PRE-EARLY | `t < 0` and `t <= (min(pre)+max(pre))/2` |
| PRE-LATE | `t < 0` and above that midpoint |
| ONSET-EARLY | `0 <= t < 5` |
| ONSET-LATE | `5 <= t < 10` |
| SPREAD | `10 <= t < 30` |
| LATE | `t >= 30` |

The state vector is the per-frequency median of valid 2-second windows. An absent state is invalid and remains zero; it is never interpolated. The audited Ictal source contains centers ending at approximately +29 s, so LATE is absent in all 256 records. The six-state tensor and mask are retained exactly; no synthetic +30 s sample is introduced.

For the three shorter padded recordings, the raw window start is computed from the real onset sample plus `(center-1)*fs`. This maps every explicit center exactly inside the source-declared valid raw span.
