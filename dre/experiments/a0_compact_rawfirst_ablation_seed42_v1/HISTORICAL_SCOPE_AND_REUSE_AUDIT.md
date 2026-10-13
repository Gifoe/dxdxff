# Historical scope and reuse

These scopes were read from actual source/protocol/reports. Numerical eligibility
still requires the executed `BASELINE_REPRODUCTION.json` and private seal checks.

| Historical source | Params | Source | Inputs / views | Training / estimand | Eligibility |
|---|---:|---|---|---|---|
| d2_no_source_no_zdelta_outer_seed42_v1 |8817|No|Masked88D, ABS+DELTA+RATIO62|Original A0 patient-update BCE;80-ID patient-equal outer|**F0 authoritative**; hash-check all5 models/banks/thresholds/sealed scores |
| d2_ictal_ablation_seed42_v1 w/o Source |8817|No|All88D, four views|Same A0 protocol /80-ID outer|**F6 restored-ZDELTA control**, not full compact model |
| d2_ictal_ablation_seed42_v1 w/o ZDELTA |9221|Yes|Masked88D, ABS+DELTA+RATIO62|Original source penalty0.001 /80-ID outer|**F7 restored-Source control**, not w/o Source |
| Old seven-arm D2 feature deletions |9221|Yes|Their own all-view masks|Source-conditioned comparisons|Not F1–F5 compact deletions; no reuse |
| d2_no_patient_z_outer_seed42_v1 |9221|Yes|All88D, no Patient Z|Source-conditioned, all views|Only its deterministic FIT-only absolute preprocessing functions reused; results/weights excluded |
| a0_b1_no_layernorm_outer_seed42_v1 |8641|No|All88D, includes ZDELTA|NoLN fitted on different full input|Not compact F5; new training required |
| rawtiny_patient_relative_seed42_v1 |5729/32993/34018|No task source head forM1/M2|RawTiny, historical A1 pooling/head|A1 weight2:1/lr1e-4, VLOO fixed-query development|Exact RawTinyEncoder and data alignment only; **no scores, checkpoint, head, loss or AP reuse** |
| a0_fcpa_lf_seed42_v1 |No new model fitted|NA|Raw/EDF fidelity audit|No classifier or new outer|Use support/provenance limitations only; not performance evidence |
| a0_purn_seed42_v1 |8817|No|Alternate normalization, all88D|Failed development gate; authorized negative outer|Excluded; no extra normalization arm |
| a0_pr_spline_seed42_v1 |up to9265|No|All88D plus remapping|Failed development gate; no new outer|Excluded; no feature remapping |
| a0_no_zdelta_seed42_v1 (immediately prior task) |8817|No|Same nominal compact interface|Separate exact A0 replay/retrain|Not substituted for authoritative historical F0; only contextual corroboration |

F0 original selected epochs3/3/2/3/4 and numeric thresholds
0.465/0.240/0.320/0.335/0.410. Its original freeze/seal SHA are pinned in
`PROTOCOL_LOCK.json`; actual checkpoint/bank/prediction SHA per fold are
emitted by admission. F6/F7 uses the old seven-arm freeze/seal, not a newer
chosen test model. Validation is deterministically reconstructed from the
frozen checkpoint and exact legal bank and must replay stored selection;
this is not a claim that the old runner saved a separate original VAL file.

The complete F0 identity was explicitly selected by the user after previously
viewed experiments. It is slightly below restored-all88D F6 on old outer.
That does not authorize changing the complete model. All new results are
exploratory repeated patient-held-out evidence, not blind confirmation.
