# N6F Final Method

N6F_NEZ_DualView_EMA_RobustRank keeps the Step4B engineered-feature backbone and adds a lightweight raw waveform view. Logits always mean NEZ probability after sigmoid. Cache labels remain EZ-positive: `labels_ez == 0` is clinically clean NEZ (`P`, target NEZ=1), while `labels_ez == 1` is observed-EZ (`U`, nominal target NEZ=0) and may contain label noise.

## Views And Alignment

The feature branch is the existing B0 spectral/classical plus static-top20, physics-gated, temporal and cross-seizure aggregation path. The raw branch independently encodes each onset-aligned raw 2-second window with a Conv1d stem and three depthwise-separable temporal blocks, then has its own temporal encoder, seizure aggregator and channel classifier.

Feature and raw records are matched with the compound identity `(subject_id, run_id, sample_id, source_seizure_id or onset/start/end)`. Within each matched record, normalized canonical channel names and explicit feature window centres identify each raw window. Duplicate or ambiguous keys fail the audit. Raw arrays are not aligned by array order. The current cache contract uses 250 Hz, 60-second onset-centred waveforms, and 500 samples per 2-second feature window.

For each channel with raw coverage, normalized feature and raw patient embeddings and both branch probabilities enter a bounded gate. `g_f = 0.05 + 0.90 * sigmoid(gate_raw)` and `z = g_f*z_feature + (1-g_f)*z_raw`. If no raw window exists for a patient-channel, the gate is forced to one and the final logit is exactly the feature logit.

## EMA Robust Learning

The student is optimized. A complete EMA copy has no gradient and is initialized from the lazy-initialized student. After every optimizer step, floating values update by `teacher = decay*teacher + (1-decay)*student`; non-floating buffers are copied. Validation, threshold selection and held-out testing use the EMA teacher.

For observed-EZ channels, teacher probability produces detached reliability

```text
r_u = clamp(1 - noise_discount * sigmoid(z_teacher), reliability_min, 1).
```

Warmup uses `r_u=1`. Clean NEZ always uses unweighted positive BCE. Observed-EZ negative BCE is weighted by `r_u` but divided by the fixed observed-EZ channel count, not by reliability mass. Losses are averaged by patient.

```text
L_total = L_fused_cls
        + lambda_feature_aux * L_feature_aux
        + lambda_raw_aux * L_raw_aux
        + lambda_rank_effective * L_rank
        + lambda_gate * L_gate.
```

The rank term compares clean-NEZ and observed-EZ logits with teacher reliability and fixed denominator `|P|*|U|`. There is no learned class prior, propensity, nnPU, hard relabeling, calibration, center bias, OOF teacher, or test-time true count.

## Evaluation Isolation And Outputs

Each outer fold splits patients into fit, validation and held-out test before training. The normalizer fits on fit patients only. Epoch selection uses the threshold-free harmonic mean of patient macro NEZ/EZ AUPRC. After the best EMA checkpoint is restored, one threshold is selected from validation predictions only; outer test is then evaluated once.

`dual_view_cache_audit.json` records cache coverage. `n6_train_loss_by_fold_epoch.csv` records robust losses, EMA reliability and gate diagnostics. `n6_validation_thresholds.csv` records validation-only thresholds. Each `n6_fold_{fold}_audit.json` records split isolation, EMA checkpoint use, raw coverage and forbidden-data assertions. Existing patient/channel predictions and held-out summary files are retained with optional branch score and gate diagnostics.
