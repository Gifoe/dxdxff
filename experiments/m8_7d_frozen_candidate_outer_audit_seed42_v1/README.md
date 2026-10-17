# Frozen candidate Outer audit, original 7D single-head models

This separate audit was authorized by the user after the five internal studies had completed. It evaluates **all ten already-trained candidates and the existing C2 reference**, regardless of their unchanged historical Development improvement gates. No old protocol or failed gate is modified. There is no candidate selection in this audit.

## Fixed scope

- Eleven arms, original five patient-held-out folds, seed42, all 80 patients / 7,635 channels / 256 seizures. Each new arm uses all 1,471,965 valid raw250 windows.
- Existing selected checkpoints, epochs, thresholds and FIT-only preprocessors are reused without training or threshold changes. Original seven physiological measurements and a single scalar classification head remain unchanged.
- Four end-to-end candidate families: temporal dispersion pooling, early physiological FiLM, channel relation and patient-balanced objective. The context-affine family assembles its existing frozen C2 body and selected lightweight adjustment into the same one-readout model.
- All 55 logical model/fold cells are frozen before new Outer signal inference. Fifty cells require new frozen inference; five C2 score cells are reused from their existing hash-locked score seal.
- A read-only adapter check replays one existing validation patient per checkpoint numerically. It neither calculates Development performance nor selects models/thresholds.
- All 55 score cells must be sealed before full-truth reporting. Exact private patient cache bindings permit engineering resume without repeating completed predictions as a new scientific test.

## Reporting

Primary endpoint is equal-patient binary Macro-F1 over all 80 patients; an observed value must be strictly greater than 0.64 to exceed the requested point target. AUROC, AP, EZ-F1, NEZ-F1, ACC, MRR, Top-1 and EZ-fraction bias accompany it. Pooled channels, fold means/SD, source summaries and coverage are separate tables. Patient pairing is retained for 10,000 seed42 bootstrap draws: 234 nominal interval rows and ten additional multiplicity-aware Macro-F1 difference intervals.

All eleven arms are reported in the protocol's fixed order. Any arm above 0.64 is an **observed exploratory result**, not a retrospectively selected independent confirmation. Historical Outer has been repeatedly viewed, and this audit evaluates multiple candidates. No post-Outer tuning is permitted.

## Execution

Deploy the exact source directory on the trusted Windows server under the experiment's declared D: code path. Run `run_outer.ps1 -Stage Admit -RunTag 01` to check adapters and freeze; then `run_outer.ps1 -Stage All -RunTag 02` in a sustained foreground SSH session. The coordinator locks its runtime, rejects duplicates, preserves logs and does not automatically retry failures. Use a fresh RunTag only after diagnosis. No original experiment source is overwritten.

The public checker `python verify_public.py` consumes only compact aggregates in `results/`. Private waveforms, identities, per-patient predictions, checkpoints and logs stay on the trusted server. Execution status and actual results will be added only after verified runs.
