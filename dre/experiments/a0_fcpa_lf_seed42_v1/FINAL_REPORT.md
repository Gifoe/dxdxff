# FCPA-LF seed42: completed negative data-fidelity audit

**Terminal: LF_DATA_FIDELITY_GATE_FAILED.** Stage A ran on the server.
Stages B–F did not run. No model training, baseline inference, threshold
selection, LF24 feature extraction or new Outer TEST was performed.

## Performance table — explicitly not evaluated

| Model | Parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---|---|---|---|---|---|
| B0 source-free A0 |8,817 historical; not instantiated here|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
| B1 generic residual |12,705 intended, not runtime-verified|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
| B2 LF-Direct |12,705 intended, not runtime-verified|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
| B3 LF-Additive |12,705 intended, not runtime-verified|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
| B4 FCPA-LF |12,705 intended, not runtime-verified|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
| B5 shuffled-LF |12,705 intended, not runtime-verified|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|

The historical matched A0 development reference remains AUROC0.710667,
AUPRC0.518235, Macro-F10.638080, EZ-F10.431198, MRR0.686550. These values
are from the existing hash-bound PMA report/task, **not a new baseline replay**.
Historical outer numbers were not used to select data, bands or a model.

## Actual Stage A results

| Check | Actually measured / established |
|---|---|
| Cohort |80 patients,256 records,7,635 canonical channels, unchanged5 folds|
| Raw complete supports |15,074 record-windows;1,471,965 channel-windows|
| Finite delta/theta/alpha |100% in all three bands|
| Median fractions of1–40Hz |Delta50.95%, theta20.21%, alpha8.69%|
| Original EDF/header-channel resolution |256/256 records|
| Accessible sidecar onset-time corroboration |162/256 records;4,592 canonical channels|
| EDF+D unresolved physical-time declaration |94 records:72 LZU,22 pediatric|
| Header-declared bandwidth candidates |925/7,635 channels,12.1153%, Multicenter only|
| Required verified-source coverage |At least95% canonical channels, all four centers|
| Exact Raw generation config/source SHA |PROVENANCE_NOT_PROVEN|
| Result |Mandatory data gate failed|

The925 count is a **documented candidate subset at header-declaration level**,
not a blanket guarantee of acquisition-device calibration or physiology.
All7,635 channels remain in the coverage denominator; no subset replaces the
cohort. Both the Raw route and verified EDF recovery route fail the task gate.

### Strong numerical correspondence does not repair provenance

On162 continuous-header diagnostic records (894,570 channel-windows),
Raw/original-EDF delta/theta/alpha median power ratios are
0.983282/0.997716/0.995868 and Spearman correlations
0.996938/0.999451/0.999698. This supports preservation of numerical spectral
content on those compared supports. It does not establish the missing
acquisition filter history, exact transformation chain or EDF+D time mapping.

The fixed10Hz negative control yields much less delta/theta power than Raw:
median Raw/control ratios1.5832e6/1704.83. The observed Raw is not equivalent
to that specific control. This is not a retrospective proof that no historical
10Hz path ever affected any source. Similarity to the fixed1Hz candidate also
does not identify the generating CLI.

The originally executed all256 index-based tables are retained and explicitly
qualified. Use `LF_PSD_COMPARISON_QUALIFIED_V2.csv` for timeline/native-bin
interpretation. Native-rate rounding means integrated masked-bin diagnostics
are not dense PSD-shape proof; total-minus-low is a residual proxy, not an
exact13–40 integral for every nonaligned grid. No spectral result chose a
source, filter, frequency band or relaxed criterion.

## Answers to the required research questions

1. **Faithful delta/theta/alpha?** Strong numeric agreement on compared
   continuous-header supports, but whole-cohort independent LF fidelity is
   unverified. The data gate does not pass.
2. **Did10Hz processing affect this Raw Cache?** Exact historical chain unknown.
   Raw differs strongly from the specified10Hz negative control. That does
   not recover the actual CLI or certify original acquisition bandwidth.
3. **Valid LF features recovered without changing A0?** A0 unchanged; no
   supplementary LF24 bank was admitted or extracted.
4. **Nonredundant versus88D?** Not tested. The frozen zero/duplicate dimensions
   and missing full alpha motivate a question, not an information-gain result.
5. **Direct LF addition improves localization?** Not evaluated; B2 did not train.
6. **FCPA improves AUROC/AUPRC?** Not evaluated; B4 did not train.
7. **Beyond generic capacity?** Not evaluated; no B4–B1 contrast exists.
8. **Multiplication beats additive/direct?** Not evaluated; no retrained controls.
9. **Correct channel correspondence matters?** Not evaluated; no B5 or frozen shuffle.
10. **Fold/center improvements?** No predictive results. Verified candidate source
    evidence does not cover all centers; all patients remain in the audit.
11. **Sensitivity versus false positives?** Not evaluated; no new predictions.
12. **Reliability q meaningful/label-independent?** No q feature bank constructed.
    Source/support masks in Stage A are label-blind; no q-performance claim.
13. **Suitable as JBHI main model?** Not established. A trainable mechanism and
    trustworthy full-cohort input are not yet demonstrated.
14. **Demonstrated versus unverified?** Measurable Raw LF, exact cohort/cache
    identity, anonymous valid-support/EDF index diagnostics were demonstrated.
    Full processing provenance, acquisition confirmation, EDF+D timing,
    incremental information and model efficacy remain unverified.

## Execution and scientific boundary

Synthetic signal/EDF calibration/name parity tests passed on the server;
synthetic95%-gate tests passed locally and on the server. The actual negative
gate passed integrity verification. A local SSH stderr pump timed out while
the healthy EDF process continued; all256 atomic pair files and completion
status were verified. No scientific retry or completed PSD was discarded.

The full negative report is the authorized terminal, not an unfinished
training plan. Conditional model/extraction/training files are intentionally
not implemented after the failed mandatory gate. Downstream CSVs explicitly
record NOT_RUN; they contain no fabricated zeros, gains or bootstrap intervals.
No architecture/band rescue, new outer evaluation or clinical efficacy claim.
The historically viewed cohort would make any future development exploratory,
not fresh blind confirmation. No OUTER_READINESS recommendation is issued.
