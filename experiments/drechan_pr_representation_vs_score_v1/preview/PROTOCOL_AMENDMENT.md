# Requested early outer preview

On 2026-10-28, before accessing any new outer results, the user requested:
“除去timeconvcnn以外的都给我把outer结果展示出来”.

This changes disclosure timing: freeze all 70 SVM/DREChanNet selections, score
and seal all 70 cells, then report the complete two-model, seven-condition table
while the unchanged TimeConv training continues. The original 105-cell freeze,
score seal and three-model report remain separate. Do not describe this preview
as having waited for all three models. No condition, metric, stopping rule,
bootstrap, comparison, patient, threshold or model weight changes.

To retain a single GPU training worker, early DREChanNet inference runs in FP32
on CPU. Its device adapter is confined to this evaluation process. Before any
new outer scoring, replay every DREChanNet validation cell whose CPU forward is
actually used (NoPR and the three new representation arms): native logit
error <=1e-5, probability error <=2e-6, and every selected-threshold classification
must match. The thresholds remain the frozen GPU-selected thresholds. Historical
NoPR/FullPR outer probabilities remain byte-bound references. Score alignment
uses replayed native logits and the frozen NoPR ranking anchor. Numerical CPU
roundoff is recorded; the later GPU report independently checks agreement.

The first admission attempt additionally replayed FullPR, whose CPU forward is
not used by this preview. Fold 2 differed by 1.245737075805664e-5 in logits and
2.7418136596679688e-6 in probabilities, with all classifications unchanged. That
attempt stopped before any new test scoring. Its code, freeze and logs remain
preserved. Version 2 admits the 20 CPU-forward cells actually used and reads
FullPR only from its hash-verified original sealed predictions. The numerical
tolerances are unchanged; no test results were accessed when making this repair.

All 70 score cells must seal before label/metric access in this preview. All
80 patients and 7,635 channels remain included. Independently verify every one
of the 10,080 patient endpoints, all 270 nominal intervals, and four available
primary intervals. The primary family size stays SIX, including the two future
TimeConv comparisons; do not shrink it to four. Only anonymous aggregates leave
the trusted server. Source and parent hashes remain unchanged. No GPU trainer
is stopped, restarted or replaced.
