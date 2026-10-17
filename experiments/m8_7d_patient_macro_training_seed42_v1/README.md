# Patient-Macro training of the retained seven-feature C2 single head

Prospective training-method study,not a new-architecture or performance claim.
The already designed C2 architecture,all10378parameters,original7D/14inputs,
Raw/Gate/completepatientZ and one scalar classifier remain unchanged.
The goal is actual full80patient-Macro>0.64 and>oldC2,not its rounded
0.6396579900303192 or a Development value.

Both new objectives mix normalized BCE and original soft patient-Macro
loss equally. T0 uses globalFIT class weights,normalized by the patient's
total class weight. T1 instead averages the two within-patient class mean
BCEs. T1−T0 addresses local versus global classbalance;either versus old
C2 includes the normalization/mix change as well. No test labels or
prevalence are needed at inference. There is no new label,feature or band.

All five original patient-heldout roles,seed42,initial tensors,Raw windows,
optimizer,30/6/6budget and201threshold selection rules are fixed. Tennew
fresh end-to-end fits,five exact oldC2 controls reused afteractualRawreplay.
All FIT patients must have both original classes;fail rather than filter.

NEW eligibility:DevelopmentMacro≥C2+.003 andAUROC/AP≥C2−.01. Amongeligible
chooseMacro−.1foldsampleSD. No eligible candidate means no newOuter.
Otherwise separatefinalsource/model/thresholdfreeze before onefull80pass.
Development65appearances47people isselection-conditioned;Outer repeatedly
viewed is exploratory repeated-test,not independent confirmation.

Report all9metrics,all54cluster intervals,fold/source/coverage and actual
epochs,including negatives. Code/hash/anonymousactualaggregates only to
GitHub;Raw,weights,scalers,individualscores/IDs and logs stay on the server.
