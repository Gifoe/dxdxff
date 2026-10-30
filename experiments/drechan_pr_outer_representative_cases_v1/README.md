# Representative and declining PR cases from frozen Ictal Outer predictions

Read-only reanalysis of the original 80-patient, 7,635-channel cohort. Compare
the frozen 12D Linear SVM No PR/Legacy PR and the original complete DREChanNet
WO_BOTH_PR/FULL at seeds 42/43/44. The latter is the 11,425-parameter original
three-seed FULL with early FiLM and late Gate, not the later no-guidance model.

`PROTOCOL_LOCK.json` fixes eligibility, median-oriented selection, stability,
negative-control priorities and privacy before candidate outcomes are computed.
The script never trains a classifier or performs a new model forward pass.

True identifiers and original contact names remain in server-private exports.
Only two selected, pseudonymized cases have channel-level exports; all80
patient-wise changes remain private. Anonymous contact aliases preserve the
original lexicographic tie order for exact rank reconstruction.
