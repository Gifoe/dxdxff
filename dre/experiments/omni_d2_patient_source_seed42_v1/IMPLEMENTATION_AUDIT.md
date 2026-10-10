# Implementation contract

Protocol froze before extraction/new outcomes at SHA256
`25e9c5b18e44ef7176915b5b5d8252e03437236c43a4e422d342373b487e807b`.
Preparation has separately sealed the137 cached records, historical models,
canonical clinical labels and baseline prediction caches. No extraction
function loads a clinical label array. Every patient/fold embedding cell is
atomic and binds the original record/CNN/protocol/extractor hashes.

Extraction reads each patient's cached60s waves once. The exact official
transform generates batch16 spectral images once, then each of the five
original models performs its own unchanged forward on those images. All
models stay eval/require_gradFalse; before/after state hashes are compared.
It does not substitute one model's representations for another. The original
float32 EDF reduction is preserved before float64 segment averaging. No new
raw extraction/filtering/resampling, altered windows or mean-sigmoid pooling.

Label-free32D normalization uses all good canonical channels within patient,
including unknowns, population variance plus1e-5. Labels are sealed separately.
The training bank API rejects any requested outer-held label access. Each
new head's training computation is float32 CPU, avoiding unnecessary CUDA
launches for35/105 parameters. Inference adds float32 learned correction to
the authoritative float64 frozen baseline logit; zero correction preserves
the exact baseline. No mixed precision or extra nonlinear head.

Inner source-stratified partitions and C3 assignments are SHA256 seeded with
42 and labels are not an input. FIT and held source permutations are separate,
preserve their source marginals, and remain fixed. All arms share partitions,
deterministic patient order and initialization policy; C2/C3 initial tensors
are identical. Three inner heads per arm/fold each train all50 epochs;
complete inner head-OOF patient AUROC/AP/earlier epoch selects length. Refits
reinitialize and train on all outer FIT for exactly selected length. One
patient update, original1/2 class weights normalized within patient, locked
count-weighted sourceL2, sharedL2, AdamW .001/.0001, clipping1.

Threshold grid201points and tie rules are FIT-only. C0 gets its own FIT-derived
threshold and exact historical0.5 diagnostics remain. All20 C0/head selections
are hash-frozen before new outer head prediction. Each outer prediction cell
binds freeze/checkpoint/bank/source, and labels are read only after writing
the frozen prediction. Safe resume never retunes any result.

Patient-equal summaries use124 patients exactly once. Bootstrap uses only
unique patients,10000draws/seed42 and four registered paired contrasts.
Historical reuse and in-sample CNN features for inner head validation are
explicit limitations; no claim of full-pipeline inner OOF or external test.

Source constant bias is a threshold/calibration term, not a within-patient
ranking term. Bias-off diagnostic ranking preserves full frozen ordering
after verifying its constant offset, avoiding finite-precision sigmoid ties.
Direction-off removes only U@V and retains shared beta/bias; all-off exactly
recovers C0 probabilities. All interventions keep C2's frozen numeric
threshold, which is not C0's separately selected threshold.

Required source matrix is a derived, non-identifying3x32 U@V diagnostic,
not the private checkpoint or individual embeddings. No patient identifiers,
channel identifiers, clinical labels, prediction ledger or optimizer tensors
are public. Engine tests, original hashes and delivered aggregates are
checked independently before publication.
