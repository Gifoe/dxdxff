# Pre-execution source audit

Reviewed Experiment A at `caaa11b864716039153cfe25c7ccbd6fa033f4e6`:
FINAL_REPORT, modeling.py, train_fold.py, evaluate_oof.py,
prepare_patient_records.py, PROTOCOL_LOCK, MODEL_PROTOCOL_LOCK_V2,
OOF_SUMMARY, FOLD_METRICS and CENTERWISE_OOF_METRICS in full.
Reviewed deployed train_fold_r3.py and evaluate_oof_r2.py, their frozen
hashes, training-completion records and OOF_FROZEN_LOCK on the original server.
Reviewed original Ictal SourceA0 model at
`394dbff2085114d5f2d162cc7802384c95f0ef8d`.

The legitimate data unit is the unique patient-channel, not the official
pooled EDF-channel test. Ordered labels are good!=1 excluded; outcome==1
and resection==0 normal0; otherwise soz==1 pathological1; otherwise unknown.
These labels do not define a strict SOZ-only target. Unknown good channels
enter normalization but never loss/metrics.

Original population is124 both-class patients,137 records,10202 labeled
canonical channels (1113 pathological/9089 normal). Source populations are
HUP10, Open-iEEG103, SourceSink11. All folds and original labels are retained.
No raw preprocessing or clinical label reinterpretation is implemented here.

TimeConv exposes feature_extractor -> cnn32D -> original fc/BN classifier.
All five baseline epoch30 checkpoint hashes are prelocked; a fold uses only
its own baseline CNN for both FIT and held representations. Inference is eval,
all parameters frozen. Good-channel NPZ caches are reused at fixed start1s,
60s/1000Hz with30 nonoverlapping2s segments and historical batch16 groups.
EDF logits are reduced in float32 at each segment; those30 mean logits are
averaged in float64 then sigmoid, as deployed historical evaluator. This is
not mean segment probabilities. The new extraction shares identical spectral
images between frozen CNNs without mixing fold representations in a head.

Existing private feature artifacts contain classical F9/F36 descriptors and
frozen scores, not the required32D TimeConv embeddings. They are not legal
substitutes. All frozen representations must be extracted and C0 freshly
reproduce every patient's metrics within1e-12, probabilities within1e-6,
before any new head training. No scores are replaced by archived values.

Original D2's operational source factorization U@V and count-weighted L2 are
adapted to32D embeddings, not transferred weights or Ictal features. C1 has
35parameters; C2/C3 have105. All identity corrections zero at initialization;
V is initialized small nonzero, U initially zero, no second zero gate.

Outer FIT CNN embeddings are in-sample for that backbone. Inner head-only
OOF is not full-pipeline OOF. Only outer held representations use a CNN
excluding that patient. Historical outcomes have been viewed: all new OOF
results are exploratory, reused-cohort, not untouched independent testing.

The prompt leaves 'materially lower', 'severe degradation', and 'meaningful'
unquantified. Before outcomes, protocol sets maximum pathologicalF1 decline
0.010, small-source AUROC decline0.050, directional AUROC contribution0.001
plus nonzero pair reversals. MRR/Top1 must not decline. No posthoc waiver.
Small HUP/SourceSink samples and count-preserving shuffled IDs which leave
many Open-iEEG IDs unchanged limit source-mechanism interpretation.

Source hashes and fixed protocol are public; identities, private manifests,
waveforms, embeddings, scores and checkpoint tensors remain server-side.
No dre-datacenter-extraction skill was used.
