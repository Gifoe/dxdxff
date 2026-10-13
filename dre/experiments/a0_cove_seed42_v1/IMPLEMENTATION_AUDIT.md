# Executed implementation audit

PASS: all five authoritative compact F0 checkpoints, legal FIT/VAL banks, original preprocessing, feature order and frozen development predictions. The reused predictions reproduce AUROC0.7165694393593033, AUPRC0.5085590459690381, Macro-F10.6420669456105904, EZ-F10.43351598043908673 and MRR0.6765905768221185. A newly evaluated baseline never replaces the frozen scores. No historical outer score file or full feature-label array is opened.

Name/index mapping identifies14 main triples,12 retained constant-zero positions,8 DELTA auxiliary duplicate copies and26 masked ZDELTA positions. Original Patient Z and FIT-only imputer/scaler remain byte-bound and numerically unchanged. Each correction updates only its mapped DELTA/RATIO main coordinate; modified DELTA auxiliaries copy their main value exactly. Protected zeros/ABS/order are preserved. There are28, not36, added trainable parameters:14 alpha and14 beta, total8,845. Nuisance coefficients are frozen buffers, not AdamW parameters.

Ridge alpha1/intercept fits use one total weight per FIT patient. Four patient-held-out nuisance partitions are label-blind and fold-seeded. The normalizer is the patient-equal population SD of OOF FIT residuals, floored at1e-6; no residual mean is subtracted at target inference. Full-FIT nuisance models alone apply to validation. All mapping, nuisance, permutation and legal-bank files freeze before comparative development outcomes. B1/B2/B3 share them.

The original shared A0 network is constructed first, preserving initialization/RNG consumption. All new models train from scratch; none warm-starts from a trained F0. Source penalty0, AdamW0.001/0.0001, original FIT NEZ-positive BCE weighting and patient update order, clip1, maximum30/minimum6/patience6, Macro-F1/EZ-F1/earlier-epoch selection and own0.005 threshold grid are unchanged.

### Indexed OOF training adapter

The unchanged trainer passes legal row indices to a constructor-bound data adapter. The adapter resolves the original88D inputs and their appropriate OOF-FIT/full-FIT-validation residuals, then applies the **same88D COVE transform and original classifier**. An index is never a network feature. Source `g` remains unchanged and unused by the source-free model. The adapter's legal-bank buffers are nonpersistent; no extra patient/channel parameter or learned head exists. Future target inference takes88D inputs and computes residuals directly with full-FIT frozen parameters. Pretraining and all15 selected validation replays verify indexed/direct parity and exact selected threshold/metrics.

### Executed engineering gates

- Bitwise identical original shared backbone initialization in all15 cells; zero28 coefficients and identity input/logits on every legal sample.
- Exact8,845 parameter counts, onlyalpha/beta new trainables; finite CPU/CUDA gradients. Direct per-coordinate Jacobians are nonzero for every numerically active correction tested at initialization.
- Protected zero/duplicate alignment, deterministic wrong-pair derangements within mean/std, label-blind nuisance exclusion and repeatable cross-fitting.
- Exact synthetic CPU model/Adam/history/CPU-CUDA RNG resumability under the original trainer. GPU bitwise training resumability is not claimed.
- All15 new selections freeze before comparative evaluation and frozen intervention summaries. No posthoc AUROC checkpoint choice.
- Independent sklearn verification of1,430 patient/appearance metric cells, reconstructed nuisance fits/normalizers, fixed-score bootstrap intervals, aggregates and model/checkpoint contracts. Additional score-only verification checks all1,170 ranking/intervention cells and14 group effects.

### Actual engineering deviation and retained evidence

Initial Python3.12.13 / NumPy2.4.3 / SciPy1.17.1 preparation suffered native access violations, including a faulthandler trace in residual diagnostics. No clinical COVE training epoch had run. Two partial preparation directories and all logs were retained. Completed nuisance fits were compared outcome-free against Python3.13.9 / NumPy2.3.5 / SciPy1.16.3 / sklearn1.7.2: max coefficient drift1.1574e-14, scale drift8.88e-16, normalized residual drift3.68e-13, product drift1.19e-13. Both use PyTorch2.11.0+cu128, cuDNN91900. Synthetic original-backbone CPU/CUDA output, gradients and AdamW update differences are exactly0 in the executed parity check.

A separate parity helper initially triggered duplicate-OpenMP loading when importing Torch before conda NumPy. Loading NumPy first fixed that helper; no `KMP_DUPLICATE_LIB_OK` bypass, driver change or disabling cuDNN was used. This does not prove the exact cause of the earlier Python3.12 statistical crashes. Formal preparation/training/evaluation completed in the verified Python3.13 environment. The scientific code/protocol/feature/split/model/loss/selection settings were not altered. `RUNTIME_RIDGE_PARITY.json` and `RUNTIME_BACKBONE_PARITY.json` record measured equivalence, not inferred equivalence.

No installed skill, new raw extraction, outer labels/predictions, additional arms/seeds or outcome-driven repair was used. Only source and anonymous aggregate files are exported; all private arrays, identifiers, nuisance fits, weights, optimizer states and logs remain on the server.
