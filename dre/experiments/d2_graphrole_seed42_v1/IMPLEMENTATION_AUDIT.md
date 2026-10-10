# Implementation audit

The original D2 frozen model/banks/predictions are SHA-verified and replayed.
`model.py` changes only the input dimension for the new scratch G1/G2/G3 topology;
dimension88 reconstructs original D2. All new arms have 10,789 parameters and use
the same initial state and fold seeds; no final D2 weight is imported into them.
`train_core.py` retains the original trainer and source regularizer, with timing
and finite-gradient checks only. Original threshold and RNG helpers are read-only
imports guarded by their original SHA.

All graph inputs are label-free. Each observed waveform window is checked against
the measured interval mask before excluding invalid electrodes and constructing
connectivity. Every record summary has equal seizure weight; missing rank stability
is retained until fold-specific FIT-only imputation. Original88 tensors are reused
bitwise; no original-feature preprocessor is refitted. G2 carries each entire16D
vector within a seeded availability-preserving derangement. G3's16D positions are
zero after preprocessing. The prescribed LayerNorm104 mixes all104 positions:
even a zero graph input can become nonzero after centering, so graph-column Linear
weights can receive gradients. The prompt's claim that zeros cannot activate these
weights is not true for this architecture. We preserve the requested model and
report G3 as a neutral-input architecture/retraining control, not a complete
substitute for the stronger G2 correspondence control. No scientific configuration
is changed to conceal this limitation.

Scientific protocol is frozen before full feature extraction and all training.
Exact metric implementation is verified before full extraction on unlabeled data.
Eigenvector undefined cases are explicit missingness, not silent zero fallback;
PageRank nonconvergence stops computation. Run caches are atomic and bound to
input contents, code and protocol. Solver implementations use deterministic exact
algorithms with one BLAS thread per worker, four workers, no approximate centrality.

Any engineering issue is disclosed without changing the registered scientific
definitions. The preliminary benchmark environment lacked psutil; memory reporting
uses the Windows working-set query instead, before protocol-bound extraction.
No model outcome existed and no training/model parameter changed for this repair.
