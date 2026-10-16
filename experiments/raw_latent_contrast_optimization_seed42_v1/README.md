# Raw latent contrast optimization — seed42

This is a new, pure-Raw study, separate from the completed physiological feature
selection research. The target is **patient-average EZ-positive F1 > 0.65**.
No result or improvement is assumed. Historical patient EZ-F1 around0.40–0.43
does not justify promising this target; Macro-F1 is not a substitute.

The fixed protocol is `PROTOCOL_LOCK.json`. Stage1 reuses20 genuinely
inner-cross-fitted Raw-only encoders and compares the complete16 combinations
of pre-onset latent subtraction, two temporal phases, between-seizure SD, and
full-patient latent Z. All genuine original Raw windows and complete channel
sets are retained. A pure96-input head is checked against the historical
160-input head with64 inactive physiology slots. Neither historical physiology
data adapters nor a D1 classifier are imported. Legacy mixed bundles are opened
only for identity/role metadata; their physiological arrays are not used.

Patient-level feature caches are atomic, private and bound to source, data,
encoder and identity hashes. Every completed patient must match the old baseline
embedding and logit within1e-6. A failed numerical gate stops execution; it is
not a reason to loosen tolerance or change the scientific protocol.

Stage1 fits560 fixed logistic screens on legal inner roles. These are **frozen
encoder probes**, not new end-to-end Raw CNN results. The16-way maxima are used
only for own-FIT recipe selection. They are not unbiased performance estimates.
Original Development and each model's own Outer labels are excluded throughout.

Stage2 is bounded to four matched arms and five folds: original pool/loss,
original pool/EZ-directed loss, selected pool/EZ loss, and selected pool plus
Raw-only phase/seizure attention/EZ loss. The attention layers start uniform
and add no manual physiological inputs. End-to-end implementation and admission
must be frozen before training. Final checkpoint and threshold selection is on
the original patient-held-out Development roles. All models, per-fold decisions
and thresholds must freeze before one sealed Outer pass. Outer has repeatedly
been viewed historically and can only be exploratory, not new confirmation.

## Execution

On the trusted server, use the existing Raw banks; do not copy/reextract them:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
E:\Anaconda\python.exe -m pytest D:\raw_latent_contrast_optimization_seed42_v1\test_model.py -q
powershell -NoProfile -File D:\raw_latent_contrast_optimization_seed42_v1\run_stage1.ps1 -RunTag 01
```

Keep the owning SSH command alive. Inspect process/log/status before any restart;
use a new RunTag and the same verified private cache after diagnosed engineering
failure. Never run duplicate workers. Full metrics, uncertainty and source
summaries will be derived only from executed runs. Publish code and anonymous
aggregates only; waveforms, weights, identities, labels and individual rows stay
on the trusted server.
