# Actual pre-training execution audit

2026-10-20, Asia/Shanghai. This document records executed engineering admission, not model accuracy.

- Scientific source was pushed and remote tip verified at `70c9a27980ce213c48364977ea84b9a47782c771` before admission/formal training. The19-file Git-tree source archive SHA256 is `4fe29b39ea7e77e3e9fe6417e75a208301e73c573cc477bfcbfc67c96d0ef0de`; server upload verified this hash before extraction into a new dedicated D-drive namespace.
- All17scientific-source hashes in the actual server admission match local committed source byte-for-byte. No original scientific source or historical checkpoint was modified.
- Nine unique arithmetic/model/reporting tests pass on local CPU and on the real server runtime. Synthetic tests are not experimental performance results.
- Five original fold initializers, preprocessors and selected validation logits are bit-exact. Original FIT/validation/Outer role counts remain unchanged; no additional performance eligibility gate is imposed.
- On the label-independently chosen largest valid-waveform-support patient in original fold2FIT, the unchanged no-PR procedure has bit-identical logits, loss, all gradients, clipped one-step Adam update, optimizer state and RNG state versus the original implementation. These two parity updates were discarded.
- All eight new paths each pass two complete-patient backward/optimizer steps on164channels/48380valid windows; all retained paths have finite effective gradients. All16admission updates were discarded. The admission artifact reports zero formal fits and no new Outer evaluation.
- The direct-summary preprocessing audit reports actual retained FIT/validation counts and maxima in `admission/PREPROCESSING_ADMISSION.csv`; no channel or patient was dropped. These numbers are preprocessing diagnostics, not label-prediction metrics.
- The user's existing password-authenticated entry successfully reached the expected server. No credentials, host-key settings or unrelated processes were changed; the previously failing extra Ed25519 adapter was not used. This confirms current connectivity, not an independent host-fingerprint attestation.
- Admission PASS is not a claim of Macro-F1/AUROC improvement, favorable ablation directions, seed stability or goal completion. Forty actual fresh fits,45legal selection checks and the sealed original80-patient fivefold Outer experiment still have to finish.

All private data, weights, per-patient/channel values and runtime logs stay on the trusted server. This repository contains only source, hashes and the four anonymous admission aggregates.
