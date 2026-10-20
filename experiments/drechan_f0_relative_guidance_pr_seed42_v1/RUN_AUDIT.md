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

## Production start and release-only checks

At15:43:20Asia/Shanghai the sustained foreground production owner started the original-protocol training worker, with source-push proof `581396c836501b4ccba99c7ac136914ba74950e1`. That commit contains exactly the same17scientific source hashes as admission; the additional changes were anonymous admission evidence and documentation. Forty fresh fits are scheduled; the original five main reference cells are reused, not retrained. No new Outer prediction is allowed until all45selections are replayed and frozen.

The independent public-package validator is a release-only addition, outside the17frozen execution files. Seven positive/adversarial archive-validation cases actually pass, including rejection of corrupted content, traversal paths, identity headers, secret markers, private-export flags and unapproved members. These tests generated no benchmark metrics. The complete final package must still pass this validator after actual server completion.

## Actual terminal state

Production finished without scientific restart or source change. Forty fresh fits completed505epochs by16:04:07;all45selections passed replay andfroze by16:04:56;all45Outer cells sealed by16:05:55;independent endpoint/bootstrap verification completed16:06:29;the coordinator reported full completion at16:06:35Asia/Shanghai. All six production-worker stderr files are empty. The local SSH stderr reader had an idle-stream timeout warning,but the main connection andallserver workers continued to normal completion;no model,checkpoint orprediction was lost orrerun because ofthat transport-reader warning.

All five newly trained joint-NoPR controls exactly match their historical selected tensors,validation logits,epochs,thresholds andtotalepoch counts. This is an actually verified unchanged-control condition,not an outcome assumption. All80patients/7635channels/1471965validRaw windows pergroup are verified. Independent720patient-row/6480endpoint/162nominal/14family checks PASS,maximum interval discrepancy8.881784197001252e-16.

Private model freezeSHA256 `07725b81bda3816edc399d0e505f56580533f60df859b0a821dd35da714fed08`;pre-Outer freezeSHA256 `ad4579246ea7d636641ce387ded8b42dadb71b952ac5845c5c9160da1301073e`;prediction sealSHA256 `bd5f968c46bdfce99c6a1c585ef8f95290832d7b4648de594d9264793bc05746`.

The actual61010-byte public-only export has30members,including29explicit anonymous payloads andmanifest,andSHA256 `7641a9552dee43b041ca31422ea51906ceb1360d7b260d458a1506c3967a1a31`. It passed the separate local public-release checker;all29extracted payload hashes wereverified before repository staging. No private files were fetched.

The computational study is complete,butthe fixed stable-contribution objective is **FAIL**. Full Macro-F1 0.6463387559404331andAUROC0.7658883112922217meet thepoint-performance conditions. All three PR removals lowerbothpoint estimates,butinterval/fold/component checks do not establishstable two-metric benefit. Themodelor threshold wasnot changed afterthis result;no favorable control waspromoted. See `FINDINGS.md` andtheunaltered server-generated `results/GOAL_AUDIT.json`.
