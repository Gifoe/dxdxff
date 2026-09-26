# Representation bottleneck audit (seed 42)

Development-only, frozen-source audit of the A1 patient-equal weighted BCE model. Audits A/B/C test temporal ordering, recruitment rank and cross-seizure persistence using frozen A1 representations and matched 15-epoch residual probes. Audit D retrains matched four-view ablations. Audit E stratifies A1 VLOO failures without training. No outer test is authorized.

The exact deterministic protocol is in `PROTOCOL_LOCK.json`. Patient-level caches and checkpoints remain private on the server. Aggregate conclusions will be recorded only after all five audits finish.
