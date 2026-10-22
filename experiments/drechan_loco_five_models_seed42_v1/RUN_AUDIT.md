# Execution audit

- Scientific plan: user-approved five-model, four-source LOCO; seed42 only.
- Parent: completed no-guidance DREChannNet branch; no trained artifact reuse.
- Privacy: code and anonymous aggregate outputs only; private artifacts stay server.
- Source d7908a429b2ad099a27c6f27ee19d927421e13dd pushed/remote-verified before any formal fit.
- Actual Git archive CPU tests passed; deployment archive SHA256
  d957dc068cd07a3a97bce68809e5ab126f8c731f33d36ec73990e4775d87dfe8.
- Admission01 stopped before formal epochs: missing torchvision/Pillow in the
  otherwise unchanged existing PyTorch2.11.0+cu128 environment. No test inference.
- Existing server torchvision0.26.0+cu128 requires exactly torch2.11.0; copied it
  and Pillow12.0.0 into this experiment's isolated dependency directory. All323
  nongenerated files were verified byte-identical to the installed source. Tree
  SHA781dfd752d3b9fc8467d9391242dab931f91beb4d945faea84864f80df59f9bc.
- Original environments, parent model files, model math and budgets unchanged.
  Failed admission source/input/roles/logs retained. Formal training still pending.
- Connection: pinned Ed25519 authentication restored2026-10-22; no security checks disabled.
- Further admission, source-push, production and verification facts will be recorded
  from actual runs, never inferred from intended commands.

## Admission02, actual server run

- Source/environment correction43fafe12bd20a3703c68d19968ac26895d597e61
  pushed and remote-verified before fitting. Deployment archive SHA256
  3e0c482ec0892d0564786133647b1d2444dfc35c4fad51fe4737bcdb83171e90.
- Started2026-10-22T23:44:20+08:00; owner completed successfully. PASS3 CPU
  tests,16 scratch initializer checks with original initializer/state equality,
  matched head slices, nonFIT preprocessing perturbation invariance, synthetic
  OOF-SVM calibration and4 complete actual legal FIT-patient backward probes.
- TimeConv direct/cache logits and loss exact; max absolute gradient difference
  9.313225746154785e-10. All BN buffers and CPU/CUDA RNG exactly preserved.
-80patients/7635channels/256seizures/1471965valid windows verified using existing
  banks, no Raw copies. Held-source channels3667/2141/925/902 respectively.
- Explicit4-file public proof archive SHA256
  e91026e1fd44f8e8f195ce43779156db476ff1297d6762ed6ddd5634be6794d1.
  Whole-file hash verified on download; admission scientific hashes match local.
- No formal model/optimizer checkpoint or test prediction has yet been created.

## Production01 launch

- Started2026-10-22T23:48:02+08:00 under sustained foreground SSH owner.
- Command: run_all.ps1 -RunTag production01 -SourceCommit
  9401bb8c73d0e3524bb110928ee9a848d617b51f -StopAfter All.
- Source/admission commit9401bb8 was pushed and remote-verified before launch.
  Later0123dde7d4286a9b2e0f8bb33660a0851c556e8a changes only Git attributes
  and byte-exact public-proof line-ending preservation, not scientific files.
- Initial worker20256 (venv child13172). First actual HUP-held-out Raw+Feature
  epochs1..5 completed, observed first4.198s and subsequent3.25..3.30s/epoch;
  no stderr. These timings do not predict TimeConv throughput or total duration.
- No test outcomes read. Completion remains pending until all20 freezes, all20
  seals,400 patient rows/3600 metrics and independent interval checks pass.
