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
