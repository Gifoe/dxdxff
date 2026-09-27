# Implementation audit

- A1 source modules/cache/split and protocol hashes are checked before execution. All 150 source validation grids replay within 1e-6.
- Ordinary A1 forward does not expose `contextual_channel_embedding` in this branch; a pre-hook on the original classifier captures exact R4, verified by direct classifier logit replay. D=64.
- A1 is frozen/eval; only adapter parameters enter AdamW. Zero-init P1/P2/P3 are identity. P2 and P3 have identical architecture/parameter count.
- Context/query assignment is label-blind, deterministic by seed/fold/source epoch/adapter epoch/subject hash; only query labels enter A1 weighted BCE. P3 teacher fits FIT patient labels only.
- The frozen A1 patient-attention stage forms each R4 from all patient channels before the adapter context/query split. Thus query *features* can affect context R4 indirectly; no query labels enter context. This is a representation-level dependency inherent to the mandated frozen R4 interface.
- Adapter epoch 20 is fixed; source epoch and threshold use exact 13-patient VLOO. Wrong-context and subset controls are never selection inputs.
- Each fold/source-epoch/variant saves only private resumable state, validation grids and patient arrays. Public outputs are aggregate only.
- Same-source-epoch patient coefficient pairs are used for pairwise cosine; vectors from independently trained source epochs are not compared directly. Numerical noncollapse means coefficient component std exceeds 1e-6; the +0.005 correct-vs-shuffled AP gate supplies the material-effect test.
- The inherited monolithic cache loader materializes all 80 patient labels at initialization, before role filtering. This violates the literal no-outer-label-read rule, although no outer loader/prediction/metric/training/selection is performed. Classify this run as development-only/non-sealed.
