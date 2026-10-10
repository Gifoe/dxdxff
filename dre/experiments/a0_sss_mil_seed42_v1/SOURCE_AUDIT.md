# Source and historical audit

The seven requested official files were read in full at
[SSS commit 00c5ee476df51b5f300ca7264ee818546a7d9188](https://github.com/xmootoo/sss-official/tree/00c5ee476df51b5f300ca7264ee818546a7d9188).
The official `sss/layers/patcher.py` was additionally inspected. Exact file
SHA256 values are in `PROTOCOL_LOCK.json`. The applicable upstream MIT notice
is retained verbatim in `SSS_MIT_LICENSE.txt`.

| Reference component | This adaptation |
|---|---|
| `sss/models/sss.py` | Local fixed-length encoder and channel-level grouping inspire the interface. No Monte Carlo dropout, latent mixer, recurrent sparse-context branch or alternative backbone is imported. |
| `sss/models/patchtst_blind.py` | Reimplemented minimum patch projection, deterministic positional encoding and two Transformer blocks. Output is a 32D embedding, not a supervised local-window prediction. |
| `sss/layers/patcher.py` | Uses `unfold` patching, but no synthetic endpoint replication: 31 patches of measured samples instead of the upstream 32 padded patches. This explicit difference is locked. |
| `sss/layers/channel_modules/ch_loss.py` | Preserves channel-level supervision concept. Does not repeat independent window targets or average supervised window logits. BCE is applied once to final patient-channel logits. |
| `sss/layers/channel_modules/channel_aggr.py` | Channel aggregation is replaced by masked attention within a run and a masked equal-run mean across runs, as specifically requested. No channel-latent mixing. |
| `sss/utils/dataloading.py` | Read sparse/equidistant/sliding-window and channel-grouping code. Do not reuse its channel split, label balancing, padding/truncation, artificial noise, external data or normalization fitting. |
| `sss/utils/datasets.py` | Variable-length grouping inspired explicit raw validity masks; do not treat the upstream padded sequence length as measured EEG. |
| `sss/jobs/exp/sss/args.yaml` | Read official example dimensions and channel-loss flags; do not import its seeds, optimizer, dropouts, dataset, training selection or splits. User-specified seed42 settings prevail. |

This is **SSS-inspired adaptation**, not an exact reproduction of the full
published SSS model or published performance. No upstream dataset is used.
The minimal implementation is independent code using PyTorch's standard
TransformerEncoderLayer; source attribution does not imply numerical parity
with the full upstream architecture.

## Historical evidence inspected before implementation

- Source-conditioned A0/D2 `code/model.py`, `train.py`, `prepare.py`, protocol,
  parameter audit, fold results and final report. Exact 9,221-parameter D2
  topology, 96D hidden state, rank4 source correction, clinical BCE weight,
  source L2 and original threshold tie semantics are retained.
- RawTiny final report, metadata audit and raw preprocessing: complete matched
  identity coverage does not establish clinical onset verification; earlier
  raw-only/hybrid scores do not substitute for this new S0 run.
- Temporal CoTAR final report and boundary masking: its engineered temporal
  summaries and frozen-logit residual failed to establish incremental gains.
  The new fusion instead jointly fine-tunes D2 hidden features with raw MIL.
- E3 initial blocker: a requested fixed `[-15,-5)` preictal crop intersected
  padding in one original run. This is not proof that the whole run is invalid.
  This experiment independently checks the available candidate window centers
  against actual measured intervals and keeps the original 256 runs.

The immutable feature/split/raw/cache and original checkpoint hashes are
explicitly verified. Archived D0 and D2 were not trained again.
