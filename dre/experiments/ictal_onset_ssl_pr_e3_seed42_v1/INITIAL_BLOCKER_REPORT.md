# E3 initial server execution audit, 2026-10-08

Historical snapshot before the user's explicit one-record exclusion approval. This is not the amended experiment's final status.

**Terminal: `BLOCKED_RAW_WINDOW_PADDING`. E3 training has not started; no model metrics exist.**

The requested E3 is the supplied onset-pair VICReg pretraining followed by the supplied patient-relative CNN, without relation attention. The user's request was for E3 only. The attachment's E0-E3 factorial study and optional controls were not expanded into additional training jobs.

## Live server checks

| Check | Observed result |
|---|---|
| Original package tests | 11 passed, 9.74 seconds, on the private server |
| Package ZIP SHA256 | `c8aaa5a33482c9eed7971bbaa30d04e57bcfda616b4e7d01c25a5ef3beac8f3c` |
| Feature cache SHA256 | `9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087`, exact match |
| Raw cache SHA256 | `011d7ffaa55469c9f34259be04b7136a987d6d95f0ce9cc443a06751557db733`, exact match |
| Frozen cohort | 80 patients, 256 matched seizure records, 7,635 unique channels |
| Run-channel incidences | 24,995 |
| Source window incidences | 1,471,965 |
| Outer TEST sizes | 16 / 16 / 17 / 15 / 16 |
| Inner validation sizes | 13 / 13 / 13 / 13 / 13 |
| FIT sizes | 51 / 51 / 50 / 52 / 51 |
| Channel, shape, rate and finiteness check | No failure detected |
| Required preictal interval | `[-15,-5)` seconds, samples `[3750,6250)` |
| Required early ictal interval | `[0,10)` seconds, samples `[7500,10000)` |
| Invalid required crops | 1 seizure from 1 LZU patient, 104 channels |
| Invalid preictal samples | 1,000 / 2,500 samples per affected channel, i.e. 4 / 10 seconds |
| Invalid early ictal crops | 0 |
| Independent onset waveform confirmations | 0 / 256 |

The affected record's reported seizure onset is 11 seconds after the cached source segment begins. Its valid waveform begins at nominal relative time -11 seconds. Thus the first four seconds of the required `[-15,-5)` baseline are padding, rather than measured iEEG. Training with that pair would violate the attached no-padding contract and could introduce a recording-boundary cue into the onset representation.

The affected patient belongs to FIT in folds 1-3, TEST in fold 4, and validation in fold 5. Silently dropping only a training occurrence would therefore make preprocessing differ across roles. The audit stopped before export/training without altering any source cache, inclusion, labels, windows or folds.

## Provenance

`SOURCE_CODE_SUPPORTED`: the legacy constructor places the reported onset at the midpoint of the 60-second waveform and preserves a valid interval. This is sufficient to identify the padding failure against the declared coordinates. It is not an independent confirmation of the clinical onset annotation.

`INDEPENDENT_EDF_UNCONFIRMED`: zero original EDF waveforms have been replayed against a verified onset sidecar in this execution. The matched cache records expose onset/start/end and validity metadata but no direct ictal EDF path in the flattened record/sample fields. Existing quality-report or interictal-source paths are not proof of ictal onset. The path coverage counts in the provenance JSON include such non-ictal fields and must not be interpreted as EDF verification coverage.

Any eventual source-only run must retain this limitation. Under the supplied execution instruction, only development/validation is admissible with source-code alignment alone. Independent onset confirmation remains a separate gate for the corresponding stronger evaluation.

## Read-only repair feasibility

Both alternative ten-second baselines `[-10,0)` and `[-11,-1)` fit within the declared valid intervals of all 256 recordings. Changing the baseline for all recordings preserves cohort size but changes the physiological reference timing and requires a protocol amendment.

Excluding only the invalid seizure leaves 255 recordings while retaining all 80 patients and all 7,635 channels. This is the smallest data exclusion by count, but it changes the frozen 256-record contract and likewise requires explicit authorization. No alternative has been applied, exported or used for training.

## Experimental statuses

| Stage | Status |
|---|---|
| Synthetic/package tests | PASS, 11 tests |
| Frozen membership/cache recheck | PASS |
| Required raw-pair data contract | BLOCKED: one padded preictal crop |
| E3 SSL pretraining | NOT STARTED |
| E3 supervised validation | NOT STARTED |
| Exploratory outer held-out evaluation | NOT STARTED |
| Independent external testing | NOT RUN |

Patient Macro-F1, EZ-AUPRC, EZ-AUROC, EZ-F1, MRR, Top1 and bootstrap confidence intervals are **not estimable from an untrained model**. Historical baseline values must not be substituted for new E3 results.

Private details remain in `C:\ictal_onset_ssl_pr_e3_seed42_runtime\PADDING_RECORDS_PRIVATE.json` and `ONSET_PROVENANCE_PRIVATE.json`. The public aggregate audit is `DATA_BLOCKER_AUDIT.json`. The private server has no running E3 training job at this terminal.
