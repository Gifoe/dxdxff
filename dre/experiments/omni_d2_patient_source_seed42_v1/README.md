# Omni-D2: patient-relative source-conditioned frozen TimeConv, seed42

Completed on the original Windows server. All 124 patients, 137 interictal
records, 10,202 labeled canonical patient-channels and five original folds
were preserved. No CNN retraining, Ictal weight transfer or pooled Task-2
evaluation was performed.

Scientific terminal: `SOURCE_SHUFFLE_CONTROL_NOT_BEATEN`.

## Frozen outer results

Every metric below is patient-equal. Each arm used its own fold-specific
threshold selected exclusively on outer-FIT patients; fixed 0.5 results are
reported separately.

| Configuration | New parameters | AUROC | AP | Macro-F1 FIT threshold | Pathological F1 FIT threshold | Macro-F1 at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| C0: original frozen TimeConv | 0 | 0.762163 | 0.477255 | 0.610352 | 0.313503 | 0.545188 |
| C1: shared patient-relative direction + source biases | 35 | 0.811823 | 0.523527 | 0.655819 | 0.392346 | 0.644327 |
| C2: C1 + rank-2 source directions | 105 | 0.807748 | 0.520349 | 0.655314 | 0.390798 | 0.640308 |
| C3: matched C2 with shuffled operational sources | 105 | 0.806116 | 0.516928 | 0.657716 | 0.393288 | 0.651207 |

The shared correction is the strongest ranking result. The source-specific
direction does **not** provide the registered incremental gain:

| Contrast | AUROC delta | Paired patient-bootstrap 95% CI | Positive folds |
|---|---:|---|---:|
| C1 - C0 | +0.049660 | [+0.019098, +0.080574] | 4/5 |
| C2 - C0 | +0.045585 | [+0.015300, +0.077009] | 4/5 |
| C2 - C1 | -0.004075 | [-0.011148, +0.003201] | 2/5 |
| C2 - C3 | +0.001632 | [-0.005742, +0.009002] | 2/5 |

Intervals use 10,000 paired unique-patient resamples, seed42, with frozen
models and numeric thresholds. Removing C2's source direction actually
increases patient AUROC by 0.001801. Removing source bias leaves ranking
unchanged but reduces matched-threshold Macro-F1 by 0.011657. Thus the data
support a shared patient-relative correction more than the proposed
source-direction mechanism; classification bias gains are not ranking gains.
The registered source-direction continuation gate fails. No joint fine-tuning
or post-result rescue was run.

C2 - C0 AUROC effects: HUP +0.025189 (n=10), Open-iEEG +0.052813 (n=103),
SourceSink -0.003556 (n=11). Gains are mainly in Open-iEEG; the two smaller
sources do not establish source-general transfer.

## Integrity and scope

- Fresh C0 scores reproduce historical predictions exactly: maximum error 0.
- 620 fold/patient representation cells were sealed; all five CNNs remained unchanged.
- 35 engineering checks and 15 real FIT-only head smoke tests passed.
- All 45 inner head runs and 15 selected-epoch refits completed.
- All 20 model/threshold selections were hash-locked before head outer prediction.
- 496 outer model/patient prediction cells were sealed and verified.
- Original inputs, CNN checkpoints and archived predictions remained byte-identical.
- Unknown good channels enter label-free context, never supervised loss or metrics.

Head inner-validation patients were used in training the corresponding frozen
outer CNN. This is head-only inner OOF, **not** full-pipeline inner OOF. Outer
patients are excluded from both the CNN and head fitting. The cohort was viewed
in previous research, so these results are exploratory repeated-cohort evidence,
not a new blind test. They cannot be compared as if equivalent to published
pooled EDF-channel AUROC 0.8061.

## Reproduction and artifacts

The source audit pins Experiment A, the operational server CNN wrapper and the
Ictal D2 mechanism. `PROTOCOL_LOCK.json` fixes inputs, initialization, losses,
selection, statistics and gate conventions before outcomes.

Execute on the original server with its existing authorized private inputs:

```text
code/prepare.py
tests/extraction_smoke.py
code/extract.py
code/validate_bank.py
tests/engineering.py
code/train_heads.py
code/evaluate.py
tests/completion_verify.py
```

Each script is run with `E:\Anaconda\envs\persist_stable_251\python.exe`.
Extraction used the baseline-compatible PyTorch 2.8.0+cu128 runtime and RTX5090;
the small head and statistics used CPU float32. Extraction took approximately
6 minutes 35 seconds. Sealed extraction/head/outer caches permit engineering
resume only; they do not authorize new scientific evaluations or selection.

See [FINAL_REPORT.md](FINAL_REPORT.md) for all 26 requested answers,
[results/OOF_PATIENT_SUMMARY.csv](results/OOF_PATIENT_SUMMARY.csv) for all ranking
and classification metrics, [results/PATIENT_BOOTSTRAP.csv](results/PATIENT_BOOTSTRAP.csv)
for every registered paired contrast, and `audit/COMPLETION_VERIFICATION.json`
for server artifact hashes. Fold, center, correction, intervention, error,
initialization and selection audits are also included.

Published: source code, protocol hashes and non-identifying aggregate outputs
only. Raw EEG, patient/channel identities, labels, embeddings, predictions,
checkpoint tensors, optimizer state and runtime logs remain private on server.
