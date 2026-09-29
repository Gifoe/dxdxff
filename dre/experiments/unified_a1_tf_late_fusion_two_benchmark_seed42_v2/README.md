# Independent A1–TF late fusion (seed 42)

This branch is the frozen, exploratory follow-up to `unified_a1_tf_two_benchmark_seed42_v1`.
The training/evaluation ran on the original Windows server. `PROTOCOL_LOCK.json`
was copied to that server before the independent TF run; the official Omni test
was accessed only after `OMNI_FUSION_FREEZE.json` fixed the checkpoints, beta and
threshold. The test was not used for further parameter selection.

The TF scorer is in `code/late_tf.py`. It uses the audited prior STFT/cache
implementation and does not receive A1 embeddings, logits or descriptors.
Ictal A1 is the unchanged historical checkpoint; the Omni A1-only baseline was
independently retrained with the exact v2 training code and official inner split.
Thus the shared architecture/fusion topology does **not** imply identical A1
training status across benchmarks.

The run used only these server-side inputs:

- Omni official cohort/split and v2 A1 feature caches, plus the prior audited TF
  cache under `F:\Omni-iEEG`; N0 historical outputs were read only for paired
  comparison after freeze.
- Ictal original 80-patient fixed partition, raw-window/feature caches, and
  historical A1/VLOO source artifacts under `D:\nips-temp`.

Reproduction order: retrain Nbase through the unchanged
`omni_ieeg_a1_interictal_v2_seed42/code/train_v2.py` into a new runtime;
run `code/train_tf_omni_late.py`; run `code/evaluate_omni_late.py --phase select`;
freeze and run `--phase inference` once; run `code/train_tf_ictal_late.py` for
folds 1–5; run `code/evaluate_ictal_late.py --phase freeze`, then `evaluate`;
finally run `code/finalize_late.py`. Paths and frozen choices are recorded in
the protocol and compact audit outputs. Runtime checkpoints, raw caches,
patient/channel predictions and private target-selection rows are intentionally
excluded from Git.

The main result is in `FINAL_REPORT.md`. The terminal is `STOP_TF_ROUTE`:
ictal A1 is preserved, but the validation-selected late fusion worsens Omni
official-test Macro-F1 and AUROC. This run is not a fresh sealed confirmation
because earlier related outcomes had already been viewed.
