# FullPool privileged patient geometry distillation

Development-only, seed 42. The first gate asks whether a strictly patient-internal OOF FULLPOOL teacher corrects A1 on unseen FIT channels. Each scored FIT patient's teacher regularization is selected from **other FIT patients only**. If the predeclared Teacher gate fails, Student training is not run.

The zero-shot Student, if eligible, inherits the exact A1 architecture and is compared against an equally continued hard-label A1 control. Runtime caches, patient/channel records, OOF scores, checkpoints and raw data remain private on the Windows server. Only compact aggregate results and code belong in Git.

The inherited A1 VLOO selector has cross-validation-patient label dependencies, and the legacy loader materializes all 80 labels. Literal strict target-label sequencing is false; this limitation must be reported without implying target-result-guided Teacher or Student selection.
