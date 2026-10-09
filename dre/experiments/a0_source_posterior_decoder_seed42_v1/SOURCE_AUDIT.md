# Historical review and actual source contract

Reviewed before implementation: original PLLR A0, temporal CoTAR T1/T2, TabM
at d0277ae6a9fb888167157dddd3371ae06384bc6e, patient-identifiability at
cc22d0194e70e9c4185a7b2110c5f86f5274d7a1, direct_patient_shift_oof,
optimal_decision_cardinality_oof, active_fewshot_patient_calibration, and
crosspatient_feature_geometry_audit from its archived branch. Their reports and
protocols were inspected, not just summary metrics.

Current A0 development .63807978285 is not the older B0 development .628380
used by direct shift/cardinality, nor historical outer .616167. Direct shift
and cardinality did not transfer (.591867/.584579). TabM has small secondary
ranking gains but no primary F1 benefit. T2 CoTAR F1 gain .002690 did not beat
its local control. PLLR F1 gain .000144 traded sensitivity for specificity.
Patient-identifiability's exhaustive label-using oracle .70628499010 gives
.06820520725 headroom, but B8 and half-support probes did not reliably recover
decision F1. A1 supervised local geometry/few-shot findings use different
models/protocols and cannot establish A0 zero-label posterior identifiability.

The new hypothesis is therefore unestablished. It differs from learned
threshold/TopK regressors: the scorer jointly trains operational-source low-rank
parameters, while decision inference assumes FIT-OOF score-density likelihoods
with an unlabeled patient mixture. The fixed-prior control tests whether MAP
supplies anything beyond source prevalence. This is not evidence in advance
that Gaussian mixtures are valid or the oracle is deployably attainable.

Actual authoritative labels come from Task1 canonical patient indices built
by the original cached acquisition adapters, not Task2 center_clinical_target.
The prior clinical-source audit verified all7,635 export/index and cross-run
EZ-union labels;22 LZU first-record differences reflect union aggregation,
not grounds for relabeling. HUP uses composite SOZ/resection/status cues,
Multicenter actual metadata supports SOZ, LZU spreadsheet clinical annotations,
Pediatric workbook-backed clinical target. Independent HUP SOZ/resection fields
were unavailable. Historical command flags/code versions were not embedded
in the cache; current adapter hashes are not execution-time attestations.

Known source is an acquisition category, never patient ID, surgical outcome,
resection variable or biological class. No source-specific VAL prior or tau,
target-label inputs or unseen-center claim is admitted. All canonical cohort
members and labels remain unchanged. Full NPZ content is SHA-gated but its
all-cohort `y` member is not materialized: only sealed original FIT/VAL labels
are used. Raw features are not re-extracted. No banned extraction skill is used.

Original network/training source `uas_core.py` SHA256 is in the lock. Legal
PR-UAS teachers exclude queries from both training and internal selection;
their TRAIN-only preprocessing and all private content hashes must pass anew.
Existing MC means are not reused; fresh eval logits are required. D2 OOF
teachers use this experiment's exact source model and training objective.
