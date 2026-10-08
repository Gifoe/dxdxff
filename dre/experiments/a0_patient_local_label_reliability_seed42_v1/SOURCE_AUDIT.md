# PLLR source audit, before implementation/training

## Reviewed sources and evidence

- CalibRank `BASELINE_AUDIT.md`, `code/prepare_b0_embeddings.py` and original
  trusted-server `task1_baselines/patient_controls.py`: exact 88D
  `p2_matched_simple` PR-MLP, 8,817 parameters, NEZ=1, one patient update,
  FIT-only mean imputation/scaling and patientwise z; original threshold/epoch
  ties and seed rules are retained. Frozen inference will be replayed again.
- PR-UAS public commit `f6a6b72b9a17d228bf2978892fe7dc54857c4356` and identical
  local tree: hard-label A0 development F1 .6380797828499001 over 65 patient-fold
  validation cells / 47 unique IDs. Its OOF target correction failed continuation;
  four selected folds had not yet activated correction. No new outer test ran.
  Reuse verified feature export and exact model/preprocessing/metric/RNG helpers,
  not uncertainty targets, teacher predictions or its outcome-based selections
  for the new B1/B2 arms. New A0 training must match that development control.
- `codex/crosspatient-feature-geometry-audit-seed42-v1` report: information was
  present but patient geometry unstable. Within-patient supervised probes were
  nondeployable and not evidence of verified mislabeled channels. That audit
  disclosed full-cache outer-label materialization before filtering. Here, only
  the mandatory frozen historical replay/sealing stage handles the source export;
  formal development consumes sealed FIT/VAL files, with no outer channels.
- P23 `c30e429274_p23_noisy_ez_loss.py`: asymmetric EZ reliability, pseudo-labels,
  EMA/ranking/core penalties and a different objective. None is used. Archived
  `P23_LITE_STATUS.json` is `not_admitted` with missing expected outputs, not a
  completed performance result; report templates also state no P23 training.
- N5 `7dafafd1e5_n5_pu_rankcal.py`: EZ-only confidence reliability, PU/focal/ranking
  and calibration mechanisms. It is not PLLR. No completed matched A0 N5
  performance evidence was found in the inspected aggregate records; do not
  call it a demonstrated improvement or failure.

## New difference and limitations

PLLR is offline, FIT-only, label-blind PCA8 followed by within-patient labeled
leave-one-channel-out prototypes. It changes channel BCE multipliers from epoch1,
not labels, the network, optimizer, threshold policy or inference. B1 permutes
the same final normalized weights within each patient/observed class. This is
the required feature-correspondence control, unlike fixed smoothing in PR-UAS.
Class mass is a coefficient sum invariant; it does not guarantee equality of
realized loss/gradient contributions, because losses correlate with weights.

Original training initial weights were not archived and the historical wrapper
initialized before reseeding. Historical checkpoint inference is exactly
auditable; bitwise historical retraining is not claimed. New A0/B1/B2 share
explicitly seeded identical fold states, matching the recent PR-UAS A0 protocol.
No cross-protocol subtraction of historical outer F1 .616167 and development
F1 .638080 is permitted.

Geometric conflict is not clinical label error. Minority prototypes can be noisy
despite minimum support; the FIT-only bootstrap audits this before training.
The prompt leaves 'extremely unstable' and 'substantially worsen worst center'
unquantified. Conservative numeric interpretations are frozen in PROTOCOL_LOCK
before inspecting feasibility or validation outcomes, not selected afterwards.

## Runtime/safety

Use the already admitted isolated Python3.12/Torch2.11 runtime from PR-UAS;
do not modify its shared base environment. Native faults occurred previously
even across runtimes, with no established root cause. Preserve every attempt and
checkpoint; no blind infinite retry. Private source export/checkpoints/logs
remain on the server. No raw EEG extraction, raw-model seizure exclusion,
public individual records or user-prohibited extraction skill.
