# Historical evidence and actual mechanism overlap

Reviewed the listed implementation/protocol reports and relevant model/training
operators read-only. `HISTORICAL_REVIEW_HASHES.json` binds the inspected source
and reports. Three earlier studies absent from this checkout were inspected
on the trusted server. Historical results below are not new FCPA-LF runs.

| Prior study | Evidence and overlap | What it does not establish |
|---|---|---|
| D2 complete ablation | Exact source-free constructor removes V/u/source biases; 8,817 shared parameters. Original Patient Z, FIT-only preprocessing and objective remain. The manifest identifies 16 zero LF dimensions and 16 duplicate auxiliary dimensions. | Zero exported LF columns do not show that LF physiology is absent or predictive. |
| A0 PMA | Adds a zero-output hidden residual and a multiplicative gate driven by two input magnitude statistics. Its registered development gate failed against matched capacity. | FCPA's multiplicative residual topology alone is not a new demonstrated advantage; separately verified LF exposure is the potentially distinct question. |
| B1 no LayerNorm outer | Read the frozen amendment: only input LayerNorm was removed; no retraining or threshold changes in that outer replay. The task's no-meaningful-outer-gain conclusion is historical context. No private outer outcome was reopened here. | It is not the authoritative A0 and does not authorize normalization changes. |
| No Source + no ZDELTA | Registered 88D zero-mask ablation, 62 retained columns, preserved Patient Z, separate validation selection. Historical repeated-outer-informed combination. | It is neither a feature deduplication protocol nor a legal substitute baseline. |
| TabM | Source implements widened MLP and four-member BatchEnsemble; member probability averaging, equal patient updates, fixed validation selection. Report: TABM_NOT_SUPPORTED, no convincing primary F1 gain or ensemble-specific advantage over width. | More parameters do not isolate information gain. |
| Temporal CoTAR T1/T2 | Local versus masked temporal core, matched 5,797 trainable residual parameters, frozen A0. Registered T2 gate failed; small sensitivity gains accompanied specificity loss. | Temporal activity or a nonzero residual is not localization utility. |
| Representation bottleneck | FIT-only trajectory/recruitment/persistence probes with matched shuffled/magnitude controls on frozen A1; their registered gates were not supported. | Existing A1 trajectory tests do not test a verified new LF24 feature bank on source-free A0. |
| Ranking bottleneck/source | Read attention removals, pairwise BCE/ranking operator, FIT center weights and semantic/random feature partitions. All registered bottleneck gates failed. | A ranking objective or center conditioning is not a warranted rescue option. |
| D2 GraphRole | Dynamic graph feature addition with matched correspondence and neutral-input controls. Original LayerNorm mixes the extended input; neutral zeros are not an inactive branch. Report does not establish graph-specific increment beyond its controls. | Graph input use does not demonstrate useful new physiological information. |
| RawTiny | Existing250Hz waveform exposure, context and residual models; exact coverage but source-supported, not independently confirmed EDF onset/filter provenance. | Its coverage and nonfinite checks cannot certify LF fidelity. |
| A1 conditioned low-rank adapter | Rank-four correction on frozen64D A1 R4, patient mean/std hypernetwork, wrong-context controls. Registered context/teacher gates failed; legacy label materialization limitation disclosed. | Patient-conditioned directions are not supported; this differs from a channel-level LF/HF gate but is relevant negative context. |

## Authoritative baseline, not a substitute

The legal B0 is source-free `w/o Source`: LN88 → Linear88×96 → GELU →
Dropout0.15 → Linear96×1. It outputs a NEZ logit; EZ=0. The actual source-free
factory and original trainer were inspected, including original shared-state
initialization, FIT-only class weights, one update per patient, seeded update
order, AdamW, clipping, early stop and Macro-F1/EZ-F1 selection. PMA's frozen
development reference agrees with the task's A0 numbers. **No new B0 replay
was run because Stage A precedes any new predictive evaluation.**

## Remaining question and limits

The genuinely untested input question is whether an independently verified,
complete1–4/4–8/8–13Hz bank carries incremental patient-local information
missing from the frozen88D representation. Ordinary LF exposure must first
beat A0 and its capacity control. Only then can retrained direct/additive/
multiplicative/shuffled controls attribute a possible interaction gain.

The proposed gating and low-rank hidden-residual machinery overlap substantially
with existing conditioning/residual methods. Renaming that machinery would not
establish novelty or utility. Neither large LF power nor an active module is
evidence of EZ localization improvement. No causal physiology claim follows
from any of these historical discriminative studies.
