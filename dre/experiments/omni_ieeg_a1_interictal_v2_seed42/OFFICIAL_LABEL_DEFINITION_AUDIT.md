# Frozen Omni channel ground-truth audit

Checked the exact public Omni code commit
`57c22a75a59b5c3a98006806ad42000f6a3fa5b6`, before v2 waveform
extraction or v2 test model scoring. Dataset revision is
`73b9c5180a57828ab2a83c040e7e9d112e77b2cc`.

The authoritative **evaluation** branch is
`omni_ieeg/channel_model/benchmark/evaluation_channel.py` lines 41–60:

- Lines 41–49 iterate the official filtered test EDFs, read patient outcome
  and the EDF's `channels.tsv`.
- Line 51 retains `good == 1` channels.
- Lines 56–60 initialize ground truth to `-1`; first assign **normal=1** if
  `outcome == 1 and resection == 0`; otherwise assign **pathological=0** if
  `soz == 1`; otherwise leave `-1` and exclude from metrics (line 130–134).
- Consequently normal has precedence over SOZ for conflicting annotations.
  The prompt's prose “SOZ still pathological” does not match this code.
  Reversing this priority would not be an exact Omni reproduction.
- The official script inverts its internal `ground_truth` to make pathology
  positive for metric calculation (lines 130–150), and selects a *test-derived*
  Youden threshold (lines 153–161). V2's primary threshold is instead locked
  from train-side validation; the Youden value is reproduction-only.

The separate **training feature-sampling** code is
`omni_ieeg/channel_model/channel_model_train/features.py`: lines 18–29 select
successful-outcome unresected good channels for its normal/“positive” examples,
with row filter `outcome==1 and has_resection` at lines 103–125. Lines 59–73
select good SOZ channels for its pathological/“negative” examples, excluding
unresected SOZ if outcome=1. V2 uses the evaluation ground-truth rule on the
frozen official train/test patient membership for its patient-equal training
objective and evaluation. It does **not** claim identical training-sample
construction to the official independent-channel CNN. The official evaluation
row filter excludes Multicenter at lines 283–288, then uses train/test split,
`frequency>900`, `interictal=True`, `length>=62`.

The metadata-only `code/audit_official_labels.py` replays that ordered label
rule against every filtered sidecar, checks the frozen split SHA and cache
revision, and detects patient-channel label conflicts. It found none.

| Scope | Train | Test |
| --- | ---: | ---: |
| Official-filtered patients / EDFs | 151 / 399 | 102 / 237 |
| Patients / EDFs with at least one official label | 141 / 296 | 96 / 174 |
| Labeled EDF-channel records | 13,350 | 8,104 |
| Pathological EDF-channel records | 1,355 | 807 |
| Normal EDF-channel records | 11,995 | 7,297 |
| Unique labeled patient-channels | 7,322 | 5,055 |
| Unique pathological patient-channels | 1,279 | 722 |
| Good but unclassified EDF-channel records | 12,665 | 8,439 |
| SOZ+normal-precedence conflicts (record count) | 185 | 66 |

The official-label positive prevalence is 1,355/13,350 = 10.15% for
training and 807/8,104 = 9.96% for official-style test EDF-channel
classification. Among unique patient-channels, test prevalence is
722/5,055 = 14.28%; those units are reserved for patient ranking, not
the official-style classification denominator.

Zurich now has **six train and four test patients** with official *normal*
labels from outcome/resection, although its SOZ is unknown. It contributes
zero pathological channels. All four eligible Zurich test patients, plus
four other test patients, have zero positives and are **not estimable** for
patient ranking; they remain in official-style pooled classification.

The counts are recorded in `audit/OFFICIAL_LABEL_COUNTS.json` and the
per-EDF `audit/OFFICIAL_COHORT_AUDIT.csv`. This is a different label
universe from v1's SOZ-vs-all-known-SOZ=0 experiment. In particular, a
v1-to-v2 metric difference cannot be attributed solely to the new reference.
