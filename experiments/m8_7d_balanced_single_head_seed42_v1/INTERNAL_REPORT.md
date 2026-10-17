# Original7D single-head Development result

The locked choice is **C2_INTERACTION**, before any new Outer inference.
Ten actual new model-fold fits completed114epochs; five exact retained S3
controls were reused. No feature coordinate, source, cohort, Raw mask,
optimizer, loss, training budget, validation threshold rule or patient role
was changed. No new Outer has been run at this reporting stage.

These are65Development appearances of47people, not a full80patient test and
not an unbiased nested-CV result. The same Development selects epoch,
threshold and architecture. Internal correctness is independently verified,
not independent predictive confirmation.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 existing7D+Gate | .665884 | .755774 | .545569 | .471578 | .860191 | .791753 | .772025 | .676923 | -.021586 |
| C1 balanced | .672947 | .762133 | .562422 | .484627 | .861266 | .794016 | .790313 | .707692 | -.016709 |
| C2 balanced+interaction | .673076 | .762169 | .561626 | .484762 | .861390 | .794168 | .790300 | .707692 | -.016861 |

Both new models pass the prospectively fixed Development gate: Macro gain
at least.003 versus C0, with AUROC/AP loss no greater than.01. Their penalized
scores are.670995/.671122 versus control.663908; the exact rule selects C2.
No close-score tolerance or post-hoc preference for another arm is introduced.

Patient-cluster10,000draw seed42 Macro differences:

| Contrast | Difference | Nominal95% CI |
|---|---:|---|
| C1−C0 | +.007062 | [−.007918,+.023032] |
| C2−C0 | +.007192 | [−.007809,+.023140] |
| C2−C1 | +.000130 | [.000000,+.000414] |

The differences versus control are uncertain; the minuscule extra C2 change
does not establish the interaction's scientific necessity. Do not call the
Development values target attainment. All three arms and nine endpoints,
fold/source summaries, sample fold SD, coverage and54nominal intervals are
retained, not only favorable endpoints. All15 actual checkpoints were
replayed with maximum logit difference zero and exact201threshold decisions.
All1,755patient-metric entries and54intervals were independently checked;
maximum interval difference was zero. No private scores/identities were fetched.

Only a separately frozen complete80patient evaluation can establish the
requested numerical target. It will still be exploratory repeated-test because
the historical Outer cohort has been repeatedly viewed. The primary remains
C2 even if a secondary later obtains a higher Outer value.
