# FIT-only nonlinear confirmation and prospective M8 integration

This is an intermediate evidence report, not completed feature discovery or final model performance. No new Outer classification inference has occurred in this phase. The full objective remains active.

## Actual completed runs

- Three numerical controls were extracted from all80 patients/7635 channels/256 seizures/1471965 genuine channel-windows. They use stable log power from the *same pinned original* Welch bandpower definition, preserving band edges and the original partial-HG inclusive125Hz endpoint. They are not new signal recovery. Legacy window-delta replay maximum error was0; finite fraction1.
- Fixed nonlinear trees completed705 patient-isolated screens and520 held-FIT-center screens. All selection training/calibration/screen patients remain inside each original outerFIT. Original validation and own-fold Outer patients were excluded from feature selection.
- Independent verification replayed all1225 held/calibration probability pipelines, imputers, class-weighted initial priors and thresholds, checking18615 patient metric rows. Maximum probability replay error0. All five per-FIT feature decisions were independently reconstructed.
- Matched old7 neural necessity experiments are separate actual retraining, not tree importances or inference switches:16 fixed full/drop/single/none variants ×5folds, each with an independently fitted D1 and complete M8. Their running/completed state must be checked in the trusted runtime; this report does not claim they have all finished.

## Fixed per-FIT decisions

| Original fold | Old groups eligible for deletion | New groups formally eligible | Diagnostic new-only/top3 route |
|---|---|---|---|
| 1 | none | none | flatness, clipped kurtosis, centroid |
| 2 | none | clipped kurtosis | clipped kurtosis, partial-HG fraction, same-band stable-log partial-HG |
| 3 | none | none | normalized line length, restricted-bin slope proxy, Hjorth complexity |
| 4 | none | none | clipped kurtosis, permutation entropy, partial-HG fraction |
| 5 | none | none | same-band stable-log beta, restricted-bin slope proxy, clipped kurtosis |

Eligibility requires registered patient-screen improvement/non-inferiority, support across screens and all four FIT-center sensitivity gates. No global feature list was chosen from averaged validation/Outer results. A failed eligibility gate does not establish physiological uselessness; the diagnostic top3 routes remain explicitly secondary even if they later look better.

The frozen selection JSON SHA256 is `b8fbe6c059e83ae9ec7709be69d0fd372a7a8532b91d859f580c1612a6008be3`.

## Model integration fixed before new efficacy analysis

`M8_INTEGRATION_LOCK.json` fixes nine procedures: same-width old7 control, eligible pruning, primary eligible additions, diagnostic top3 additions, new-only, numerical bandpower replacements, D1-only new measurements, gate-only new measurements, and availability-matched wrong-channel pairing. Identical inputs/routes are genuine aliases, not additional independent fits.

The widened physiological model has2113 D1 parameters and9577 correction parameters:11690 total versus11018 legacy (+6.10%). Its normalization represents94 positions (original88 plus6), not native20D LayerNorm. The old7 same-width control is mandatory because changing normalization geometry itself can change performance.

Outcome-free CPU/GPU synthetic checks passed: full94-position float64 folding error1.7763568394002505e-15 including non-default affine LayerNorm; exact constructor CPU RNG consumption; zero original-Raw-prefix feature error on both devices; positive intended gradients after three discarded steps; frozen base. Real whole-FIT admissions, source bindings and all training completions must additionally pass before any final evaluation. Synthetic tests alone are not sufficient.

## Interpretation limits

The measured original RMS floor and patient-Z fallback motivate numerical controls, not selective repair of previously completed original models. No new LF/HFO/PLHG support is asserted. The slope is a restricted-band proxy, not an aperiodic exponent or E/I estimate. Early-onset permutation entropy is not postictal PEI.

No claim of improvement, necessity, general optimality or completion follows from these intermediate screens. Final comparisons require frozen models/scalers/feature maps/thresholds and independent metric verification. Historical Outer viewing means any final evaluation is exploratory repeated-test; conditional bootstrap intervals do not remove research reuse or training-seed uncertainty.
