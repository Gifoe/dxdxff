# EZ-conditioned reference gains and channel composition

This new exploratory supplement was designed after viewing the parent historical-
cohort results. It is not independent confirmation. `PROTOCOL_LOCK.json` is the
complete machine-readable numerical specification.

## Two experiments

E1 replays the exact100 frozen bidirectional channel halves and reference moments.
Reconstruct channel CRPS before class averaging. The primary H is mean-patient
[(GlobalNEZ-FullNEZ)-(GlobalEZ-FullEZ)]. Report all five arms' absolute class losses
and gains. H>0 means NEZ benefits more; it does not establish localization value.

E2 uses each channel as a held-out query. Delete that query from each existing
SHA256 ranking and use ceil(rho*(C-1)) reference channels at25/50/75/100percent.
Sets are nested;100percent still excludes the query. D is mean-patient Full25
minusFull100 CRPS. Report overall/class losses,H,absolute-relative gaps and support.
All80 patients,7,635 channels and12 original coordinates remain.

Reference EZ composition is annotated after freezing label-free references. The
slope per10percentagepoints uses repeated reference composition and Full CRPS,
centered within the exact same patient/query. Zero predictor variance means not
estimable,not slope0.100percent is structurally unidentifiable. Composition is an
association; counts do not measure anatomical implantation range.

## Inference and controls

Average repetitions within queries andqueries within patients before inference.
Equal-patient means and10000 fixed seed42 paired within-source bootstrap draws.
Two new primary endpoints have nominal95percent andadditional97.5percent family
intervals. Exactly11 secondary comparisons have additional Bonferroni intervals.
All sources/coordinates retained descriptively.1000 count-preserving label shuffles
give conditional E1 diagnostic distributions. No classification metrics or tuning.

Legacy absolute-atypicality tables are copied unchanged with the original six-
family intervals. New evidence cannot replace that inconclusive result or the
older unsuccessful transferable-separation probe. Nonsignificance is not equality.

## Reproducibility and privacy

Freeze code,inputs,all reference indices/moments before new endpoints. Seal endpoints
before independent replay. Test query isolation,nesting,Global/Source invariance,
100percent invariance,class-weighted reconstruction andlabel-swap antisymmetry.
Only anonymous aggregates andfigures leave the trusted server. Real completion
requires actual input admission andindependent PASS;offline synthetic tests are
reported separately. Never overwrite the parent or a failed runtime.
