# Macro-aligned learned Raw dispersion residual

Target: patient-equal Macro-F1>0.65, with the actual0.630493 Raw-only reference.
The preceding105-head-fit study and full fivefold evaluation found no gain
from refitting/conditioning only the classifier. This separate finite study
changes the Raw encoder and fine-tunes it under the original Macro objective.
No performance numbers are filled before the corresponding verified run.

The original two convolutions produce24 learned responses at30 positions in
each2-second waveform, then discard everything except the temporal mean.
`DISPERSION` adds `0.25*tanh(W*latentSD)` to that mean. W is24×24 with no bias
and starts atzero. `MEAN_RESIDUAL` has the identical576 extra parameters but
uses the mean instead of SD. `MEAN_TUNE` controls warm fine-tuning without
either new module. All share the same legal initial CNN and one classifier.
This is a hypothesis about learned response dispersion, not a handcrafted
physiological feature and not an established novelty claim.

Original pre-latent reference, two stages, cross-seizure mean/SD and complete
patient Z are unchanged. All valid windows and complete patient channels enter
the differentiable forward. No artificial physiological input or D1 branch is
accepted by the model. The64 unused historical head slots are literal zeros.

`PROTOCOL_LOCK.json` fixes all roles, initialization, learning rates, Macro
loss, epoch/threshold rules, racing and future final budget. Stage1 fits all
three controls/candidates on each first ownFIT path. Only an internally
advancing candidate and its matched comparator get two additional paths.
Maximum35 internal fits/700 epochs, not an open-ended parameter search.

Actual largest training-patient forward/backward, all15 Native CAL bridges and
four targeted source tests must pass before fitting. CAL bridges retain
probability<=1e-6 and every201-grid decision; original versus new deterministic
logit drift is recorded rather than called bit-equivalent. No window/channel
sampling or truncated-gradient shortcut is an admission substitute.

```powershell
& D:\raw_latent_dispersion_macro_seed42_v1\run_internal.ps1 -RunTag 01 -StopAfter Admission
& D:\raw_latent_dispersion_macro_seed42_v1\run_internal.ps1 -RunTag 02 -StopAfter All
```

Keep the foreground SSH owner alive and use new RunTags for diagnosed resumes.
Original checkpoints, scores, sources and logs are immutable; do not touch
unrelated processes. Code and anonymous aggregates are pushed, never raw,
identities, weights, labels, embeddings, individual scores or logs.

Internal completion cannot establish the target. A later fully frozen joint
fivefold evaluation is required and remains exploratory repeated-test because
historical Outer outcomes have already been viewed.
