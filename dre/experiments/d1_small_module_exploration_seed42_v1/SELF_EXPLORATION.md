# What to explore and how to interpret it

The fixed Phase1 command trains/evaluates the complete three-candidate matrix, not just an implementation smoke test. It is resume-safe but refuses a different model list under the same lock. This prevents silent candidate/threshold substitutions. For a different experiment, copy the source and protocol to a new named phase/runtime, change the prospective lock before training and retain the old results.

- **M1** asks whether meaningful scaled physiological gates add to Raw morphology. It is not a new backbone and does not change the already published A2 run.
- **M2** asks whether early/late physiology, which the original all-window R14 collapses, has value with only722 new parameters. No neural Raw encoding or independent onset verification is claimed.
- **M3** asks whether nonlinear per-window feature interactions/distributions have value with1354 new parameters. Within-stage ordering is deliberately discarded; no chronological mechanism claim.

Selection: original patient-held-out own-fold FIT training, validation MacroF1/EZF1/earlier epoch, then a common global candidate from the fully reported matrix. No architecture choice per winning fold. Never use the original Outer outputs to debug, pick modules or change thresholds. Original validation has been used repeatedly: treat all outcomes as hypothesis-generation.

Read `results/TARGET_SCREEN.csv` and `SCIENTIFIC_GATE.json` for actual two-point gain and AUROC/AP/EZF1/sensitivity side gates, not just the highest single metric. Patient bootstrap retains repeated identities; uncertainty does not remove model-selection optimism. Check corrected-scale clipping, residual magnitude, confused EZ/NEZ decisions and the exact residual-disabled identity before interpreting a gain.

The appropriate next decision depends on real results. If no candidate meets the target, do not label the goal achieved or inflate a smaller improvement into two points. A broad open-ended validation search is not independent evidence. Strong claims require a separately authorized genuinely unused evaluation after selecting/finalizing a model; this phase never opens Outer automatically.

## Completed best recipe

All three prospectively locked phases have now executed. M8 reaches the requested primary Development point target:0.666795 versus D1.644546 (+absolute.022248), with AUROC/AP/EZF1 increases. The2-parameter margin gate is not needed for this winner. Use `tools/best_recipe.py` to construct the exact zero-output Raw morphology correction and its fixed.75 weightedBCE+.25 patient SoftMacroF1 loss. Construction requires the original frozen1537-parameter D1; the forward interface remains original x/raw/mask/times/windowDELTA/FITscale. No label or patient-count input is required for inference.

`phase3/README.md` supplies the executed server command and environment paths; `tools/replay_best_development.py` is read-only and verifies saved selected Development predictions, not a new test. All five strict checkpoint loads and replays actually passed with zero logit drift. For a different objective, cap, module, seed or threshold policy, create a new named protocol/runtime **before** training rather than modifying a completed phase. Keep previous outputs and disclose adaptive selection. Do not call the current repeated-Development gain an independent generalization guarantee.
