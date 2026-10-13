# COVE seed42: complete development study, no useful interaction demonstrated

| Model | Parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| B0 authoritative compact F0 | 8,817 | 0.716569 | 0.508559 | 0.642067 | 0.433516 | 0.676591 |
| B1 additive conditional innovation | 8,845 | 0.715985 | 0.509518 | 0.642241 | 0.437107 | 0.674484 |
| B2 incorrect-pair interaction | 8,845 | 0.716554 | 0.508527 | 0.642067 | 0.433516 | 0.676591 |
| B3 correctly paired COVE | 8,845 | 0.716295 | 0.512349 | 0.642981 | 0.439535 | 0.689704 |

Primary unit: equal65 validation appearances,47 unique patients, all original five folds. These are actual completed predictions. All15 new model-fold cells ran. **No new outer TEST inference or evaluation was performed.** Exact terminal: `COVE_DEVELOPMENT_GATE_FAILED`.

## What the feature audit establishes

1. The12 retained delta/theta-band constant-zero positions contain no varying numeric information in the frozen cache. Eight retained auxiliary DELTA copies contain no additional measurement beyond their main coordinates. Neither observation warrants a biological claim that these frequency bands are useless; duplicates can also affect the fixed LayerNorm geometry.
2. Beta/low/high-gamma **DELTA views** have numerically tiny raw cache values and unstable residual availability at the current interface. In validation,93.85%–100% of their appearances trigger the original Patient-Z SD≤1e-8 denominator1 rule. Their COVE correction groups have no observed AUROC/AP/Macro-F1 effect when individually disabled. **This does not establish that the original beta/gamma ABS or RATIO inputs are useless.**
3. RMS/variance residual products make only1e-5–2e-5 AUROC contributions in the fixed B3 backbone, entropy approximately6e-5–7e-5; line-length products slightly reduce AUROC/AP. None approaches the desired0.010 increment. These are descriptive correction-removal interventions, not newly trained original-feature ablations.
4. Residual variation remains after linear prediction, including in RATIO. But the matched four-arm study and FIT-only probe do **not** demonstrate that correctly combining these residuals helps localization. No feature subset was picked or retrained after observing this result.

The complete14-group table, actual raw-scale diagnostics, original normalization-rule incidence, cross-fit/full-FIT drift, residual stability and FIT probe are in [CONDITIONAL_INFORMATION_AUDIT.md](CONDITIONAL_INFORMATION_AUDIT.md). This is a more specific conclusion than declaring all nonzero residual features useful, or deleting features on the basis of learned coefficients.

## Secondary development statistics

| Model | NEZ-F1 | Balanced accuracy | Top1-EZ | Sensitivity | Specificity | Accuracy | Predicted EZ fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.850618 | 0.669667 | 0.553846 | 0.464334 | 0.875001 | 0.779216 | 0.191949 |
| B1 | 0.847376 | 0.671118 | 0.553846 | 0.475687 | 0.866550 | 0.776777 | 0.200881 |
| B2 | 0.850618 | 0.669667 | 0.553846 | 0.464334 | 0.875001 | 0.779216 | 0.191949 |
| B3 | 0.846427 | 0.672173 | 0.569231 | 0.479944 | 0.864403 | 0.776316 | 0.202828 |

B3 increases sensitivity by0.015611 but loses0.010599 specificity, exceeding the registered0.010 maximum loss. Macro-F1/AP point changes are small and not supported as a general utility gain. Mean confusion counts are in the CSV; they are patient means, not pooled counts.

Secondary equal47-ID estimator (average each ID's repeated appearances first): B0/B1/B2/B3 AUROC =0.723153/0.722630/0.723107/0.723482; AP =0.521982/0.523401/0.521911/0.527133. This changes the estimand, not the primary gate. It must not rescue the failed65-appearance criterion.

## Paired evidence

Patient-ID cluster bootstrap,10,000 common draws, seed42; repeated validation appearances travel with their sampled ID. Epochs, thresholds and nuisance models stay fixed. These are unadjusted exploratory intervals, not independent confirmation.

| Comparison | ΔAUROC [95% interval] | ΔAUPRC [95% interval] | ΔMacro-F1 [95% interval] |
|---|---:|---:|---:|
| B3−B0 | -0.000275 [-0.004780,0.004999] | +0.003790 [-0.001991,0.010278] | +0.000914 [-0.003854,0.006456] |
| B3−B1 | +0.000310 [-0.003879,0.005337] | +0.002831 [-0.001555,0.007651] | +0.000740 [-0.004042,0.005907] |
| B3−B2 | -0.000259 [-0.004789,0.005081] | +0.003822 [-0.001974,0.010354] | +0.000914 [-0.003854,0.006456] |
| B1−B0 | -0.000584 [-0.002129,0.000746] | +0.000959 [-0.001543,0.004136] | +0.000174 [-0.002235,0.002589] |
| B2−B0 | -0.000016 [-0.000188,0.000167] | -0.000032 [-0.000216,0.000127] | 0 [0,0] |

The AUROC interval for B3−B0 includes zero. Correct matching does not beat wrong matching in the independent trained comparison, and its paired interval does not have a positive lower bound. All required EZ-F1/MRR/sensitivity/specificity contrasts are provided in `PAIRED_BOOTSTRAP.csv`.

## Fold and center consistency

| Fold | B0 AUROC | B1 | B2 | B3 | B3−B0 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.707185 | 0.707272 | 0.707108 | 0.706851 | -0.000333 |
| 2 | 0.728455 | 0.728300 | 0.728275 | 0.728455 | 0 |
| 3 | 0.702028 | 0.702084 | 0.701924 | 0.702143 | +0.000115 |
| 4 | 0.760672 | 0.761070 | 0.760954 | 0.760936 | +0.000264 |
| 5 | 0.684507 | 0.681199 | 0.684507 | 0.683088 | -0.001420 |

B3 improves in2/5 folds, not4/5. Its largest AP change is concentrated in fold5, where the selected epoch differs substantially from F0 (COVE12 versus F04).

| Center | Appearances / unique IDs | B0 AUROC | B3 AUROC | B3−B0 |
|---|---:|---:|---:|---:|
| HUP | 28/21 | 0.785366 | 0.782219 | -0.003147 |
| LZU | 14/11 | 0.645195 | 0.647530 | +0.002334 |
| Multicenter | 11/9 | 0.810274 | 0.808805 | -0.001469 |
| Pediatric | 12/6 | 0.553417 | 0.557895 | +0.004478 |

COVE beats both mechanism controls in AUROC in only LZU/Pediatric, not all centers. Small repeated center cohorts do not establish generalization. Channel-count strata use fixed≤64,65–128,>128 cutoffs; all strata and their counts are reported without using them for selection.

## What caused the small prediction changes?

See [MECHANISM_AUDIT.md](MECHANISM_AUDIT.md). Full B3 versus correction-disabled B3 differs by only+0.000130 AUROC, with an interval spanning zero. Disabling corrections changes296/352,239 within-patient pairs (0.08403%), no Top1 decisions and no MRR. Most AP point gain over B0 and all MRR/Top1 gain persist with corrections disabled. Wrong correspondence slightly increases frozen B3 AUROC. These results do not support a useful correctly paired interaction mechanism; ordinary joint optimization/checkpoint selection remains a plausible explanation for most differences.

## Registered continuation gate

| Condition | Actual | Required | Status |
|---|---:|---:|---|
| B3−B0 AUROC | -0.000275 | ≥0.010 | FAIL |
| B3−B1 AUROC | +0.000310 | ≥0.005 | FAIL |
| B3−B2 AUROC | -0.000259 | ≥0.005 | FAIL |
| B3 AP | 0.512349 | ≥0.508559 | PASS |
| B3 Macro-F1 | 0.642981 | ≥0.642067 | PASS |
| B3 EZ-F1 | 0.439535 | ≥0.428516 | PASS |
| B3 specificity | 0.864403 | ≥0.865001 | FAIL |
| Positive AUROC folds | 2 | ≥4 | FAIL |
| Correct trained pairing beats wrong | No | Yes | FAIL |
| Engineering/nuisance/leakage/independent verification | PASS | PASS | PASS |

## Answers to the15 research questions

1. **Conditional predictability:** heterogeneous, not uniformly strong. RMS RATIO is relatively predictable; gamma RATIO is poorly predicted. The14-group OOF table prevents conflating them.
2. **Residual variation:** all14 FIT OOF groups have nondegenerate measured coordinates under the fixed numeric gate, but beta/gamma DELTA distributions are unreliable at validation scale. "Meaningful physiology" is not established by variance alone.
3. **RATIO after ABS+DELTA:** nonzero variation remains in every group; its predictive usefulness is not thereby proven.
4. **Patient/center stability:** not universal. Band-DELTA has pronounced OOF/full-FIT/validation scale differences; full patient/center diagnostics are disclosed.
5. **FIT interaction probe:** no AUROC/AP benefit for correct products; wrong products perform at least as well. Not a conditional-information estimator.
6. **COVE versus F0:** no primary AUROC improvement; small AP/F1 point changes have intervals containing zero.
7. **At least+0.010 AUROC:** no,−0.000275.
8. **Versus additive:** +0.000310 AUROC, not the required+0.005 and not statistically resolved.
9. **Correct versus wrong:** no AUROC advantage;−0.000259.
10. **Across folds/centers:**2/5 positive folds,2/4 positive centers; not robust.
11. **Sensitivity/specificity:** sensitivity increases, specificity loss breaches the gate. Not a general localization gain.
12. **Module dependence:** numerically nonzero but tiny; disabling leaves Top1/MRR unchanged and AUROC effect unresolved.
13. **Residual-specific versus ordinary reparameterization:** attribution is not established; most observed gains survive disabling the module on the trained backbone.
14. **View-specific mechanism without independent-source claims:** not supported by these controls. All outputs remain deterministic functions of existing inputs, not new measurements or independent physiological sources.
15. **Potential JBHI method contribution:** this run supplies a negative development result and a useful feature-quality diagnostic, not a validated new localization method.

## Completion, provenance and stop

F0's frozen development scores are reused after checkpoint/hash/metric verification. All15 new cells completed144 epochs under the unchanged trainer; summed measured training invocations were18.02 seconds. Preparation, environmental diagnostics, probes, interventions and verification are additional work. Feature archive/order/split hashes, original legal banks, mapping/nuisance freezes, code/test binding and prediction seals remain private/hash-bound. The cohort's256-seizure contract is inherited from the historical audit; no raw seizure re-extraction was performed here.

The Python3.12 native preparation failure and outcome-free validated move to Python3.13 are documented in `IMPLEMENTATION_AUDIT.md`; neither changed science or discarded completed clinical training epochs. Primary independent verification covers1,430 metric cells and all fixed contrasts. Additional score-only attribution verification covers1,170 cells. Extra label-free feature-scale/floor diagnostics were generated after the model study solely for interpretation; they changed no fitted model, feature or gate and do not select a follow-up subset.

Only code, protocol hashes and non-identifying aggregates are public. Labels, patient/channel IDs, input banks, nuisance parameters, weights, optimizer/RNG states, individual predictions and logs remain on the trusted server. The validation cohort has been reused extensively, and historical outer outcomes were previously viewed in other studies. These are exploratory development results, not new participants or independent confirmation. No new outer inference, threshold retuning, larger module, selected feature retraining, new seed search or COVE-v2 is performed.
