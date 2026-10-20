# Findings: selected42/44/62 no-guidance ablation study

## Fixed scope and verification

The user selected this no-Early/no-Late six-feature main and these three seeds AFTER historical Outer viewing. These are outcome-informed subset sensitivity results on exploratory repeated-test patients, not independent validation or random-seed robustness. Variance and Head Interaction remain absent. The prior seeds43/52 are disclosed, not silently dropped from the historical record.

All255choices froze before newOuter; all255scores sealed before metric reporting. Fifty strict reuses and205fresh fits completed2405epochs, retaining all80patients/7635channels/256seizures/1471965validRaw250windows. Independent verification passed4080patient-seed rows,36720endpoint values,306nominal and32additional-family intervals (maximum discrepancy7.771561172376096e-16). Fourteen scientific source files remain unchanged after admission.

Primary results average each patient's endpoints over seeds42/44/62, then average patients. Bootstrap uses10000seed42draws of the same80unique patients after seed-metric averaging, not240independent rows. Fold, source, pooled and coverage tables are separate. No threshold, feature, arm or seed was selected from these new outcomes.

## Main and PR evidence

Main Macro-F1=0.643771 (seed sampleSD0.002070), AUROC=0.766660(SD0.006766), AP=0.570795, EZ-F1=0.451637, NEZ-F1=0.835906, ACC=0.760550, MRR=0.787698,Top-1=0.695833,EZ-fraction bias=-0.026136. Main nominal patient-bootstrap intervals:Macro-F1[0.616862,0.671263],AUROC[0.732656,0.801281]. Seed sampleSD is not patient uncertainty.

| Main-minus-control | Delta Macro-F1 | Nominal95%CI | Delta AUROC | Nominal95%CI |
| --- | ---: | --- | ---: | --- |
| No Feature PR | +0.022261 | [0.008070,0.036441] | +0.030188 | [0.010702,0.051429] |
| No Raw PR | +0.016594 | [-0.001174,0.034322] | +0.003073 | [-0.019542,0.026023] |
| No BOTH PR | +0.151140 | [0.113352,0.191416] | +0.016924 | [-0.005055,0.041670] |
| No self pre-reference | +0.026815 | [0.013005,0.040686] | +0.034958 | [0.012894,0.057520] |
| Raw-only | +0.029066 | [0.015183,0.043296] | +0.037871 | [0.016967,0.060439] |

Across the32Macro-F1/AUROC primary comparisons, Bonferroni-percentile intervals retain positive bounds for both endpoints versus no pre-reference and versus Raw-only. FeaturePR AUROC remains positive [0.000745,0.067529], but its Macro-F1 family interval[-0.001720,0.044919]includes0. BothPR Macro-F1 remains positive[0.091130,0.218461], while its AUROC family interval[-0.017140,0.055819]includes0. RawPR alone has uncertain benefit.

The two-PR removal produces Macro-F1=0.492631 but AUROC=0.749737: the large classification decline does not imply equally large ranking collapse. NEZ-F1 falls to0.614448 and EZ-fraction bias becomes+0.192474. Thus the probability/decision operating point and predicted class fraction are central, not only EZ-versus-NEZ ordering. All thresholds were chosen internally before Outer; no retrospective threshold repair is permitted.

BothPR removal lowers Macro-F1/AUROC point estimates in all three selected seeds, but the AUROC patient interval still includes0. Sign consistency and statistical precision are different facts. Do not write the requested directions as a guarantee or a population-level result.

The PR factorial contrast P11-P01-P10+P00 is-0.112286 for Macro-F1, nominal95%[-0.154307,-0.072058]; AUROC+0.016337[-0.007457,0.038981]. This is non-additive response/shared normalization compensation, not positive biological synergy.

## Physiological input and six conditional deletions

Joint input exceeds Raw-only in both primary endpoints after the stated family adjustment, while the joint-minus-physiology-only differences are less precise. Pure branches retain original64hidden neurons, input scaling and weight slices, not widened compensation. The joint model nevertheless has more input weights than either pure branch, so this comparison alone cannot attribute all gain exclusively to physiology rather than added representational capacity. It supports incremental evidence for this fixed joint input package.

- Removing lowgamma gives Macro-F1/AUROC0.647015/0.769884 versus0.643771/0.766660. Both point estimates improve at each of the three selected seeds. However, main-minus-deletion nominal intervals[-0.007009,0.000363]forMacro-F1 and[-0.007905,0.000888]forAUROC still include0. This is a candidate for future legally selected simplification, not grounds for post-Outer deletion now.
- Removing Line Length gives Macro-F1=0.647034, but AUROC=0.765649 slightly below main; this is not uniform domination.
- Beta/partialhighgamma deletions have near-zero conditional differences; RMS/entropy deletions modestly lower average AUROC. None establishes indispensable conditional contribution across both family-adjusted endpoints.
- No single deletion establishes global subset optimality or physiological irrelevance. The test masks two direct summary coordinates before patient-relative statistics, keeps input widths/shared initialization and trains from scratch; there is no hidden FiLM/Gate route in these groups.

Main remains fixed; no additional combinations or thresholds were tried after seeing these results.

## Structural and capacity controls

Restoring Early FiLM yields Macro-F1/AUROC0.641733/0.765902, not a clear improvement in those endpoints. But AP/MRR/Top-1 become0.592297/0.842269/0.791667 versus main0.570795/0.787698/0.695833. Consequently no-FiLM is not superior on all localization endpoints. These point-estimate tradeoffs are retained; absence of a Macro-F1/AUROC win does not prove FiLM physiologically useless.

The restored model has10377parameters, main9329. The ordinary head73 capacity control has10337parameters,40fewer(-0.39%)than restoredEarly; it approximately matches parameter count, not function class or compute. Its Macro-F1/AUROC0.643904/0.770690 do not establish a reliable uniform gain over the fixed main under the prescribed intervals.

P1=0.642490/0.766303 andP2=0.644494/0.767045 do not establish reliable Macro-F1/AUROC superiority. P1 changes partitioning AND within-seizure statistic, so it is not a pure segmentation test. P2 adds26parameters and uses support-masked scorer inputs before matmul; extra capacity is explicit. Both alternatives have no physiological conditioning.

## Seed-selection and testing limits

Known main Macro-F1 seeds42/43/44/52/62 are0.644071/0.633324/0.641568/0.633450/0.645675. The selected three average0.643771; all five historically viewed main seeds average0.639618. These unselected low outcomes remain visible. Neither small SD within the requested subset nor repeated patient bootstrap removes outcome-informed seed/architecture selection bias.

All claims remain descriptive/conditional on this historical cohort and selected seed set. Three seeds do not restore independent confirmation. No main promotion is made from the17-arm table.

## Engineering provenance

Production training/validation/inference completed without restart or lost epoch/prediction. Original reporting stopped at a missing common.CONTEXT roots entry, after generating its metric prefix. An external reporting-only wrapper calls the original setup before original report/verifier/package. All14scientific sources,255sealed score files,32existing aggregate files and the existing private patient metric file remained byte-identical; no training, new prediction, metric formula, threshold or bootstrap change. See the separate execution manifest/proofs and RUN_AUDIT. Only compact anonymous aggregates are public.
