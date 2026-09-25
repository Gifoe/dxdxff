# V3-QBC Report

Formal predictions use one outer-train fixed-validation threshold on patient robust-z NEZ logits.
Test labels and true EZ counts are not used by the formal decoder.

- Held-out patients: 36
- Formal patient Macro-F1: 0.602527
- True-K diagnostic Macro-F1: 0.690401
- True-K status: DIAGNOSTIC_ONLY_NOT_DEPLOYABLE
