# Mini G1 cheap causal-feature audit

## Target and checkpoint

The primary action target is `CONTINUE_BENEFICIAL = 1 iff V_t > 0` (RESCUE); STABLE, CORRUPTION, and UNRECOVERED are negative. G1 uses the DEV-selected and now frozen checkpoint **0.60**. G3 holdout rows are absent from every fit, feature selection, threshold decision, and performance report.

## Causal features

| feature | definition | constant_at_primary |
| --- | --- | --- |
| feature_progress | Completed iterations / frozen total of 256 (constant at a fixed checkpoint). | True |
| feature_prefix_length | Number of observed iterations through t (constant at a fixed checkpoint). | True |
| feature_current_masked_ratio | Fraction not yet committed after iteration t. | True |
| feature_current_committed_ratio | Fraction committed after iteration t. | True |
| feature_current_candidate_token_length | Token count of the complete stop-now candidate. | True |
| feature_current_text_length | Decoded character count at t. | False |
| feature_current_text_token_length | Whitespace-delimited decoded text length at t. | False |
| feature_current_answer_present | Whether the canonical parser finds an answer at t. | False |
| feature_current_answer_length | Canonical answer character length. | False |
| feature_current_answer_token_length | Whitespace-delimited canonical answer length. | False |
| feature_current_answer_persistence | Consecutive prefix steps with the current parsed answer. | False |
| feature_current_answer_persistence_normalized | Persistence divided by observed prefix length. | False |
| feature_current_answer_first_seen_time | First prefix index containing the current answer. | False |
| feature_current_answer_first_seen_normalized | First-seen index normalized by the observed prefix. | False |
| feature_recent_answer_change_count | Parsed-answer changes in the last 16 causal transitions. | False |
| feature_recent_answer_change_rate | Fraction of last 16 transitions changing parsed answer. | False |
| feature_recent_answer_edit_distance | Mean normalized character edit distance over recent answers. | False |
| feature_current_answer_stability | One minus most recent parsed-answer edit distance. | False |
| feature_recent_answer_stability | Mean recent parsed-answer stability. | False |
| feature_recent_answer_stability_slope | Linear slope of recent parsed-answer stability. | False |
| feature_current_token_change_rate | Normalized token edit distance from t-1 to t. | False |
| feature_recent_token_change_rate | Mean normalized token edit distance in the recent window. | False |
| feature_current_token_stability | One minus current token change rate. | False |
| feature_recent_stability | Mean recent full-candidate token stability. | False |
| feature_recent_stability_slope | Linear slope of recent full-candidate token stability. | False |
| feature_current_commit_speed | Latest causal increase in committed ratio. | True |
| feature_recent_commit_speed | Mean causal increase in committed ratio over the recent window. | True |
| feature_past_oscillation_count | Returns to a previously left non-empty parsed answer through t. | False |
| feature_distinct_answers_seen | Number of distinct non-empty parsed answers observed through t. | False |

Because this public trajectory configuration commits exactly one token per iteration, nominal progress, prefix length, masked ratio, committed ratio, and commit speed are constant at a fixed checkpoint. They are retained as schema/audit features but cannot provide discrimination. No confidence proxy is present in the `.pt` schema, and no model weights were downloaded.

## Evaluation protocol

- RepeatedStratifiedKFold: 5 folds × 3 repeats.
- Every imputer/scaler/classifier is fit inside its training fold.
- Class imbalance: balanced class weights; no negative downsampling.
- Hard threshold: fixed at 0.5; no threshold tuning.
- Confidence intervals: 1,000 deterministic case-level bootstrap resamples of averaged repeated OOF predictions.
- Models: every single-feature balanced logistic baseline, all-feature balanced logistic regression, and balanced HistGradientBoosting.
- Gate aggregation: maximum OOF AUROC over this frozen model universe. This is conservative against advancing to G2, but its bootstrap CI is conditional on the selected model and does not correct for model-selection multiplicity.

## Primary results

The table shows both prespecified full models and the five strongest single-feature baselines. `reports/g1_primary_metrics.csv` contains every baseline.

| model | n_samples | n_positive | auroc | auprc | balanced_accuracy | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_all_features | 1055 | 428 | 0.733865 | 0.617171 | 0.678163 | 0.636459 | 0.209342 |
| hist_gradient_boosting_balanced | 1055 | 428 | 0.777847 | 0.657636 | 0.696189 | 0.650707 | 0.192503 |
| single_logistic::feature_current_answer_persistence | 1055 | 428 | 0.727955 | 0.610598 | 0.671477 | 0.657778 | 0.218412 |
| single_logistic::feature_current_answer_persistence_normalized | 1055 | 428 | 0.727955 | 0.610598 | 0.671477 | 0.657778 | 0.218412 |
| single_logistic::feature_recent_answer_change_count | 1055 | 428 | 0.710388 | 0.607200 | 0.643228 | 0.589659 | 0.216990 |
| single_logistic::feature_recent_answer_change_rate | 1055 | 428 | 0.710388 | 0.607200 | 0.643228 | 0.589659 | 0.216990 |
| single_logistic::feature_recent_answer_edit_distance | 1055 | 428 | 0.707335 | 0.599149 | 0.643675 | 0.579977 | 0.219886 |

Best cheap model: `hist_gradient_boosting_balanced`, AUROC=0.7778 (95% CI 0.7496–0.8048), AUPRC=0.6576; positive prevalence=0.4057.

Across the 15 individual validation folds for that model, AUROC ranged from 0.7331 to 0.8237 (mean 0.7710, sample SD 0.0307). This variability is reported explicitly rather than treating the aggregate OOF point estimate as uniformly stable.

## Matched controls

| task | model | n_samples | n_positive | n_negative | auroc | auprc | balanced_accuracy | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_same_current_wrong | logistic_all_features | 608 | 428 | 180 | 0.582671 | 0.757011 | 0.558126 | 0.668387 | 0.244847 |
| A_same_current_wrong | hist_gradient_boosting_balanced | 608 | 428 | 180 | 0.604569 | 0.774000 | 0.588422 | 0.775510 | 0.221061 |
| B_same_current_correct | logistic_all_features | 447 | 16 | 431 | 0.599188 | 0.052058 | 0.518199 | 0.071429 | 0.185427 |
| B_same_current_correct | hist_gradient_boosting_balanced | 447 | 16 | 431 | 0.534948 | 0.037803 | 0.487239 | 0.000000 | 0.054680 |
| C_same_final_correct | logistic_all_features | 859 | 428 | 431 | 0.803093 | 0.793615 | 0.732260 | 0.732558 | 0.181126 |
| C_same_final_correct | hist_gradient_boosting_balanced | 859 | 428 | 431 | 0.864584 | 0.838992 | 0.783540 | 0.787185 | 0.150241 |
| D_same_final_wrong | logistic_all_features | 196 | 16 | 180 | 0.698611 | 0.137733 | 0.604861 | 0.218750 | 0.184000 |
| D_same_final_wrong | hist_gradient_boosting_balanced | 196 | 16 | 180 | 0.542014 | 0.090684 | 0.461111 | 0.000000 | 0.117563 |

The corruption-conditioned tasks have limited positive counts and must not be interpreted as high-power null results. The protocol's ten-positive cutoff means only "minimum-count evaluable"; it is not a power analysis. Full bootstrap intervals are in `reports/g1_matched_bootstrap_ci.csv`.

## Gate decision

- Status: **G1_PASS**
- Reason: Cheap features did not stably solve both frozen key matched controls; the simple proxy set is incomplete.

This gate asks whether cheap temporal proxies already make a hidden-state audit unnecessary. It does not claim that a weak cheap baseline proves hidden states will work.

The machine-readable rule was frozen before this parser-v1.1 rerun in `reports/g1_cv_protocol.json` (SHA256 `5222fe457bf0d08e00e7b3bdb5ef44725deca347a868d8dadc8e6f847d268d08`). Like G0's exploratory rule, its operational details were finalized after the initial v1.0 DEV audit and are not confirmatory preregistration.
