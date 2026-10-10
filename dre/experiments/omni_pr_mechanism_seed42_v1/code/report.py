"""Aggregate-only report generated from registered frozen comparisons."""
from source import *
def final_report(frame,diagnostic,boot,gate):
    means=frame.groupby('method')[METRICS+[m+'_0_5' for m in CLASS]].mean();center=frame.groupby(['center','method'])[METRICS].mean()
    def v(a,m):return f'{means.loc[a,m]:.6f}'
    def d(a,b,m='auroc'):return float(means.loc[a,m]-means.loc[b,m])
    prabs=boot.loc[('C1-PR-C1-ABS','auroc')];prsh=boot.loc[('C1-PR-C1-SHUFFLE','auroc')]
    lines=['# Omni-PR mechanism identification, seed42','',f'Terminal: `{gate["terminal"]}`. PR gate: {gate["PR_gate_pass"]}; alignment gate: {gate["alignment_gate_pass"]}.','',
           '| Arm | Patient AUROC | AP | MRR | Top1 | NDCG | Macro-F1 FIT tau | Pathological F1 FIT tau | Macro-F1 at0.5 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['C0','C1-PR','C1-ABS','C1-SHUFFLE']:lines.append('| '+arm+' | '+' | '.join(v(arm,m) for m in ['auroc','ap','mrr','top1','ndcg','macro_f1','pathological_f1','macro_f1_0_5'])+' |')
    lines+=['','All summaries are patient-equal across the same124 patients/10202 labeled canonical channels. C0/PR are exact historical replay, not retrained candidates or new confirmations. Each arm uses its frozen fold-specific FIT-selected threshold;0.5 results are separate. Published pooled Task2 AUROC0.8061 is not used.','',
            '## Registered paired patient contrasts','', '| Contrast | AUROC delta |95% percentile CI | Positive folds | Positive patients |','|---|---:|---|---:|---:|']
    for contrast in ['C1-PR-C1-ABS','C1-PR-C1-SHUFFLE','C1-ABS-C0','C1-SHUFFLE-C0','C1-PR-C0']:
        r=boot.loc[(contrast,'auroc')];lines.append(f'| {contrast} | {r.delta:+.6f} | [{r.CI_low:+.6f}, {r.CI_high:+.6f}] | {int(r.positive_folds)}/5 | {int(r.positive_patients)}/124 |')
    lines+=['','10,000 unique-patient paired draws,seed42,all numeric thresholds and models fixed. Patient effects median/IQR and per-center contrasts are provided. Intervals are exploratory and do not correct historical cohort reuse or broader research-selection bias. No additional hypothesis test or outcome-based model search was run.','',
            '## Source decomposition','', '| Source | n | PR AUROC | ABS AUROC | SHUFFLE AUROC | PR-ABS | PR-SHUFFLE |','|---|---:|---:|---:|---:|---:|---:|']
    for s,n in [('HUP',10),('Open-iEEG',103),('SourceSink',11)]:
        a=center.loc[(s,'C1-PR'),'auroc'];b=center.loc[(s,'C1-ABS'),'auroc'];c=center.loc[(s,'C1-SHUFFLE'),'auroc'];lines.append(f'| {s} | {n} | {a:.6f} | {b:.6f} | {c:.6f} | {a-b:+.6f} | {a-c:+.6f} |')
    diag=diagnostic.groupby('diagnostic')[METRICS].mean();audit=json.loads((ROOT/'public/SCALE_ONLY_INVARIANCE_AUDIT.json').read_text())
    lines+=['','## Frozen inference diagnostics','',
      'For a fixed beta, PR=(beta/sigma_patient) dot h + patient-constant offset. Removing mean subtraction at inference cannot alter exact within-patient ranking. Direct float64 formulas and pairwise ordering were verified using the historical float32 patient moments. Sigmoid-saturation ties are resolved using logits for this mathematical diagnostic, not by changing primary historical scores.',
      f'Scale-only maximum mathematical rank-metric error={audit["formal_max_math_rank_error"]}; maximum difference against original operational PR logit-ranking metrics={audit["formal_max_operational_PR_logit_ranking_error"]}; difference against archived probability-ranking metrics={audit["formal_max_historical_probability_metric_error"]}.',
      f'Scale-only Macro-F1={diag.loc["SCALE_ONLY","macro_f1"]:.6f}, delta historical PR={diag.loc["SCALE_ONLY","macro_f1"]-means.loc["C1-PR","macro_f1"]:+.6f}, at the original PR threshold with no refit.',
      f'Frozen aligned PR head with inference-shuffled inputs: AUROC={diag.loc["INFERENCE_SHUFFLE","auroc"]:.6f}, AP={diag.loc["INFERENCE_SHUFFLE","ap"]:.6f}, Macro-F1={diag.loc["INFERENCE_SHUFFLE","macro_f1"]:.6f}. This differs from C1-SHUFFLE,which was trained from scratch on shuffled embeddings.',
      '', '## Answers to26 requested questions','',
      '1. Yes. Historical C0/PR logits,selected checkpoint tensors,thresholds and patient metrics replay exactly; HISTORICAL_REPRODUCTION.json gives measured errors.',
      '2. Yes. All124 patients,137 records and10202 labeled patient-channels retained with original five folds and labels.',
      f'3. ABS-C0 AUROC={d("C1-ABS","C0"):+.6f}; AP={d("C1-ABS","C0","ap"):+.6f}. See registered intervals before claiming an improvement.',
      f'4. PR-ABS AUROC={prabs.delta:+.6f},CI[{prabs.CI_low:+.6f},{prabs.CI_high:+.6f}]; full PR increment gate={gate["PR_gate_pass"]}.',
      f'5. PR-SHUFFLE AUROC={prsh.delta:+.6f},CI[{prsh.CI_low:+.6f},{prsh.CI_high:+.6f}]; alignment gate={gate["alignment_gate_pass"]}.',
      f'6. PR-ABS positivefolds={int(prabs.positive_folds)}/5; PR-SHUFFLE={int(prsh.positive_folds)}/5. Full fold metrics retained.',
      f'7. PR-ABS AP={d("C1-PR","C1-ABS","ap"):+.6f}; PR-SHUFFLE AP={d("C1-PR","C1-SHUFFLE","ap"):+.6f}. AUROC and AP are not interchangeable.',
      f'8. FIT-threshold Macro-F1 PR/ABS/SHUFFLE={v("C1-PR","macro_f1")}/{v("C1-ABS","macro_f1")}/{v("C1-SHUFFLE","macro_f1")}; no test reoptimization.',
      f'9. FIT-threshold pathological F1 PR/ABS/SHUFFLE={v("C1-PR","pathological_f1")}/{v("C1-ABS","pathological_f1")}/{v("C1-SHUFFLE","pathological_f1")}.',
      f'10. PR-ABS MRR/Top1 deltas={d("C1-PR","C1-ABS","mrr"):+.6f}/{d("C1-PR","C1-ABS","top1"):+.6f}; PR-SHUFFLE={d("C1-PR","C1-SHUFFLE","mrr"):+.6f}/{d("C1-PR","C1-SHUFFLE","top1"):+.6f}.',
      '11. '+('The registered predictive incremental PR gate passes under this frozen-backbone protocol.' if gate['PR_gate_pass'] else 'The complete predictive incremental PR gate does not pass; do not claim patient normalization is necessary.'),
      '12. '+('The registered trained-alignment control supports a contribution from correct embedding correspondence.' if gate['alignment_gate_pass'] else 'Correct electrode alignment is not established as necessary by the registered alignment gate.'),
      '13. Source bias is constant within patient and cannot affect exact patient ranking; classification can change. All three heads have identical source-bias topology and loss regularization.',
      f'14. Scale-only mathematical AUROC/AP invariance passed; error={audit["formal_max_math_rank_error"]}. Operational/saturation differences are measured above,not silently replaced.',
      '15. PR versus ABS compares patient-specific normalization to patient-equal globally standardized embeddings. It does not individually identify variance scaling,training centering,optimization conditioning or their interaction with learned beta and frozen logits.',
      '16. See every source PR-minus-ABS/SHUFFLE effect above and complete center metrics; no center or patient was dropped and no source-specific threshold fitted.',
      '17. Yes. HUP10 and SourceSink11 are small samples; center effects are descriptive and cannot establish broad source generalization.',
      f'18. ABS-C0 AUROC gain={d("C1-ABS","C0"):+.6f},compared to historical PR gain={d("C1-PR","C0"):+.6f}; descriptive comparability rule={gate["descriptive_comparability_rule_pass"]}.',
      f'19. Train/evaluate SHUFFLE-C0 AUROC={d("C1-SHUFFLE","C0"):+.6f},AP={d("C1-SHUFFLE","C0","ap"):+.6f}. A surviving point estimate alone is not proof of nonspecific optimization; use its uncertainty and the matched alignment contrast.',
      '20. Yes. Inner ABS scalers use inner-training patients only. Final scalers use outer-FIT only. All good channels enter moments equally within each equally weighted patient; between-patient variance is retained.',
      '21. Yes.35 trainable parameters each; global moments are fitted statistics,not learned parameters. Original PR was never retrained.',
      '22. '+('Both the minimum PR effect and positive paired AUROC lower bound are supported,subject to all gate conditions above.' if gate['PR_gate_pass'] else 'The paired bootstrap does not establish the full registered PR hypothesis under all prespecified requirements.'),
      '23. Yes for outer patients. The original fold CNN and new head both exclude them. Inner head-validation patients were previously used by the frozen CNN; inner head OOF is not end-to-end OOF.',
      '24. Training centering,patient-specific variance scaling,conditioning and learned-beta/base-logit interactions remain causally unidentified. Inference scale-only invariance only rules out mean subtraction alone as a direct ranking effect for fixed beta.',
      '25. '+('Qualified exploratory predictive PR increment is supported; a causal or independent external mechanism claim is not.' if gate['PR_gate_pass'] and gate['alignment_gate_pass'] else 'No unqualified Patient-Relative mechanism claim is justified. Follow the terminal and uncertainty,not the largest point estimate.'),
      '26. '+('Historical35-parameter PR head is the simplest registered candidate with supported incremental normalization and alignment.' if gate['PR_gate_pass'] and gate['alignment_gate_pass'] else 'Use the simplest matched aligned linear head justified by the registered comparisons; if uncertainty remains,retain both descriptions without selecting a new ensemble or rerunning.'),
      '', '## Registered gates','',json.dumps(gate,indent=2),
      '', '## Scope and provenance','',
      'Historical C0/PR are reused observations,not a new confirmation. No new CNN/PR training,raw signal extraction,feature regeneration,architecture/seed/threshold search or post-outcome rescue. Only30 inner control heads and10 selected-length refits were trained. All models and transformations froze before the two new outer arms were evaluated.',
      'Primary scores preserve historical sigmoid conventions. Scale-only logit-ranking diagnostics are explicitly separate. All patient/clinical/electrode identities,labels,embeddings,predictions,checkpoints,scalers,permutations and optimizer state remain private. Public files contain aggregate outputs and hashes only.',
      f'Protocol SHA256 `{sha(PROTOCOL)}`. New pre-outer freeze `{sha(ROOT/"FROZEN_BEFORE_OUTER_PRIVATE.json")}`. Historical published tree is pinned in PROTOCOL_LOCK.json.']
    (ROOT/'public/FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
