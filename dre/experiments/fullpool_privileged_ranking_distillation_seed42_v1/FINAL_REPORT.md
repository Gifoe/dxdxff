# FullPool Privileged Patient Geometry Distillation Study

This run stopped at the prespecified FIT-only Teacher Gate 0. No Student was trained or evaluated.

## Sequential answers

1. A1 exact reproduction: prior 150-checkpoint/R4 replay PASS; fixed-query A1 AP 0.576743463. The audit was reused, not rerun from raw data in this branch.
2. Retuned FULLPOOL reference: PASS, matched fixed-query AP 0.692464487 (nondeployable, prior audited replay).
3. Cross-fitted FIT Teacher: AP 0.679229 vs FIT A1 0.601246, delta +0.077982; 5/5 outer folds positive; FIT patient-ID bootstrap 95% CI [+0.048096, +0.108110]. MRR delta -0.007661 (CI [-0.055464, +0.040055]) and Top1 delta -0.039100 (CI [-0.108007, +0.028624]) violate the prelocked point-estimate <0.005 decline requirement. Both ranking CIs cross zero, so this is a protocol gate failure, not proof of a population-level decline.
4. D0+ continued hard-label training: NOT RUN because Gate 0 failed.
5. D1 Pairwise KD: NOT RUN.
6. D2 Listwise KD: NOT RUN.
7. D3 correction-focused KD: NOT RUN; cannot rank KD variants.
8. D4 Pair+List: NOT RUN; its FIT gate was never reached.
9–10. No Student AP/MRR/Top1/F1/BA or Student-vs-D0+ bootstrap CI exists. Do not substitute Teacher metrics.
11. Teacher-added headroom denominator is 0.115721 AP; transfer efficiency eta is undefined without a Student.
12–14. Student 0.590/current-B8/best-B8 milestones: NOT EVALUATED.
15. Teacher correction-pair mechanism: NOT EVALUATED in a Student, because no Student passed Gate 0.
16. The stop is due to Teacher ranking-quality point-estimate gate failure, not evidence that privileged signal is absent: FIT OOF AP/AUROC/F1 improved, while early-rank MRR/Top1 point estimates declined with CIs spanning zero. The claim that signal is not zero-shot distillable is untested.
17. Current evidence does not justify claiming failure of all R4-only distillation; this *specified* Student study stops by its predeclared gate. A future protocol would need to address Teacher early-rank quality prospectively, not retune this run after seeing outcomes.

Exact terminal: `PRIVILEGED_TEACHER_SIGNAL_GATE_FAILED`.

Limitations: FIT OOF scores are not on the 65 fixed-query target cells and cannot be directly compared to 0.692464 FULLPOOL reference. The inherited A1 VLOO/loader prevents literal strict target-label sequencing; no new target labels were used for Teacher fitting or Student selection.
