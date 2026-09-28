"""Protocol-complete negative handoff when Teacher Gate 0 fails."""
from __future__ import annotations

import json

import protocol as p


def skipped_csv(name,reason="NOT_RUN_TEACHER_GATE0_FAILED"):
    p.afc.write_csv(p.ROOT/name,[dict(status=reason,scope="not_evaluated",value="")])


def main():
    p.preflight()
    audit=json.loads((p.ROOT/"TEACHER_CROSSFIT_AUDIT.json").read_text(encoding="utf-8"))
    ref=json.loads((p.ROOT/"FULLPOOL_REFERENCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if audit["lock_sha"]!=p.LOCK_SHA or audit["PRIVILEGED_TEACHER_SIGNAL_VALID"]:
        raise RuntimeError("Gate 0 did not fail; Student implementation is required")
    if audit["n_source_contexts"]!=17 or audit["n_channel_crossfit_heads"]!=5*audit["n_fit_patient_contexts"] or not ref["pass_reference"]:
        raise RuntimeError("Teacher/source reference incomplete")
    if not (audit["mean_delta_ap"]>.03 and audit["positive_outer_folds"]>=4 and
            (audit["mean_delta_mrr"]<=-.005 or audit["mean_delta_top1"]<=-.005)):
        raise RuntimeError("Unexpected Gate 0 failure mode; inspect before finalizing")
    teacher_rows=p.afc.read_csv(p.ROOT/"FIT_OOF_TEACHER_METRICS.csv")
    source_boot=p.afc.read_csv(p.ROOT/"FIT_OOF_TEACHER_BOOTSTRAP.csv")
    fold_rows=[r for r in teacher_rows if r.get("n_contexts") and r["fold"]!="ALL"]
    if len(fold_rows)!=5 or len(source_boot)!=3:raise RuntimeError("FIT teacher metrics incomplete")
    p.afc.write_csv(p.ROOT/"FOLD_CONSISTENCY.csv",
                     [dict(fold=r["fold"],scope="FIT_OOF_TEACHER_ONLY",a1_ap=r["a1_ap"],
                           teacher_ap=r["teacher_ap"],delta_ap=r["delta_ap"],
                           delta_mrr=r["delta_mrr"],delta_top1=r["delta_top1"],
                           student_status="NOT_RUN_GATE0") for r in fold_rows])
    p.afc.write_csv(p.ROOT/"PATIENT_CLUSTER_BOOTSTRAP.csv",
                     [dict(scope="FIT_OOF_TEACHER_VS_A1",metric=r["metric"],delta_mean=r["delta_mean"],
                           ci_low=r["ci_low"],ci_high=r["ci_high"],
                           unique_patient_ids=r["unique_fit_patient_ids"],resamples=r["resamples"],
                           target_student_comparison="NOT_RUN_GATE0") for r in source_boot])
    for name in ("FIT_KD_HYPERPARAM_SELECTION.csv","CONTINUED_TRAINING_CONTROL.csv",
                 "TEACHER_STUDENT_PREFERENCE_AGREEMENT.csv","CORRECTION_STRATIFIED_AUDIT.csv",
                 "TARGET_FAILURE_STRATIFIED_ANALYSIS.csv"):
        skipped_csv(name)
    p.afc.write_csv(p.ROOT/"STUDENT_VARIANT_MATRIX.csv",
                     [dict(variant=name,status=("HISTORICAL_REFERENCE_ONLY" if name=="D0_ORIGINAL_A1" else "NOT_RUN_GATE0"),
                           target_ap=(p.A1_AP if name=="D0_ORIGINAL_A1" else ""),
                           target_mrr="",target_top1="",compared_to_D0_plus="NOT_EVALUATED")
                      for name in ("D0_ORIGINAL_A1","D0_CONTINUED_HARDLABEL","D1_PAIRWISE_KD",
                                   "D2_LISTWISE_KD","D3_CORRECTION_FOCUSED_KD","D4_PAIR_LIST_KD")])
    denominator=p.FULLPOOL_AP-p.A1_AP
    p.afc.write_csv(p.ROOT/"HEADROOM_TRANSFER.csv",
                     [dict(variant="NO_STUDENT_TRAINED",status="NOT_RUN_GATE0",a1_target_ap=p.A1_AP,
                           nondeployable_fullpool_target_ap=p.FULLPOOL_AP,
                           teacher_added_headroom_denominator=denominator,
                           student_target_ap="",teacher_headroom_transfer_percent="")])
    gates=dict(lock_sha=p.LOCK_SHA,terminal="PRIVILEGED_TEACHER_SIGNAL_GATE_FAILED",
               stop_stage="FIT_OOF_TEACHER",teacher_fit_oof_delta_ap=audit["mean_delta_ap"],
               teacher_delta_mrr=audit["mean_delta_mrr"],teacher_delta_top1=audit["mean_delta_top1"],
               teacher_positive_outer_folds=audit["positive_outer_folds"],
               PRIVILEGED_TEACHER_SIGNAL_VALID=False,
               PRIVILEGED_DISTILLATION_SUPPORTED="NOT_EVALUATED_GATE0",
               PRIVILEGED_STUDENT_AP_059_REACHED="NOT_EVALUATED_GATE0",
               PRIVILEGED_STUDENT_REACHES_CURRENT_B8="NOT_EVALUATED_GATE0",
               PRIVILEGED_STUDENT_REACHES_BEST_B8="NOT_EVALUATED_GATE0",
               student_training_started=False,target_student_scores_created=False,
               target_student_outcomes_accessed=False)
    p.afc.write_json(p.ROOT/"DISTILLATION_GATES.json",gates)
    p.afc.write_json(p.ROOT/"LABEL_USAGE_AUDIT.json",
                     dict(lock_sha=p.LOCK_SHA,student_inference_budget=0,
                          teacher_fit_uses_only_other_four_channel_folds=True,
                          teacher_lambda_for_each_fit_patient_selected_from_other_fit_patients=True,
                          fit_labels_used_for_teacher_training_and_audit=True,
                          validation_target_labels_used_for_new_teacher_or_student=False,
                          no_new_target_student_predictions=True,
                          legacy_loader_materializes_all_80_labels=True,
                          strict_target_label_sequencing=False,
                          reason="Inherited A1 VLOO source selector has cross-target label dependencies; legacy loader materializes labels."))
    (p.ROOT/"IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n"
        f"- Frozen lock SHA-256 `{p.LOCK_SHA}` was pushed before new Teacher outcomes.\n"
        "- Prior exact A1 150-checkpoint/R4 replay and B0 identity were reused and hash-checked. "
        "Prior matched fixed-query retuned 64D FULLPOOL AP was replayed from the audited reference; no new target FULLPOOL fit was run.\n"
        f"- {audit['n_fit_patient_contexts']} FIT patient-contexts across 17 exact A1 contexts; "
        f"{audit['n_channel_crossfit_heads']} convex heads. Five deterministic label-blind channel folds per patient. "
        "Each channel's score came from a head fitted on the other four folds only.\n"
        "- The 21-point regularization grid used three deterministic FIT half-split episodes per patient. "
        "For each OOF-scored patient, lambda was selected only from other FIT patients in that context. "
        "This is stricter than the prior global FIT-lambda selection and prevents the scored patient's label from choosing its own lambda.\n"
        "- FIT OOF metrics weight contexts by inherited target-cell multiplicity and then average five outer folds equally. "
        "The supplementary 10,000-draw bootstrap clusters repeated FIT appearances by exact patient ID.\n"
        "- The prespecified Gate 0 fails on MRR and Top1 despite a large AP gain. No Student, continued-training control, "
        "target scores, KD tuning, or target outcome evaluation were executed. CSV files for these stages explicitly say NOT_RUN_GATE0.\n"
        "- Legacy A1 VLOO uses cross-target labels and its loader materializes all 80 labels; literal strict target-label sequencing is false.\n",
        encoding="utf-8")
    ap_boot=next(r for r in source_boot if r["metric"]=="ap")
    mrr_boot=next(r for r in source_boot if r["metric"]=="mrr")
    top_boot=next(r for r in source_boot if r["metric"]=="top1")
    text=["# FullPool Privileged Patient Geometry Distillation Study", "",
          "This run stopped at the prespecified FIT-only Teacher Gate 0. No Student was trained or evaluated.","",
          "## Sequential answers", "",
          f"1. A1 exact reproduction: prior 150-checkpoint/R4 replay PASS; fixed-query A1 AP {p.A1_AP:.9f}. The audit was reused, not rerun from raw data in this branch.",
          f"2. Retuned FULLPOOL reference: PASS, matched fixed-query AP {ref['mean_ap']:.9f} (nondeployable, prior audited replay).",
          f"3. Cross-fitted FIT Teacher: AP {audit['fit_oof_teacher_ap']:.6f} vs FIT A1 {audit['fit_oof_a1_ap']:.6f}, delta {audit['mean_delta_ap']:+.6f}; 5/5 outer folds positive; FIT patient-ID bootstrap 95% CI [{float(ap_boot['ci_low']):+.6f}, {float(ap_boot['ci_high']):+.6f}]. MRR delta {audit['mean_delta_mrr']:+.6f} (CI [{float(mrr_boot['ci_low']):+.6f}, {float(mrr_boot['ci_high']):+.6f}]) and Top1 delta {audit['mean_delta_top1']:+.6f} (CI [{float(top_boot['ci_low']):+.6f}, {float(top_boot['ci_high']):+.6f}]) violate the prelocked point-estimate <0.005 decline requirement. Both ranking CIs cross zero, so this is a protocol gate failure, not proof of a population-level decline.",
          "4. D0+ continued hard-label training: NOT RUN because Gate 0 failed.",
          "5. D1 Pairwise KD: NOT RUN.","6. D2 Listwise KD: NOT RUN.",
          "7. D3 correction-focused KD: NOT RUN; cannot rank KD variants.",
          "8. D4 Pair+List: NOT RUN; its FIT gate was never reached.",
          "9–10. No Student AP/MRR/Top1/F1/BA or Student-vs-D0+ bootstrap CI exists. Do not substitute Teacher metrics.",
          f"11. Teacher-added headroom denominator is {denominator:.6f} AP; transfer efficiency eta is undefined without a Student.",
          "12–14. Student 0.590/current-B8/best-B8 milestones: NOT EVALUATED.",
          "15. Teacher correction-pair mechanism: NOT EVALUATED in a Student, because no Student passed Gate 0.",
          "16. The stop is due to Teacher ranking-quality point-estimate gate failure, not evidence that privileged signal is absent: FIT OOF AP/AUROC/F1 improved, while early-rank MRR/Top1 point estimates declined with CIs spanning zero. The claim that signal is not zero-shot distillable is untested.",
          "17. Current evidence does not justify claiming failure of all R4-only distillation; this *specified* Student study stops by its predeclared gate. A future protocol would need to address Teacher early-rank quality prospectively, not retune this run after seeing outcomes.",
          "",f"Exact terminal: `{gates['terminal']}`.",
          "", "Limitations: FIT OOF scores are not on the 65 fixed-query target cells and cannot be directly compared to 0.692464 FULLPOOL reference. The inherited A1 VLOO/loader prevents literal strict target-label sequencing; no new target labels were used for Teacher fitting or Student selection.",""]
    (p.ROOT/"FINAL_REPORT.md").write_text("\n".join(text),encoding="utf-8")
    print(f"[GATE0_FINALIZED] {gates['terminal']} Student=NOT_RUN",flush=True)


if __name__=="__main__":main()
