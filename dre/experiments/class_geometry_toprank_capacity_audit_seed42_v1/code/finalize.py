"""Validate completed FIT/validation audit, write aggregate report, never inspect outer."""

from __future__ import annotations

import json
from pathlib import Path

from common import EXPERIMENT, RUNTIME, ensure_source, read_csv, write_csv, write_json


def only(path):
    rows=read_csv(EXPERIMENT/path)
    if len(rows)!=1:
        raise RuntimeError(f"Expected one comparison row: {path}")
    return rows[0]


def number(row,key):
    return float(row[key])


def main():
    ensure_source()
    source=json.loads((EXPERIMENT/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    unit=json.loads((EXPERIMENT/"CLASS_BALANCED_LOSS_UNIT_TEST.json").read_text(encoding="utf-8"))
    if not source["pass"] or source["checkpoints"]!=150 or source["max_validation_grid_error"]>1e-8 or not unit["pass"]:
        raise RuntimeError("A1 reproduction or pretraining unit check failed")
    gates={name:json.loads((EXPERIMENT/path).read_text(encoding="utf-8")) for name,path in (
        ("class","audit_a_class_geometry/CLASS_GEOMETRY_GATE.json"),
        ("top","audit_b_top_heavy/HARD_NEGATIVE_GATE.json"),
        ("capacity","audit_c_capacity/CAPACITY_GATE.json"))}
    trained=("B10","B01","B11","CAP1","CAP2")+(("FP",) if gates["top"]["first_positive_triggered"] else ())
    for variant in trained:
        for fold in range(1,6):
            completed=json.loads((RUNTIME/"private"/"training"/variant/f"fold_{fold}"/"complete.json").read_text(encoding="utf-8"))
            if completed["epochs"]!=30 or completed["outer_test_accessed"]:
                raise RuntimeError(f"Incomplete training: {variant} fold {fold}")
    a=only("audit_a_class_geometry/CLASS_GEOMETRY_COMPARISON.csv")
    b={row["candidate"]:row for row in read_csv(EXPERIMENT/"audit_b_top_heavy"/"TOP_HEAVY_COMPARISON.csv")}
    c={row["candidate"]:row for row in read_csv(EXPERIMENT/"audit_c_capacity"/"CAPACITY_COMPARISON.csv")}
    matches=(("class_balanced_bce",a,"B00 exact A1",gates["class"]["pass"],gates["class"]["terminal"]),
             ("hard_negative_on_original",b["B01"],"B00 exact A1",gates["top"]["effects"]["B01"]["pass"],gates["top"]["terminal"]),
             ("hard_negative_on_class_balanced",b["B11"],"B10 class-balanced",gates["top"]["effects"]["B11"]["pass"],gates["top"]["terminal"]),
             ("feature_capacity_64",c["CAP1"],"CAP0 frozen BCE base",gates["capacity"]["effects"]["CAP1"]["pass"],gates["capacity"]["terminal"]),
             ("feature_capacity_128",c["CAP2"],"CAP0 frozen BCE base",gates["capacity"]["effects"]["CAP2"]["pass"],gates["capacity"]["terminal"]))
    summary=[]
    for name,row,control,passed,interpretation in matches:
        out={"source":name}
        for friendly,metric in (("macro_f1","patient_macro_f1"),("auprc","patient_ez_auprc"),
                                ("mrr","patient_ez_mrr"),("top1","top1_is_ez")):
            for prefix in ("baseline","candidate","delta"):
                out[f"{prefix}_{friendly}"]=row[f"{prefix}_{metric}"]
        out.update({"positive_auprc_folds":row["positive_auprc_folds"],
                    "positive_mrr_folds":row["positive_mrr_folds"],
                    "matched_control":control,"mechanism_gate_pass":passed,"interpretation":interpretation})
        summary.append(out)
    write_csv(EXPERIMENT/"RANKING_GEOMETRY_SUMMARY.csv",summary)
    bootstrap=read_csv(EXPERIMENT/"diagnostics"/"PAIRED_BOOTSTRAP_SUMMARY.csv")
    if len(bootstrap)!=15 or any(int(row["n_paired_cases"])!=65 for row in bootstrap):
        raise RuntimeError("Paired bootstrap incomplete")
    if len(read_csv(EXPERIMENT/"audit_c_capacity"/"TRAIN_VALIDATION_RANKING_GAP.csv"))!=450:
        raise RuntimeError("Capacity epoch trajectories incomplete")
    if len(read_csv(EXPERIMENT/"diagnostics"/"EZ_PREVALENCE_EFFECT_SUMMARY.csv"))<40:
        raise RuntimeError("Prevalence analysis incomplete")
    fp=None
    if gates["top"]["first_positive_triggered"]:
        fp=json.loads((EXPERIMENT/"first_positive_diagnostic"/"FIRST_POSITIVE_DIAGNOSTIC.json").read_text(encoding="utf-8"))
        if not fp["diagnostic_only"]:
            raise RuntimeError("First-positive must remain diagnostic-only")
    terminal=("CLASS_GEOMETRY_OR_TOPRANK_SOURCE_IDENTIFIED" if any(gate["pass"] for gate in gates.values())
              else "RANKING_SOURCE_STILL_UNRESOLVED")
    mass={row["metric"]:row for row in read_csv(EXPERIMENT/"audit_a_class_geometry"/"LOSS_MASS_DIAGNOSTIC.csv")}
    margins={row["matched_effect"]:row for row in read_csv(EXPERIMENT/"audit_b_top_heavy"/"HARD_NEGATIVE_MARGIN_DIAGNOSTICS.csv")}
    def fmt(value): return f"{float(value):+.6f}"
    rows=["# Class geometry, top-rank and capacity audit (seed 42)","",
          "Development-only FIT/validation experiment; no outer-test loader, prediction or performance was used. Historical outer exposure in earlier project work means these are exploratory development findings, not sealed confirmation.",
          f"A1 reproduction: {source['checkpoints']}/150 checkpoints, maximum grid error {source['max_validation_grid_error']}, VLOO Macro-F1 {source['mean_macro_f1']:.10f}. Six balanced-loss unit tests and all-patient class-support checks passed.",
          "","| Matched effect | Delta EZ-AUPRC | Delta EZ-MRR | Delta Top1 | Delta Macro-F1 | AUPRC-positive folds | Gate |",
          "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        rows.append(f"| {row['source']} | {fmt(row['delta_auprc'])} | {fmt(row['delta_mrr'])} | {fmt(row['delta_top1'])} | {fmt(row['delta_macro_f1'])} | {row['positive_auprc_folds']}/5 | {'PASS' if row['mechanism_gate_pass'] else 'FAIL'} |")
    class_auprc=number(a,"delta_patient_ez_auprc")
    im=mass["auprc"]
    gap1=number(c["CAP1"],"gap_increase")
    gap2=number(c["CAP2"],"gap_increase")
    b01=b["B01"]; b11=b["B11"]
    cap1=c["CAP1"]; cap2=c["CAP2"]
    source_f1=number(a,"baseline_patient_macro_f1")
    best_f1=max((number(row,"candidate_macro_f1"),row["source"],number(row,"delta_auprc")) for row in summary)
    jointly_qualified=[row["source"] for row in summary if number(row,"candidate_macro_f1")>=0.640 and
                       number(row,"delta_auprc")>=0 and number(row,"delta_mrr")>=0]
    rows.extend(["","## Direct answers","",
                 f"1. True patient-level class balancing changes EZ-AUPRC by {fmt(class_auprc)}; {gates['class']['terminal']}.",
                 f"2. Original-loss-mass imbalance versus AUPRC gain has Pearson {float(im['pearson']):+.4f}, Spearman {float(im['spearman']):+.4f}. Quartile aggregates are in `audit_a_class_geometry/EZ_FRACTION_QUARTILES.csv`; correlation is descriptive, not selection.",
                 f"3. AUPRC delta is {fmt(class_auprc)} and Macro-F1 delta is {fmt(number(a,'delta_patient_macro_f1'))}; class balance is not a ranking improvement under the gate." if not gates['class']['pass'] else
                 f"3. AUPRC and Macro-F1 deltas are {fmt(class_auprc)} and {fmt(number(a,'delta_patient_macro_f1'))}, respectively.",
                 f"4. Hard-negative MRR deltas are {fmt(number(b01,'delta_patient_ez_mrr'))} on original BCE and {fmt(number(b11,'delta_patient_ez_mrr'))} on balanced BCE; Top1 deltas are {fmt(number(b01,'delta_top1_is_ez'))} and {fmt(number(b11,'delta_top1_is_ez'))}.",
                 f"5. Hard-negative AUPRC deltas are {fmt(number(b01,'delta_patient_ez_auprc'))} and {fmt(number(b11,'delta_patient_ez_auprc'))}; preservation checks are explicit in `HARD_NEGATIVE_GATE.json`.",
                 f"6. Matched B11−B10 versus B01−B00 results do not justify claiming complementary benefit unless the prelocked matched gate passes; overall terminal: {gates['top']['terminal']}.",
                 f"7. Hard-negative margin deltas are {fmt(margins['B01_minus_B00']['delta_hard_negative_margin'])} and {fmt(margins['B11_minus_B10']['delta_hard_negative_margin'])}. This is a label-using diagnostic only.",
                 f"8. Capacity 64/128 validation AUPRC deltas are {fmt(number(cap1,'delta_patient_ez_auprc'))} and {fmt(number(cap2,'delta_patient_ez_auprc'))}; {gates['capacity']['terminal']}.",
                 f"9. FIT AUPRC and validation AUPRC are reported separately in `CAPACITY_COMPARISON.csv`; train–validation gap increases are {fmt(gap1)} and {fmt(gap2)}. A FIT-only gain is more consistent with generalization difficulty than a simple under-capacity explanation, but does not prove a feature ceiling.",
                 "10. No mechanism has matched causal support sufficient to name a ranking source." if not any(g["pass"] for g in gates.values()) else
                 "10. At least one mechanism passes its prelocked matched development gate; see table and gates for the specific source.",
                 f"11. Source A1 Macro-F1 is {source_f1:.6f}. Best candidate among the five matched rows is {best_f1[1]} at {best_f1[0]:.6f}, with AUPRC delta {fmt(best_f1[2])}. Candidates jointly reaching ≥0.640 with nondecreasing AUPRC and MRR: {', '.join(jointly_qualified) if jointly_qualified else 'none'}.",
                 "12. No single new model redesign is justified from this audit alone." if terminal=="RANKING_SOURCE_STILL_UNRESOLVED" else
                 "12. One mechanism passed a prelocked development gate; independent validation remains necessary before redesign claims.",
                 "",f"Optional first-positive diagnostic: {'run; MRR delta '+fmt(fp['delta_mrr']) if fp else 'not triggered'}. It is not eligible as a final candidate.",
                 f"Exact terminal: `{terminal}`.","OUTER_TEST_ACCESSED = NO",""])
    (EXPERIMENT/"FINAL_REPORT.md").write_text("\n".join(rows),encoding="utf-8")
    implementation=["# Implementation audit","",
                    "- New branch starts from the exact A1 source tip. Source replay independently checked all 150 validation checkpoints and the five locked Macro-F1 values.",
                    "- All new variants used the same frozen five FIT/validation folds, 30 epochs, source optimizer and per-epoch shuffle RNG. No early stopping, no S-RANK selector and no outer-test loader.",
                    "- B00 and A0 reuse exact A1 checkpoint grids. B10 is patient class-balanced BCE; B01/B11 differ only in the fixed hard-negative loss. CAP0 uses the prelocked Audit-A gate to choose the BCE base.",
                    "- Common CAP parameters were restored from the same A1 fold initialization. Only the feature MLP was replaced; downstream dimension remains 32.",
                    "- S-F1 selection excluded each validation patient from checkpoint and threshold choice. Patient-level records, selected checkpoints, optimizer state and logs remain private on the server.",
                    "- Label-using imbalance, prevalence quartile, margin and top-K diagnostics were not used for training configuration or selection. FP, if triggered, was diagnostic only.",
                    "- Existing source construction indexes cohort metadata, but this audit never builds an outer-test loader or reads outer labels, predictions or performance.",""]
    (EXPERIMENT/"IMPLEMENTATION_AUDIT.md").write_text("\n".join(implementation),encoding="utf-8")
    (EXPERIMENT/"README.md").write_text("# Class geometry, top-rank and capacity audit (seed 42)\n\n"+
        f"Completed FIT/validation-only audit. Exact terminal: `{terminal}`. See `FINAL_REPORT.md`, `RANKING_GEOMETRY_SUMMARY.csv` and frozen protocol files. Private records/checkpoints remain server-side; no outer test was accessed.\n",encoding="utf-8")
    write_json(EXPERIMENT/"VALIDATION.json",{"pass":True,"source_reproduced":True,
               "trained_fold_variant_cells":len(trained)*5,"trained_epochs":len(trained)*150,
               "selected_vloo_cases_per_variant":65,"paired_bootstrap_rows":15,
               "capacity_trajectory_rows":450,"outer_test_accessed":False,"terminal":terminal})
    print(json.dumps({"terminal":terminal,"outer_test_accessed":False}),flush=True)


if __name__=="__main__": main()
