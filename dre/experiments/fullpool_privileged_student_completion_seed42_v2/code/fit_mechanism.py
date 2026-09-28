"""FIT-meta Teacher/Student preference diagnostics; no target outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from student_core import LOCK_SHA, VARIANTS, _pairs, make_fold, new_model


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Empty FIT diagnostic")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_meta(data, model) -> dict:
    model.eval()
    result = {}
    with torch.no_grad():
        for raw in data.meta_loader:
            batch = data.move(raw, data.exp.device)
            out = model(batch)
            for i, sid in enumerate(batch["subject_id"]):
                valid = batch["channel_mask"][i].bool()
                result[sid] = -out["logits"][i][valid].detach().cpu().numpy()
    if set(result) != set(data.meta_ids):
        raise RuntimeError("FIT-meta score membership mismatch")
    return result


def mean(values) -> float | None:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    freeze = json.loads((a.output / "FIT_SELECTION_FREEZE_AUDIT.json").read_text(encoding="utf-8"))
    if freeze["lock_sha"] != LOCK_SHA or not freeze["all_models_fit_selected_before_target_outcomes"]:
        raise RuntimeError("FIT model selections are incomplete")
    variants = list(VARIANTS[:5]) + ([VARIANTS[5]] if freeze["D4_FIT_ELIGIBLE"] else [])
    # Each entry is one deterministic pair in one frozen FIT Teacher context.
    pair_rows = []
    for fold in range(1, 6):
        data = make_fold(fold, a.runtime)
        for variant in variants:
            selected = json.loads((a.runtime / "private" / variant / f"fold_{fold}" /
                                   "FIT_SELECTION.json").read_text(encoding="utf-8"))["selected"]
            model = new_model(data)
            checkpoint = (a.runtime / "private" / variant / f"fold_{fold}" /
                          selected["candidate_id"] / f"epoch_{selected['best_epoch']:02d}.pt")
            state = torch.load(checkpoint, map_location=data.exp.device, weights_only=False)
            if state["lock_sha"] != LOCK_SHA or state["variant"] != variant:
                raise RuntimeError("FIT diagnostic checkpoint identity mismatch")
            model.load_state_dict(state["model"], strict=True)
            scores = score_meta(data, model)
            for sid in data.meta_ids:
                student = np.asarray(scores[sid], dtype=np.float64)
                student = (student - student.mean()) / max(float(student.std()), 1e-6)
                for view in data.teacher[sid]:
                    t, source = view["t"], view["a"]
                    ia, ib = _pairs(len(t), fold, 0, 0, sid)
                    pt = 1 / (1 + np.exp(-(t[ia] - t[ib])))
                    pa = 1 / (1 + np.exp(-(source[ia] - source[ib])))
                    w = np.abs(pt - pa)
                    truth = t[ia] > t[ib]
                    a1_pref = source[ia] > source[ib]
                    learner = student[ia] > student[ib]
                    for j in range(len(ia)):
                        pair_rows.append((variant, fold, sid, float(view["weight"]), float(w[j]),
                                          bool(truth[j] == a1_pref[j]), bool(truth[j] == learner[j])))
            print(f"[FIT_DIAGNOSTIC] {variant} fold={fold}", flush=True)
    weights = np.asarray([row[4] for row in pair_rows if row[0] == variants[0]], dtype=float)
    q1, q2 = np.quantile(weights, [1 / 3, 2 / 3])
    if q1 >= q2:
        raise RuntimeError("Correction strata collapsed")
    def stratum(w: float) -> str:
        return "low" if w <= q1 else "medium" if w <= q2 else "high"
    def aggregate(variant: str, subset: str | None, discordant: bool | None = None) -> tuple[float | None, int]:
        grouped = defaultdict(list)
        for model, fold, sid, weight, w, a1_agrees, student_agrees in pair_rows:
            if model != variant or (subset is not None and stratum(w) != subset):
                continue
            if discordant is not None and (not a1_agrees) != discordant:
                continue
            grouped[(fold, sid)].append((weight, student_agrees))
        per_patient = [np.average([x[1] for x in values], weights=[x[0] for x in values])
                       for values in grouped.values()]
        return mean(per_patient), len(per_patient)
    agreement = []
    source_grouped = defaultdict(list)
    for model, fold, sid, weight, _w, a1_agrees, _student_agrees in pair_rows:
        if model == variants[0]:
            source_grouped[(fold, sid)].append((weight, a1_agrees))
    teacher_a1_agreement = mean(np.average([x[1] for x in values],
                                           weights=[x[0] for x in values])
                                for values in source_grouped.values())
    for variant in variants:
        all_agree, n = aggregate(variant, None)
        disagreement_agree, n_disagreement = aggregate(variant, None, True)
        agreement.append({"variant": variant, "fit_meta_patient_contexts": n,
                          "teacher_A1_pair_agreement": teacher_a1_agreement,
                          "teacher_student_pair_agreement": all_agree,
                          "teacher_student_agreement_on_teacher_A1_disagreement": disagreement_agree,
                          "meta_patients_with_disagreement_pairs": n_disagreement,
                          "correction_temperature": 1.0, "target_outcomes_used": False})
    strata = []
    for variant in variants:
        for name in ("low", "medium", "high"):
            value, n = aggregate(variant, name)
            control, _ = aggregate(VARIANTS[0], name)
            strata.append({"variant": variant, "correction_stratum": name,
                           "global_fit_meta_q1": float(q1), "global_fit_meta_q2": float(q2),
                           "fit_meta_patient_contexts": n,
                           "teacher_student_pair_agreement": value,
                           "delta_agreement_vs_D0plus": value - control if value is not None and control is not None else None,
                           "target_outcomes_used": False})
    write_csv(a.output / "TEACHER_STUDENT_PREFERENCE_AGREEMENT.csv", agreement)
    write_csv(a.output / "CORRECTION_STRATIFIED_AUDIT.csv", strata)
    print("FIT_MECHANISM_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
