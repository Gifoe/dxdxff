"""Post-hoc aggregate-only analysis of B0 uncertainty and adapter decision changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_calibrank import CalibRank, patient_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    private_rows = []
    for fold in range(1, 6):
        with np.load(args.embeddings / f"fold{fold}.npz", allow_pickle=True) as archive:
            groups = patient_groups(archive, "test", torch.device("cpu"))
        for variant in ("B0", "B1", "B2", "B3"):
            cell = args.train / variant / f"fold{fold}"
            selected = json.loads((cell / "COMPLETE.json").read_text(encoding="utf-8"))["fold_result"]
            model = None
            if variant != "B0" and selected["selected_epoch"] > 0:
                model = CalibRank(variant)
                checkpoint = torch.load(cell / "ADAPTER_PRIVATE.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
            with torch.no_grad():
                for group in groups:
                    base_ez = torch.sigmoid(group["a"])
                    gate = (4 * base_ez * (1 - base_ez)).numpy()
                    score_nez = torch.sigmoid(-group["a"] if model is None else -model(group)[0]).numpy()
                    for index in range(len(score_nez)):
                        private_rows.append({"subject_id": group["patient_id"], "fold": fold,
                                             "variant": variant, "channel_index": index,
                                             "uncertainty": float(gate[index]),
                                             "label_nez": int(group["y_nez"][index]),
                                             "score_nez": float(score_nez[index]),
                                             "prediction_nez": int(score_nez[index] >= selected["threshold_nez"])})
    frame = pd.DataFrame(private_rows)
    base = frame[frame.variant == "B0"].drop(columns="variant")
    if len(base) != 7635 or base.subject_id.nunique() != 80:
        raise RuntimeError("Expected 7,635 channels from 80 patients")
    quantiles = np.quantile(base.uncertainty, [1 / 3, 2 / 3])
    base["channel_bin"] = np.searchsorted(quantiles, base.uncertainty, side="right")
    patient_gate = base.groupby("subject_id").uncertainty.mean()
    patient_quantiles = np.quantile(patient_gate, [1 / 3, 2 / 3])
    patient_bin = patient_gate.map(lambda value: int(np.searchsorted(patient_quantiles, value, side="right")))
    rows = []
    for variant in ("B1", "B2", "B3"):
        other = frame[frame.variant == variant]
        merged = base.merge(other, on=["subject_id", "fold", "channel_index", "label_nez"],
                            validate="one_to_one", suffixes=("_b0", "_method"))
        if len(merged) != 7635:
            raise RuntimeError(f"Channel alignment failed for {variant}")
        for bin_index, part in merged.groupby("channel_bin"):
            old = part.prediction_nez_b0.to_numpy() == part.label_nez.to_numpy()
            new = part.prediction_nez_method.to_numpy() == part.label_nez.to_numpy()
            rows.append({"level": "channel", "variant": variant, "uncertainty_tertile": int(bin_index),
                         "records": len(part), "patients": part.subject_id.nunique(),
                         "uncertainty_mean": float(part.uncertainty_b0.mean()),
                         "changed_decisions": int((part.prediction_nez_b0 != part.prediction_nez_method).sum()),
                         "corrected_decisions": int((~old & new).sum()),
                         "spoiled_decisions": int((old & ~new).sum()),
                         "mean_abs_probability_change": float(np.abs(part.score_nez_method - part.score_nez_b0).mean()),
                         "delta_patient_macro_f1": np.nan})
        reference = pd.concat([pd.read_csv(args.train / "B0" / f"fold{fold}" / "TEST_PATIENT_PRIVATE.csv")
                               for fold in range(1, 6)], ignore_index=True)
        method = pd.concat([pd.read_csv(args.train / variant / f"fold{fold}" / "TEST_PATIENT_PRIVATE.csv")
                            for fold in range(1, 6)], ignore_index=True)
        paired = method.merge(reference, on="subject_id", validate="one_to_one", suffixes=("_method", "_b0"))
        paired["uncertainty_tertile"] = paired.subject_id.map(patient_bin)
        for bin_index, part in paired.groupby("uncertainty_tertile"):
            rows.append({"level": "patient", "variant": variant, "uncertainty_tertile": int(bin_index),
                         "records": len(part), "patients": len(part),
                         "uncertainty_mean": float(patient_gate.loc[part.subject_id].mean()),
                         "changed_decisions": np.nan, "corrected_decisions": np.nan,
                         "spoiled_decisions": np.nan, "mean_abs_probability_change": np.nan,
                         "delta_patient_macro_f1": float((part.macro_f1_method - part.macro_f1_b0).mean())})
    pd.DataFrame(rows).sort_values(["variant", "level", "uncertainty_tertile"]).to_csv(args.output, index=False)
    print("UNCERTAINTY_COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
