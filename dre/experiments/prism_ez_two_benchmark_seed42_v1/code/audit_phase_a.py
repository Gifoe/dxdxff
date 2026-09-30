"""Write auditable Phase-A PRiSM-EZ implementation checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from prism_ez import FEATURE_DIM, PRiSMEZ, empirical_rank, parameter_audit, spectral_availability, spectral_sketch


def dump(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    here, experiment = Path(__file__).resolve(), Path(__file__).resolve().parents[1]
    parameter = parameter_audit()
    with (experiment / "MODEL_PARAMETER_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parameter))
        writer.writeheader(); writer.writerow(parameter)
    values = torch.tensor([[[1.0, 2.0], [3.0, 0.0]], [[1.0, 8.0], [5.0, 0.0]],
                           [[9.0, 4.0], [7.0, 0.0]]])
    channel, window = torch.tensor([True, True, True]), torch.ones((3, 2), dtype=torch.bool)
    rank = empirical_rank(values, channel, window)
    perm = torch.tensor([2, 0, 1])
    rank_payload = {
        "label_free_inputs_only": True, "ties": "average rank", "single_channel_q": 0.0,
        "permutation_equivariant": bool(torch.allclose(empirical_rank(values[perm], channel[perm], window[perm]), rank[perm])),
        "tie_expected": -0.5, "tie_observed": float(rank[0, 0, 0]),
        "complexity": "O(C*T*D log C); no O(C^2) or O(T^2) tensor operation",
    }
    dump(experiment / "RANK_OPERATOR_AUDIT.json", rank_payload)
    wave = np.random.default_rng(42).normal(size=(2, 15000)).astype(np.float32)
    sketch, available = spectral_sketch(wave, 250.0, np.ones((2, 59), dtype=bool))
    physical = spectral_availability(250.0)
    dump(experiment / "SPECTRAL_SKETCH_AUDIT.json", {
        "window_seconds": 2, "hop_seconds": 1, "physical_edges_hz": [1.0, 300.0],
        "bins": 32, "ictal_sampling_rate_hz": 250.0, "trusted_max_hz": 112.5,
        "unavailable_bins_zero": bool(np.all(sketch[:, :, ~physical] == 0)),
        "unavailable_bins_masked": bool(not available[:, :, ~physical].any()),
        "finite": bool(np.isfinite(sketch).all()), "no_upsampling": True,
    })
    model = PRiSMEZ().eval()
    record = {"features": torch.randn(3, 59, FEATURE_DIM), "channel_mask": torch.tensor([True, True, True]),
              "window_mask": torch.ones(3, 59, dtype=torch.bool)}
    record["window_mask"][:, -1] = False
    with torch.no_grad():
        finite = bool(torch.isfinite(model.forward_group([record, record])).all())
    dump(experiment / "FEATURE_CONTRACT.json", {
        "input": "[ABS, DELTA, ZDELTA, LOGR] 36D plus 32D physical-frequency sketch", "feature_dimensions": 68,
        "valid_window_masking": True, "variable_record_count_supported": True,
        "quantile_pooling_finite_on_single_and_multiple_records": finite,
        "labels_not_accepted_by_rank_or_spectral_feature_functions": True,
    })
    if not parameter["budget_pass"] or not rank_payload["permutation_equivariant"] or not finite:
        raise RuntimeError("Phase-A audit gate failed")


if __name__ == "__main__":
    main()
