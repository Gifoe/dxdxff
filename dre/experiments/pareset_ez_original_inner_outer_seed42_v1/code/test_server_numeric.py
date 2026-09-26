"""Check the historical server head's single-seizure and zero-variance gradient."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


def main() -> None:
    server_root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    sys.path.insert(0, str(server_root))
    from neuroez_c.p23_seizure_tail import P23CrossSeizureTailEvidence

    cases = {}
    for name, seizures in (("single", 1), ("constant_three", 3)):
        torch.manual_seed(17)
        head = P23CrossSeizureTailEvidence(model_dim=32, feature_mode="atc7")
        head.train()
        embedding = torch.zeros(1, seizures, 3, 32, requires_grad=True)
        valid_seizures = torch.ones(1, seizures, dtype=torch.bool)
        valid_channels = torch.ones(1, seizures, 3, dtype=torch.bool)
        result = head(embedding, valid_seizures, valid_channels)
        value = result["u_seizure"].sum() + result["seizure_nez_probability_std"].sum()
        value.backward()
        cases[name] = {
            "value_finite": bool(torch.isfinite(value)),
            "input_gradient_finite": bool(torch.isfinite(embedding.grad).all()),
            "parameter_gradients_finite": all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head.parameters()
            ),
        }
    outcome = {
        "server_module": "neuroez_c.p23_seizure_tail.P23CrossSeizureTailEvidence",
        "cases": cases,
        "pass": all(all(test.values()) for test in cases.values()),
        "interpretation": "Only this server ATC7 tail path is tested; no claim for every historical PRQ/BCR variant.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome))
    if not outcome["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
