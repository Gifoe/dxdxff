from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import torch

from outcome_hifos.metrics import MetricBundle, compute_patient_metrics


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device, threshold: float = 0.5) -> tuple[MetricBundle, pd.DataFrame]:
    model.eval()
    rows = []
    for batch in loader:
        model_input = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch["model_input"].items()}
        output = model(model_input, persist_diagnostics=False)
        logits = output["logits"].detach().cpu()
        probabilities = torch.sigmoid(logits)
        for index, subject_id in enumerate(batch["subject_id"]):
            rows.append(
                {
                    "subject_id": str(subject_id),
                    "center": str(batch["center"][index]),
                    "outcome": float(batch["outcome"][index]),
                    "logit": float(logits[index]),
                    "probability": float(probabilities[index]),
                }
            )
    frame = pd.DataFrame(rows)
    metrics = compute_patient_metrics(frame["outcome"].to_numpy(), frame["probability"].to_numpy(), threshold)
    return metrics, frame


__all__ = ["evaluate_model"]
