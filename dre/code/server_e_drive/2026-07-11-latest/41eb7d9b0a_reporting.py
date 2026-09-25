"""Generate a fail-closed report: missing formal outputs remain pending."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_report(output_root: str | Path) -> Path:
    root = Path(output_root); reports = root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    statistics = root / "statistics" / "multiseed_overall.csv"
    reproduction = root / "audit" / "seed42_reference_reproduction.json"
    sections = ["# Task 1 Confirmatory Experiment Report", "", "## Protocol and Cohort", "NEZ=1 and EZ=0. The primary metric is patient-equal Macro-F1. CDEL is frozen probability fusion: 0.80 PRQ-Net + 0.20 BCR-Net."]
    if reproduction.is_file():
        payload = json.loads(reproduction.read_text(encoding="utf-8")); sections += ["", "## Seed-42 Reproduction", "```json", json.dumps(payload, indent=2), "```"]
    if statistics.is_file():
        table = pd.read_csv(statistics); sections += ["", "## Multi-Seed Patient-Wise Five-Fold Results", table.to_markdown(index=False)]
    else:
        sections += ["", "## Formal Results", "PENDING: no complete formal 42/52/62 multi-seed result exists. Smoke, dry-run, partial, and failed results are excluded."]
    sections += ["", "## Limitations", "No seed, LOCO, bootstrap, threshold-stability, or efficiency result is claimed until its completed inputs are present in the registry."]
    target = reports / "TASK1_CONFIRMATORY_EXPERIMENT_REPORT.md"; target.write_text("\n".join(sections) + "\n", encoding="utf-8"); return target
